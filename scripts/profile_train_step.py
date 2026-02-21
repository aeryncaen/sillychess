import argparse
import gc
import resource
import sys
from typing import Dict

import torch

from sillychess.model import TwoStageTransformerModel
from sillychess.san_features import FEATURE_SIZES


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def bytes_to_mb(n_bytes: int) -> float:
    return float(n_bytes) / (1024.0 * 1024.0)


def current_rss_mb() -> float:
    try:
        import psutil  # type: ignore

        return bytes_to_mb(psutil.Process().memory_info().rss)
    except Exception:
        r = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # ru_maxrss units differ by OS:
        # - macOS: bytes
        # - Linux: kilobytes
        if sys.platform == "darwin":
            return r / (1024.0 * 1024.0)
        return r / 1024.0


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return auto_device()
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("requested CUDA device but CUDA is not available")
        return requested
    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError("requested MPS device but MPS is not available")
        return requested
    if requested == "cpu":
        return requested
    raise ValueError(f"unknown --device value: {requested}")


def make_batch(
    feature_sizes: Dict[str, int],
    batch_size: int,
    block_size: int,
    device: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    x = {
        name: torch.randint(0, size, (batch_size, block_size), dtype=torch.long, device=device)
        for name, size in feature_sizes.items()
    }
    y = {
        name: torch.randint(0, size, (batch_size, block_size), dtype=torch.long, device=device)
        for name, size in feature_sizes.items()
    }
    return x, y



def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    x: dict[str, torch.Tensor],
    y: dict[str, torch.Tensor],
    grad_clip: float,
) -> torch.Tensor:
    out = model(x)
    losses = []
    for name, logits in out.items():
        losses.append(
            torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y[name].reshape(-1),
            )
        )
    loss = sum(losses) / len(losses)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--w-dim", type=int, default=48)
    parser.add_argument("--rows-per-feature", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--with-profiler", action="store_true")
    parser.add_argument("--profile-row-limit", type=int, default=25)
    args = parser.parse_args()
    device = resolve_device(args.device)

    torch.set_num_threads(args.num_threads)

    model = TwoStageTransformerModel(
        block_size=args.block_size,
        w_dim=args.w_dim,
        rows_per_feature=args.rows_per_feature,
        n_layer=args.n_layer,
        dropout=args.dropout,
        feature_sizes=FEATURE_SIZES,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    total_rows = len(FEATURE_SIZES) * args.rows_per_feature
    total_params, trainable_params = count_parameters(model)
    print("setup:")
    print(
        f"  device={device} batch_size={args.batch_size} block_size={args.block_size}"
        f" w_dim={args.w_dim} rows_per_feature={args.rows_per_feature} total_rows={total_rows}"
    )
    print(f"  n_layer={args.n_layer}")
    print(f"  params_total={total_params:,} params_trainable={trainable_params:,}")
    print(f"  tokens: [B, T, {total_rows}, {args.w_dim}]")

    x, y = make_batch(FEATURE_SIZES, args.batch_size, args.block_size, device)

    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    print(f"rss_before_step={current_rss_mb():.1f} MB")

    if not args.with_profiler:
        with torch.no_grad():
            _ = model(x)
        print(f"rss_after_warmup_forward={current_rss_mb():.1f} MB")

        loss = train_step(model, optimizer, x, y, args.grad_clip)
        print(f"loss={loss.item():.4f}")
        print(f"rss_after_train_step={current_rss_mb():.1f} MB")
        if device.startswith("cuda") and torch.cuda.is_available():
            print(
                "cuda_peak_allocated="
                f"{bytes_to_mb(torch.cuda.max_memory_allocated()):.1f} MB"
            )
            print(
                "cuda_peak_reserved="
                f"{bytes_to_mb(torch.cuda.max_memory_reserved()):.1f} MB"
            )
        return

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.startswith("cuda") and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        loss = train_step(model, optimizer, x, y, args.grad_clip)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    print(f"loss={loss.item():.4f}")
    print(f"rss_after_train_step={current_rss_mb():.1f} MB")
    if device.startswith("cuda") and torch.cuda.is_available():
        print(
            "cuda_peak_allocated="
            f"{bytes_to_mb(torch.cuda.max_memory_allocated()):.1f} MB"
        )
        print(
            "cuda_peak_reserved="
            f"{bytes_to_mb(torch.cuda.max_memory_reserved()):.1f} MB"
        )

    print("\nTop ops by self_cpu_memory_usage:")
    print(
        prof.key_averages().table(
            sort_by="self_cpu_memory_usage",
            row_limit=args.profile_row_limit,
        )
    )
    print("\nTop ops by cpu_memory_usage:")
    print(
        prof.key_averages().table(
            sort_by="cpu_memory_usage",
            row_limit=args.profile_row_limit,
        )
    )


if __name__ == "__main__":
    main()

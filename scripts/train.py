import argparse
import math

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sillychess.dataset import (
    BucketBatchSampler,
    CachedChessDataset,
    ChessMoveDataset,
    iter_jsonl_games,
    iter_pgn_games,
)
from sillychess.eval import eval_legality, eval_loss, eval_loss_uci, eval_legality_uci, split_eval
from sillychess.model import PlainTransformerModel, TwoStageTransformerModel
from sillychess.san_features import FEATURE_SIZES
from sillychess.uci_vocab import UCI_VOCAB_SIZE, UCI_PLAIN_VOCAB_SIZE, PAD_ID


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_games(path, fmt, perspective, winner_only=False, max_games=None):
    if fmt == "jsonl":
        if perspective == "both":
            return list(iter_jsonl_games(path, max_games=max_games, self_color="white")) + list(
                iter_jsonl_games(path, max_games=max_games, self_color="black")
            )
        return list(
            iter_jsonl_games(
                path, max_games=max_games, self_color=perspective, winner_only=winner_only
            )
        )
    if fmt == "pgn":
        if perspective == "both":
            return list(iter_pgn_games(path, max_games=max_games, self_color="white")) + list(
                iter_pgn_games(path, max_games=max_games, self_color="black")
            )
        return list(
            iter_pgn_games(
                path, max_games=max_games, self_color=perspective, winner_only=winner_only
            )
        )
    raise ValueError("unknown data format")


def auto_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(requested):
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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--format", choices=["jsonl", "pgn"], default="jsonl")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine", "wsd"], default="constant")
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--lr-min-ratio", type=float, default=0.1,
                        help="min LR as fraction of peak LR (for cosine and WSD)")
    parser.add_argument("--wsd-stable-fraction", type=float, default=0.7,
                        help="fraction of steps at peak LR before decay (WSD only)")
    parser.add_argument("--wsd-decay", choices=["cosine", "linear", "rsqrt"], default="cosine",
                        help="decay shape for WSD schedule")
    parser.add_argument("--w-dim", type=int, default=48)
    parser.add_argument("--rows-per-feature", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--perspective", choices=["white", "black", "both", "winner"], default="winner"
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-legality-games", type=int, default=50)
    parser.add_argument("--uci", action="store_true",
                        help="use single UCI move head on 2D feature-attention model")
    parser.add_argument("--uci-plain", action="store_true",
                        help="plain 1D transformer: UCI tokens in, UCI tokens out")
    parser.add_argument("--n-head", type=int, default=4,
                        help="attention heads for --uci-plain (d_model must be divisible)")
    parser.add_argument("--lerp", action="store_true",
                        help="enable CausalLerp in --uci-plain blocks")
    parser.add_argument("--feat-attn", action="store_true",
                        help="replace SwiGLU with feature attention in --uci-plain blocks")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-model", default="model.pt")
    args = parser.parse_args()

    device = resolve_device(args.device)
    if args.uci_plain:
        args.uci = True

    print("init: resolving dataset...")
    target_windows = args.steps * args.batch_size
    winner_only = args.perspective == "winner"

    if args.cache_dir:
        dataset = CachedChessDataset(
            args.cache_dir,
            block_size=args.block_size,
            max_games=None,
            target_windows=target_windows,
            uci_plain=args.uci_plain,
        )
        print(
            f"init: dataset ready — "
            f"shards={dataset.loaded_shards} games={dataset.loaded_games}"
        )
        if dataset.uci_plain:
            print(f"init: buckets={dataset.buckets} dropped={dataset.dropped_games}")

        # Split train/eval
        if len(dataset.sequences) >= 10:
            train_dataset, eval_dataset = dataset.split_eval(eval_fraction=0.1)
            print(f"init: train={len(train_dataset)} eval={len(eval_dataset)}")
        else:
            train_dataset = dataset
            eval_dataset = None
            print("init: too few games to split, no eval set")

        if dataset.uci_plain:
            sampler = BucketBatchSampler(train_dataset.bucket_ids, args.batch_size, shuffle=True)
            loader = DataLoader(train_dataset, batch_sampler=sampler)
        else:
            loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    else:
        if not args.data:
            raise ValueError("--data is required when not using --cache-dir")
        games = load_games(
            args.data, args.format, args.perspective,
            winner_only=winner_only, max_games=None,
        )
        dataset = ChessMoveDataset(games, args.block_size, target_windows=target_windows)
        print("init: dataset ready")
        train_dataset = dataset
        eval_dataset = None
        loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    if args.uci_plain:
        args.uci = True  # --uci-plain implies --uci

    print("init: building model...")
    if args.uci_plain:
        vocab_size = UCI_PLAIN_VOCAB_SIZE
        # Block size for full-game mode = largest bucket - 1 (input is bucket_len - 1)
        block_size = max(dataset.buckets) - 1 if dataset.uci_plain else args.block_size
        model = PlainTransformerModel(
            block_size=block_size,
            d_model=args.w_dim,
            n_head=args.n_head,
            n_layer=args.n_layer,
            vocab_size=vocab_size,
            dropout=args.dropout,
            use_lerp=args.lerp,
            use_feat_attn=args.feat_attn,
        ).to(device)
        flags = []
        if args.lerp:
            flags.append("lerp")
        if args.feat_attn:
            flags.append("feat_attn")
        flag_str = f" +{'+'.join(flags)}" if flags else ""
        print(f"init: model ready (plain 1D, d_model={args.w_dim}, n_head={args.n_head}, vocab={vocab_size}{flag_str})")
    else:
        model = TwoStageTransformerModel(
            block_size=args.block_size,
            w_dim=args.w_dim,
            rows_per_feature=args.rows_per_feature,
            n_layer=args.n_layer,
            dropout=args.dropout,
            feature_sizes=FEATURE_SIZES,
            uci_vocab_size=UCI_VOCAB_SIZE if args.uci else None,
        ).to(device)
        print(f"init: model ready (2D, uci_mode={model.uci_mode})")

    total_params, trainable_params = count_parameters(model)
    if hasattr(dataset, "sequences") and dataset.sequences is not None:
        game_count = len(dataset.sequences)
    else:
        game_count = "unknown"
    print(
        "setup:",
        f"device={device}",
        f"games={game_count}",
        f"samples={len(dataset)}",
        f"params_total={total_params:,}",
        f"params_trainable={trainable_params:,}",
    )
    if args.uci_plain:
        print(
            "model:",
            f"block_size={block_size}",
            f"d_model={args.w_dim}",
            f"n_head={args.n_head}",
            f"n_layer={args.n_layer}",
            f"vocab_size={vocab_size}",
        )
    else:
        feature_classes = sum(FEATURE_SIZES.values())
        total_rows = len(FEATURE_SIZES) * args.rows_per_feature
        print(
            "model:",
            f"block_size={args.block_size}",
            f"w_dim={args.w_dim}",
            f"rows_per_feature={args.rows_per_feature}",
            f"total_rows={total_rows}",
            f"n_layer={args.n_layer}",
            f"features={len(FEATURE_SIZES)}",
            f"feature_classes={feature_classes}",
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def get_lr(step):
        """Returns LR multiplier (0..1) for the given step."""
        # Warmup
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        if args.lr_schedule == "constant":
            return 1.0
        if args.lr_schedule == "cosine":
            # Cosine decay from peak to min after warmup
            decay_steps = args.steps - args.warmup_steps
            progress = (step - args.warmup_steps) / max(1, decay_steps)
            progress = min(progress, 1.0)
            return args.lr_min_ratio + (1.0 - args.lr_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        if args.lr_schedule == "wsd":
            # Warmup-Stable-Decay
            stable_end = args.warmup_steps + int(args.wsd_stable_fraction * (args.steps - args.warmup_steps))
            if step < stable_end:
                return 1.0  # stable phase
            decay_steps = args.steps - stable_end
            progress = (step - stable_end) / max(1, decay_steps)
            progress = min(progress, 1.0)
            if args.wsd_decay == "cosine":
                mult = 0.5 * (1.0 + math.cos(math.pi * progress))
            elif args.wsd_decay == "linear":
                mult = 1.0 - progress
            elif args.wsd_decay == "rsqrt":
                mult = 1.0 / math.sqrt(1.0 + progress * (1.0 / args.lr_min_ratio**2 - 1.0))
            else:
                mult = 1.0
            return args.lr_min_ratio + (1.0 - args.lr_min_ratio) * mult
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)
    print(f"schedule: {args.lr_schedule} warmup={args.warmup_steps} lr={args.lr} min_ratio={args.lr_min_ratio}")
    if args.lr_schedule == "wsd":
        print(f"  wsd: stable_fraction={args.wsd_stable_fraction} decay={args.wsd_decay}")

    model.train()

    step = 0
    data_iter = iter(loader)
    pbar = tqdm(
        total=args.steps,
        desc="train",
        unit="step",
        miniters=1,
        mininterval=0.0,
    )
    try:
        while step < args.steps:
            try:
                feat_x, feat_y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                feat_x, feat_y = next(data_iter)

            feat_x = {name: tensor.to(device) for name, tensor in feat_x.items()}
            feat_y = {name: tensor.to(device) for name, tensor in feat_y.items()}

            outputs = model(feat_x)

            if args.uci_plain:
                targets = feat_y["uci_move"]
                loss = torch.nn.functional.cross_entropy(
                    outputs.reshape(-1, outputs.size(-1)),
                    targets.reshape(-1),
                    ignore_index=PAD_ID,
                )
            elif args.uci:
                pad_mask = (feat_y["step"] != 0).float()
                uci_targets = feat_y["uci_move"]
                valid = (uci_targets >= 0) & (pad_mask > 0)
                uci_targets_safe = uci_targets.clamp(min=0)
                raw = torch.nn.functional.cross_entropy(
                    outputs.reshape(-1, outputs.size(-1)),
                    uci_targets_safe.reshape(-1),
                    reduction="none",
                )
                raw = raw.view(pad_mask.shape)
                loss = (raw * valid.float()).sum() / valid.float().sum().clamp_min(1.0)
            else:
                pad_mask = (feat_y["step"] != 0).float()
                denom = pad_mask.sum().clamp_min(1.0)
                # Per-feature heads
                feature_losses = []
                for name, logits in outputs.items():
                    targets = feat_y[name]
                    raw = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1),
                        reduction="none",
                    )
                    raw = raw.view(pad_mask.shape)
                    feature_losses.append((raw * pad_mask).sum() / denom)
                loss = sum(feature_losses) / len(feature_losses)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = None
            if args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            if grad_norm is not None:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    grad=f"{float(grad_norm):.3f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )
            else:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )
            pbar.update(1)
            step += 1

            # --- Eval ---
            if eval_dataset is not None and step % args.eval_every == 0:
                if args.uci_plain:
                    val = eval_loss_uci(model, eval_dataset, args.batch_size, device)
                    tqdm.write(f"  eval step={step}: val_loss={val:.4f}")
                elif args.uci:
                    val = eval_loss_uci(model, eval_dataset, args.batch_size, device)
                    leg, leg_total = eval_legality_uci(
                        model, eval_dataset.sequences, args.block_size, device,
                        max_games=args.eval_legality_games,
                    )
                    leg_pct = 100.0 * leg / max(1, leg_total)
                    tqdm.write(
                        f"  eval step={step}: val_loss={val:.4f} "
                        f"legality={leg}/{leg_total} ({leg_pct:.1f}%)"
                    )
                else:
                    val = eval_loss(model, eval_dataset, args.batch_size, device)
                    leg, leg_total = eval_legality(
                        model, eval_dataset.sequences, args.block_size, device,
                        max_games=args.eval_legality_games,
                    )
                    leg_pct = 100.0 * leg / max(1, leg_total)
                    tqdm.write(
                        f"  eval step={step}: val_loss={val:.4f} "
                        f"legality={leg}/{leg_total} ({leg_pct:.1f}%)"
                    )
    finally:
        pbar.close()

    torch.save({"model": model.state_dict()}, args.save_model)


if __name__ == "__main__":
    main()

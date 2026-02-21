"""Eval utilities: dataset splitting and move legality checking."""

import chess
import torch
from torch.utils.data import DataLoader

from sillychess.dataset import CachedChessDataset, _windows_for_length
from sillychess.san_features import FEATURE_IDS, FEATURE_SPECS
from sillychess.uci_vocab import UCI_MOVES


def split_eval(dataset, eval_fraction=0.1):
    """Split a dataset's games into train and eval.

    Takes the first ``eval_fraction`` of games for eval, rebuilds window
    indices for both.  Returns (train_dataset, eval_dataset).
    """
    n_games = len(dataset.sequences)
    n_eval = max(1, int(n_games * eval_fraction))

    def _make_split(sequences):
        ds = CachedChessDataset.__new__(CachedChessDataset)
        ds.block_size = dataset.block_size
        ds.sequences = sequences
        ds.loaded_shards = 0
        ds.loaded_games = len(sequences)
        ds._loaded_windows = 0
        ds._target_windows = None
        ds.index = []
        for g_idx, seq in enumerate(sequences):
            length = len(seq[next(iter(FEATURE_IDS))])
            max_start = max(0, length - dataset.block_size - 1)
            for start in range(max_start + 1):
                ds.index.append((g_idx, start))
        return ds

    eval_ds = _make_split(dataset.sequences[:n_eval])
    train_ds = _make_split(dataset.sequences[n_eval:])
    return train_ds, eval_ds


def eval_loss(model, eval_dataset, batch_size, device, max_batches=16):
    """Compute average cross-entropy loss on a random subset of the eval set.

    ``max_batches`` caps how many batches we evaluate (default 16) so eval
    doesn't dominate training time.
    """
    loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, (feat_x, feat_y) in enumerate(loader):
            if i >= max_batches:
                break
            feat_x = {name: t.to(device) for name, t in feat_x.items()}
            feat_y = {name: t.to(device) for name, t in feat_y.items()}

            outputs = model(feat_x)
            pad_mask = (feat_y["step"] != 0).float()
            denom = pad_mask.sum().clamp_min(1.0)

            batch_loss = 0.0
            for name, logits in outputs.items():
                targets = feat_y[name]
                raw = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction="none",
                )
                raw = raw.view(pad_mask.shape)
                batch_loss += (raw * pad_mask).sum() / denom

            batch_loss = batch_loss / len(outputs)
            n_tokens = denom.item()
            total_loss += batch_loss.item() * n_tokens
            total_tokens += n_tokens

    model.train()
    if total_tokens == 0:
        return float("nan")
    return total_loss / total_tokens


# Reverse lookups for legality checking
_FROM_SQUARES = FEATURE_SPECS["from_square"]
_TO_SQUARES = FEATURE_SPECS["to_square"]
_PROMOTIONS = FEATURE_SPECS["promotion"]
_PROMO_MAP = {"N": chess.KNIGHT, "B": chess.BISHOP, "R": chess.ROOK, "Q": chess.QUEEN}


def _id_to_move(from_id, to_id, promo_id):
    """Convert predicted feature IDs to a chess.Move, or None if invalid."""
    from_name = _FROM_SQUARES[from_id] if from_id < len(_FROM_SQUARES) else "NULL"
    to_name = _TO_SQUARES[to_id] if to_id < len(_TO_SQUARES) else "NULL"
    if from_name == "NULL" or to_name == "NULL":
        return None
    try:
        from_sq = chess.parse_square(from_name)
        to_sq = chess.parse_square(to_name)
    except ValueError:
        return None
    promo_name = _PROMOTIONS[promo_id] if promo_id < len(_PROMOTIONS) else "NULL"
    promotion = _PROMO_MAP.get(promo_name)
    return chess.Move(from_sq, to_sq, promotion=promotion)


def eval_legality(model, eval_sequences, block_size, device, max_games=50,
                   batch_size=64):
    """Check move legality on complete eval games (batched).

    Replays each game on a board.  Collects all (game_idx, time_step) pairs,
    batches them, runs model in chunks of ``batch_size``, and checks whether
    the top-1 predicted from_square/to_square/promotion is a legal move.

    Returns (legal_count, total_count).
    """
    model.eval()
    seqs = eval_sequences[:max_games]

    # --- 1. Build all windows and record game/time metadata ---
    windows = []  # list of {name: tensor[T']} dicts (variable length)
    meta = []     # (game_idx, time_step) for board replay
    for g_idx, game_seq in enumerate(seqs):
        seq_len = len(game_seq["piece"])
        if seq_len < 2:
            continue
        for t in range(seq_len - 1):
            start = max(0, t + 1 - block_size)
            window = {}
            for name in FEATURE_IDS:
                window[name] = game_seq[name][start:t + 1]
            windows.append(window)
            meta.append((g_idx, t))

    if not windows:
        model.train()
        return 0, 0

    # --- 2. Run batched inference ---
    pred_from_all = []
    pred_to_all = []
    pred_promo_all = []
    feature_names = list(FEATURE_IDS.keys())

    with torch.no_grad():
        for b_start in range(0, len(windows), batch_size):
            batch_windows = windows[b_start:b_start + batch_size]
            # Pad to max length in this batch
            max_len = max(len(w["piece"]) for w in batch_windows)
            feat_x = {}
            for name in feature_names:
                padded = []
                for w in batch_windows:
                    vals = w[name]
                    pad_len = max_len - len(vals)
                    if pad_len > 0:
                        padded.append(vals + [0] * pad_len)
                    else:
                        padded.append(vals)
                feat_x[name] = torch.tensor(padded, dtype=torch.long, device=device)

            out = model(feat_x)
            # Gather last real position for each window
            lengths = [len(w["piece"]) for w in batch_windows]
            last_idxs = torch.tensor([l - 1 for l in lengths], device=device)
            batch_idx = torch.arange(len(batch_windows), device=device)

            pred_from_all.append(out["from_square"][batch_idx, last_idxs].argmax(-1).cpu())
            pred_to_all.append(out["to_square"][batch_idx, last_idxs].argmax(-1).cpu())
            pred_promo_all.append(out["promotion"][batch_idx, last_idxs].argmax(-1).cpu())

    pred_from_all = torch.cat(pred_from_all)
    pred_to_all = torch.cat(pred_to_all)
    pred_promo_all = torch.cat(pred_promo_all)

    # --- 3. Replay boards and check legality ---
    # Rebuild boards per-game up to each timestep
    boards = {}  # game_idx -> chess.Board (advanced incrementally)
    next_t = {}  # game_idx -> next expected t (for incremental replay)
    legal = 0
    total = 0

    for i, (g_idx, t) in enumerate(meta):
        # Initialize board for this game if needed
        if g_idx not in boards:
            boards[g_idx] = chess.Board()
            next_t[g_idx] = 0

        board = boards[g_idx]
        # Advance board to position t (replay moves next_t..t-1)
        game_seq = seqs[g_idx]
        while next_t[g_idx] < t:
            tt = next_t[g_idx]
            try:
                actual_from = _FROM_SQUARES[game_seq["from_square"][tt]]
                actual_to = _TO_SQUARES[game_seq["to_square"][tt]]
                actual_promo_name = _PROMOTIONS[game_seq["promotion"][tt]]
                actual_promotion = _PROMO_MAP.get(actual_promo_name)
                actual_move = chess.Move(
                    chess.parse_square(actual_from),
                    chess.parse_square(actual_to),
                    promotion=actual_promotion,
                )
                board.push(actual_move)
            except (ValueError, AssertionError):
                break
            next_t[g_idx] = tt + 1

        # Check predicted move legality
        move = _id_to_move(
            pred_from_all[i].item(),
            pred_to_all[i].item(),
            pred_promo_all[i].item(),
        )
        if move is not None and move in board.legal_moves:
            legal += 1
        total += 1

    model.train()
    return legal, total


# ---------------------------------------------------------------------------
# UCI-mode eval
# ---------------------------------------------------------------------------

def _uci_to_chess_move(uci_str):
    """Parse a UCI string like 'e2e4' or 'e7e8q' into chess.Move, or None."""
    if uci_str is None:
        return None
    try:
        return chess.Move.from_uci(uci_str)
    except (ValueError, chess.InvalidMoveError):
        return None


def eval_loss_uci(model, eval_dataset, batch_size, device, max_batches=16):
    """Eval loss for UCI-mode model (single move head).

    Works for both windowed (has 'step' key) and full-game (PAD_ID=0) modes.
    """
    from sillychess.uci_vocab import PAD_ID
    uci_plain = getattr(eval_dataset, 'uci_plain', False)
    if uci_plain:
        from sillychess.dataset import BucketBatchSampler
        sampler = BucketBatchSampler(eval_dataset.bucket_ids, batch_size, shuffle=False)
        loader = DataLoader(eval_dataset, batch_sampler=sampler)
    else:
        loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, (feat_x, feat_y) in enumerate(loader):
            if i >= max_batches:
                break
            feat_x = {name: t.to(device) for name, t in feat_x.items()}
            feat_y = {name: t.to(device) for name, t in feat_y.items()}

            logits = model(feat_x)  # (B, T, vocab_size)
            targets = feat_y["uci_move"]

            if uci_plain:
                # Full-game: use ignore_index for PAD
                batch_loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=PAD_ID,
                )
                n_tokens = (targets != PAD_ID).sum().item()
            else:
                # Windowed: mask by step != 0 and valid UCI targets
                pad_mask = feat_y["step"] != 0
                valid = (targets >= 0) & pad_mask
                targets_safe = targets.clamp(min=0)
                raw = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets_safe.reshape(-1),
                    reduction="none",
                )
                raw = raw.view(valid.shape)
                n_valid = valid.float().sum().clamp_min(1.0)
                batch_loss = (raw * valid.float()).sum() / n_valid
                n_tokens = n_valid.item()

            if n_tokens > 0:
                total_loss += batch_loss.item() * n_tokens
                total_tokens += n_tokens

    model.train()
    if total_tokens == 0:
        return float("nan")
    return total_loss / total_tokens


def eval_legality_uci(model, eval_sequences, block_size, device, max_games=50,
                      batch_size=64):
    """Legality eval for UCI-mode model (batched).

    Replays games, batches positions, checks whether the model's top-1
    predicted UCI move is legal on the current board.
    """
    model.eval()
    seqs = eval_sequences[:max_games]

    # --- 1. Build windows ---
    windows = []
    meta = []
    for g_idx, game_seq in enumerate(seqs):
        seq_len = len(game_seq["piece"])
        if seq_len < 2:
            continue
        has_uci = "uci_move" in game_seq
        for t in range(seq_len - 1):
            start = max(0, t + 1 - block_size)
            window = {}
            for name in FEATURE_IDS:
                window[name] = game_seq[name][start:t + 1]
            if has_uci:
                window["uci_move"] = game_seq["uci_move"][start:t + 1]
            windows.append(window)
            meta.append((g_idx, t))

    if not windows:
        model.train()
        return 0, 0

    # --- 2. Batched inference ---
    pred_uci_ids = []
    all_names = list(FEATURE_IDS.keys())
    if windows[0].get("uci_move") is not None:
        all_names.append("uci_move")

    with torch.no_grad():
        for b_start in range(0, len(windows), batch_size):
            batch_windows = windows[b_start:b_start + batch_size]
            max_len = max(len(w["piece"]) for w in batch_windows)
            feat_x = {}
            for name in all_names:
                padded = []
                for w in batch_windows:
                    vals = w[name]
                    pad_len = max_len - len(vals)
                    if pad_len > 0:
                        padded.append(vals + [0] * pad_len)
                    else:
                        padded.append(vals)
                feat_x[name] = torch.tensor(padded, dtype=torch.long, device=device)

            logits = model(feat_x)  # (B, T, vocab_size)
            lengths = [len(w["piece"]) for w in batch_windows]
            last_idxs = torch.tensor([l - 1 for l in lengths], device=device)
            batch_idx = torch.arange(len(batch_windows), device=device)

            pred_uci_ids.append(logits[batch_idx, last_idxs].argmax(-1).cpu())

    pred_uci_ids = torch.cat(pred_uci_ids)

    # --- 3. Replay boards + check legality ---
    boards = {}
    next_t = {}
    legal = 0
    total = 0

    for i, (g_idx, t) in enumerate(meta):
        if g_idx not in boards:
            boards[g_idx] = chess.Board()
            next_t[g_idx] = 0

        board = boards[g_idx]
        game_seq = seqs[g_idx]
        while next_t[g_idx] < t:
            tt = next_t[g_idx]
            try:
                actual_from = _FROM_SQUARES[game_seq["from_square"][tt]]
                actual_to = _TO_SQUARES[game_seq["to_square"][tt]]
                actual_promo_name = _PROMOTIONS[game_seq["promotion"][tt]]
                actual_promotion = _PROMO_MAP.get(actual_promo_name)
                actual_move = chess.Move(
                    chess.parse_square(actual_from),
                    chess.parse_square(actual_to),
                    promotion=actual_promotion,
                )
                board.push(actual_move)
            except (ValueError, AssertionError):
                break
            next_t[g_idx] = tt + 1

        uci_id = pred_uci_ids[i].item()
        uci_str = UCI_MOVES[uci_id] if 0 <= uci_id < len(UCI_MOVES) else None
        move = _uci_to_chess_move(uci_str)
        if move is not None and move in board.legal_moves:
            legal += 1
        total += 1

    model.train()
    return legal, total

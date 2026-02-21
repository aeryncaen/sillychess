"""Eval utilities: loss, accuracy, and move legality checking."""

import chess
import numpy as np
import torch
from torch.utils.data import DataLoader

from sillychess.dataset import BucketBatchSampler
from sillychess.san_features import FEATURE_IDS, FEATURE_SPECS
from sillychess.uci_vocab import UCI_MOVES, PAD_ID, BOG_ID, MOVE_OFFSET


def eval_loss(model, eval_dataset, batch_size, device, max_batches=16):
    """Compute average cross-entropy loss on a random subset of the eval set.

    For non-UCI (per-feature head) models.
    """
    sampler = BucketBatchSampler(eval_dataset.bucket_ids, batch_size, shuffle=False)
    loader = DataLoader(eval_dataset, batch_sampler=sampler)
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


def eval_legality(model, eval_sequences, device, max_games=50):
    """Check move legality on complete eval games (per-feature head model).

    Runs one forward pass per game on the full sequence, then replays
    the board and checks top-1 predicted from/to/promotion at each step.

    Returns (legal_count, total_count).
    """
    model.eval()
    seqs = eval_sequences[:max_games]
    legal = 0
    total = 0

    with torch.no_grad():
        for game_seq in seqs:
            seq_len = len(game_seq["piece"])
            if seq_len < 2:
                continue

            # Build full-game input (batch of 1)
            feat_x = {}
            for name in FEATURE_IDS:
                arr = np.asarray(game_seq[name], dtype=np.int64)
                feat_x[name] = torch.from_numpy(arr).unsqueeze(0).to(device)

            out = model(feat_x)  # dict of {name: (1, T, n_classes)}

            # Replay board and check legality at each position
            board = chess.Board()
            for t in range(seq_len - 1):
                from_id = out["from_square"][0, t].argmax(-1).item()
                to_id = out["to_square"][0, t].argmax(-1).item()
                promo_id = out["promotion"][0, t].argmax(-1).item()

                move = _id_to_move(from_id, to_id, promo_id)
                if move is not None and move in board.legal_moves:
                    legal += 1
                total += 1

                # Advance board with actual move
                try:
                    actual_from = _FROM_SQUARES[game_seq["from_square"][t]]
                    actual_to = _TO_SQUARES[game_seq["to_square"][t]]
                    actual_promo_name = _PROMOTIONS[game_seq["promotion"][t]]
                    actual_promotion = _PROMO_MAP.get(actual_promo_name)
                    actual_move = chess.Move(
                        chess.parse_square(actual_from),
                        chess.parse_square(actual_to),
                        promotion=actual_promotion,
                    )
                    board.push(actual_move)
                except (ValueError, AssertionError):
                    break

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

    Works for both uci_plain (PAD_ID masking) and composite (step masking).
    """
    uci_plain = getattr(eval_dataset, 'uci_plain', False)
    sampler = BucketBatchSampler(eval_dataset.bucket_ids, batch_size, shuffle=False)
    loader = DataLoader(eval_dataset, batch_sampler=sampler)
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for i, (feat_x, feat_y) in enumerate(loader):
            if i >= max_batches:
                break
            feat_x = {name: t.to(device) for name, t in feat_x.items()}
            feat_y = {name: t.to(device) for name, t in feat_y.items()}

            logits = model(feat_x)  # (B, T, vocab_size)
            targets = feat_y["uci_move"]
            preds = logits.argmax(dim=-1)

            if uci_plain:
                mask = targets != PAD_ID
                batch_loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=PAD_ID,
                )
                n_tokens = mask.sum().item()
                n_correct = ((preds == targets) & mask).sum().item()
            else:
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
                n_correct = ((preds == targets_safe) & valid).sum().item()

            if n_tokens > 0:
                total_loss += batch_loss.item() * n_tokens
                total_correct += n_correct
                total_tokens += n_tokens

    model.train()
    if total_tokens == 0:
        return float("nan"), 0.0
    return total_loss / total_tokens, total_correct / total_tokens


def eval_legality_uci(model, eval_sequences, device, max_games=50):
    """Legality eval for UCI-mode model.

    Runs one forward pass per game on the full sequence.  For plain models,
    builds the BOG + offset token sequence.  For composite models, passes
    the feature columns directly.

    Returns (legal_count, total_count).
    """
    model.eval()
    seqs = eval_sequences[:max_games]
    is_plain = getattr(model, 'embed_mode', 'plain') == 'plain'
    legal = 0
    total = 0

    with torch.no_grad():
        for game_seq in seqs:
            first_feat = next(iter(FEATURE_IDS))
            seq_len = len(game_seq[first_feat])
            if seq_len < 2:
                continue

            if is_plain:
                # Build BOG + moves token sequence
                moves = game_seq["uci_move"]
                n = len(moves)
                tokens = np.empty(n + 1, dtype=np.int64)
                tokens[0] = BOG_ID
                tokens[1:] = np.asarray(moves, dtype=np.int64) + MOVE_OFFSET
                feat_x = {"uci_move": torch.from_numpy(tokens).unsqueeze(0).to(device)}
            else:
                feat_x = {}
                all_names = list(FEATURE_IDS.keys())
                if "uci_move" in game_seq:
                    all_names.append("uci_move")
                for name in all_names:
                    arr = np.asarray(game_seq[name], dtype=np.int64)
                    feat_x[name] = torch.from_numpy(arr).unsqueeze(0).to(device)

            logits = model(feat_x)  # (1, T, vocab_size)

            # Replay board and check legality at each position
            board = chess.Board()
            for t in range(seq_len - 1):
                if is_plain:
                    # Position t+1 in token seq (after BOG) predicts move t+1
                    # Position t in token seq was fed move t, output predicts move t+1
                    # But we want to check what the model predicts for move t
                    # Input token[t] = BOG (t=0) or move_{t-1}+offset (t>0)
                    # Output logit[t] predicts the next token = move_t+offset
                    uci_id = logits[0, t].argmax(-1).item() - MOVE_OFFSET
                else:
                    uci_id = logits[0, t].argmax(-1).item()

                uci_str = UCI_MOVES[uci_id] if 0 <= uci_id < len(UCI_MOVES) else None
                move = _uci_to_chess_move(uci_str)
                if move is not None and move in board.legal_moves:
                    legal += 1
                total += 1

                # Advance board with actual move
                try:
                    actual_from = _FROM_SQUARES[game_seq["from_square"][t]]
                    actual_to = _TO_SQUARES[game_seq["to_square"][t]]
                    actual_promo_name = _PROMOTIONS[game_seq["promotion"][t]]
                    actual_promotion = _PROMO_MAP.get(actual_promo_name)
                    actual_move = chess.Move(
                        chess.parse_square(actual_from),
                        chess.parse_square(actual_to),
                        promotion=actual_promotion,
                    )
                    board.push(actual_move)
                except (ValueError, AssertionError):
                    break

    model.train()
    return legal, total

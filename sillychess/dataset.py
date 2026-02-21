import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import chess
import chess.pgn
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from sillychess.san_features import FEATURE_IDS, FEATURE_SPECS, move_features
from sillychess.uci_vocab import PAD_ID, BOG_ID, EOG_ID, MOVE_OFFSET
from pathlib import Path


@dataclass
class GameMoves:
    features: List[Dict[str, str]]


def _winner_color(result: str) -> Optional[str]:
    if result == "1-0":
        return "white"
    if result == "0-1":
        return "black"
    return None


def iter_pgn_games(pgn_path, max_games=None, self_color="white", winner_only=False):
    with open(pgn_path, "r", encoding="utf-8") as f:
        count = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            if self_color == "winner":
                winner = _winner_color(game.headers.get("Result", ""))
                if winner is None:
                    if winner_only:
                        continue
                    winner = "white"
                game_self_color = winner
            else:
                game_self_color = self_color
            board = game.board()
            features = []
            step = 1
            for move in game.mainline_moves():
                features.append(move_features(board, move, step, game_self_color))
                board.push(move)
                step += 1
            yield GameMoves(features=features)
            count += 1
            if max_games is not None and count >= max_games:
                break


def iter_jsonl_games(jsonl_path, max_games=None, self_color="white", winner_only=False):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        count = 0
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            moves = payload["moves"]
            if self_color == "winner":
                winner = _winner_color(payload.get("result", ""))
                if winner is None:
                    if winner_only:
                        continue
                    winner = "white"
                game_self_color = winner
            else:
                game_self_color = self_color
            board = chess.Board()
            features = []
            step = 1
            for san in moves:
                move = board.parse_san(san)
                features.append(move_features(board, move, step, game_self_color))
                board.push(move)
                step += 1
            yield GameMoves(features=features)
            count += 1
            if max_games is not None and count >= max_games:
                break


class CachedChessDataset(Dataset):
    """Map-style dataset that streams one shard at a time.

    Only one shard's worth of game data lives in memory.  Call
    ``advance_shard()`` to discard the current shard and load the next.

    Both uci_plain and 2D (composite) modes use full-game bucketed sequences.
    """

    def __init__(self, cache_dir, uci_plain=False, n_buckets=8, drop_above_percentile=99):
        self.uci_plain = uci_plain
        self._n_buckets = n_buckets
        self._drop_pct = drop_above_percentile

        cache_path = Path(cache_dir)
        parquet_shards = sorted(cache_path.glob("shard-*.parquet"))
        pt_shards = sorted(cache_path.glob("shard-*.pt"))

        if parquet_shards:
            self._shard_paths = parquet_shards
            self._shard_fmt = "parquet"
        elif pt_shards:
            self._shard_paths = pt_shards
            self._shard_fmt = "pt"
        else:
            raise ValueError("no shard-*.pt or shard-*.parquet files found in cache dir")

        self.total_shards = len(self._shard_paths)
        self._shard_idx = 0
        self._load_current_shard()

    # ------------------------------------------------------------------
    # Single-shard loaders
    # ------------------------------------------------------------------

    def _load_current_shard(self):
        """Load shard at ``_shard_idx`` and build bucket metadata."""
        self.sequences = []
        path = self._shard_paths[self._shard_idx]
        if self._shard_fmt == "parquet":
            self._load_one_parquet(path)
        else:
            self._load_one_pt(path)
        self.loaded_games = len(self.sequences)
        self._build_buckets(self._n_buckets, self._drop_pct)

    def _load_one_parquet(self, path):
        import pandas as pd
        feature_names = list(FEATURE_IDS.keys())
        df = pd.read_parquet(path)
        has_uci = "uci_move" in df.columns
        cols = feature_names + (["uci_move"] if has_uci else [])
        col_arrays = {name: df[name].values for name in cols}
        for idx in range(len(df)):
            feature_seq = {}
            for name in feature_names:
                val = col_arrays[name][idx]
                feature_seq[name] = np.asarray(val, dtype=np.int32) if val is not None else np.empty(0, dtype=np.int32)
            if has_uci:
                val = col_arrays["uci_move"][idx]
                feature_seq["uci_move"] = (np.asarray(val, dtype=np.int32) - 1) if val is not None else np.empty(0, dtype=np.int32)
            self.sequences.append(feature_seq)

    def _load_one_pt(self, path):
        payload = torch.load(path, map_location="cpu")
        features = payload["features"]
        n = len(next(iter(features.values())))
        for idx in range(n):
            self.sequences.append({name: features[name][idx] for name in FEATURE_IDS})

    # ------------------------------------------------------------------
    # Shard rotation
    # ------------------------------------------------------------------

    def advance_shard(self):
        """Discard current shard, load next (cycling).  Returns new shard index."""
        self._shard_idx = (self._shard_idx + 1) % len(self._shard_paths)
        self._load_current_shard()
        return self._shard_idx

    # ------------------------------------------------------------------
    # Bucketing (both modes)
    # ------------------------------------------------------------------

    def _build_buckets(self, n_buckets, drop_above_percentile):
        """Compute bucket boundaries and per-game metadata.

        uci_plain: token length = len(uci_move) + 2  (BOG + moves + EOG)
        composite: token length = len(game)  (raw feature length)
        """
        first_feat = next(iter(FEATURE_IDS))
        if self.uci_plain:
            if not self.sequences or "uci_move" not in self.sequences[0]:
                raise ValueError(
                    "uci_plain mode requires shards with 'uci_move' column -- "
                    "re-run the Go preprocessor with -vocab uci_vocab.txt"
                )
            tok_lens = np.array([len(s["uci_move"]) + 2 for s in self.sequences])
        else:
            tok_lens = np.array([len(s[first_feat]) for s in self.sequences])

        max_len = int(np.percentile(tok_lens, drop_above_percentile))
        kept_mask = tok_lens <= max_len
        kept_lens = tok_lens[kept_mask]
        percentiles = np.linspace(100 / n_buckets, 100, n_buckets)
        raw_boundaries = np.percentile(kept_lens, percentiles).astype(int)
        self.buckets = sorted(set(raw_boundaries.tolist()))

        self.bucket_ids = []
        self.padded_lens = []
        self._kept_indices = []  # indices into self.sequences
        self.dropped_games = 0
        for i, seq in enumerate(self.sequences):
            tl = tok_lens[i]
            bid = None
            for j, bmax in enumerate(self.buckets):
                if tl <= bmax:
                    bid = j
                    break
            if bid is None:
                self.dropped_games += 1
                continue
            self._kept_indices.append(i)
            self.bucket_ids.append(bid)
            self.padded_lens.append(self.buckets[bid])

    # ------------------------------------------------------------------
    # Train / eval split
    # ------------------------------------------------------------------

    def split_eval(self, eval_fraction=0.1):
        """Carve eval data from the current (first) shard.

        Returns ``(self, eval_ds)``.  ``self`` retains shard paths and can
        rotate via ``advance_shard()``.  ``eval_ds`` is a static snapshot
        that never rotates.
        """
        import copy

        n = len(self._kept_indices)
        n_eval = max(1, int(n * eval_fraction))
        n_train = n - n_eval

        eval_ds = copy.copy(self)
        eval_ds._shard_paths = []  # prevent rotation

        eval_ds._kept_indices = self._kept_indices[n_train:]
        eval_ds.bucket_ids = self.bucket_ids[n_train:]
        eval_ds.padded_lens = self.padded_lens[n_train:]
        eval_ds.loaded_games = n_eval

        self._kept_indices = self._kept_indices[:n_train]
        self.bucket_ids = self.bucket_ids[:n_train]
        self.padded_lens = self.padded_lens[:n_train]
        self.loaded_games = n_train

        return self, eval_ds

    def __len__(self):
        return len(self._kept_indices)

    def __getitem__(self, idx):
        if self.uci_plain:
            return self._getitem_uci_plain(idx)
        return self._getitem_2d(idx)

    def _getitem_uci_plain(self, idx):
        seq = self.sequences[self._kept_indices[idx]]
        moves = seq["uci_move"]
        padded_len = self.padded_lens[idx]

        tokens = np.empty(padded_len, dtype=np.int64)
        tokens[0] = BOG_ID
        n = len(moves)
        tokens[1:n + 1] = moves + MOVE_OFFSET
        tokens[n + 1] = EOG_ID
        if n + 2 < padded_len:
            tokens[n + 2:] = PAD_ID

        t = torch.from_numpy(tokens)
        return {"uci_move": t[:-1]}, {"uci_move": t[1:]}

    def _getitem_2d(self, idx):
        """Full-game bucketed item for composite (2D) mode.

        Pads all feature columns (including uci_move if present) to the
        bucket boundary.  Returns x=seq[:-1], y=seq[1:] for all columns.
        """
        seq = self.sequences[self._kept_indices[idx]]
        padded_len = self.padded_lens[idx]
        feat_x = {}
        feat_y = {}
        for name, values in seq.items():
            arr = np.asarray(values, dtype=np.int64)
            if len(arr) < padded_len:
                arr = np.pad(arr, (0, padded_len - len(arr)))
            t = torch.from_numpy(arr)
            feat_x[name] = t[:-1]
            feat_y[name] = t[1:]
        return feat_x, feat_y


# ---------------------------------------------------------------------------
# Bucket sampler for full-game mode
# ---------------------------------------------------------------------------

class BucketBatchSampler(Sampler):
    """Yields batches where all samples come from the same length bucket.

    Each epoch: shuffle within each bucket, then yield batches.
    Buckets are interleaved so training sees a mix of lengths.
    Drop-last per bucket to keep uniform batch shapes.
    """

    def __init__(self, bucket_ids, batch_size, shuffle=True):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle

        from collections import defaultdict
        buckets = defaultdict(list)
        for idx, bid in enumerate(bucket_ids):
            buckets[bid].append(idx)
        self._buckets = dict(buckets)

    def __iter__(self):
        all_batches = []
        for bid, indices in self._buckets.items():
            if self.shuffle:
                perm = [indices[i] for i in torch.randperm(len(indices)).tolist()]
            else:
                perm = indices[:]
            for start in range(0, len(perm) - self.batch_size + 1, self.batch_size):
                all_batches.append(perm[start:start + self.batch_size])

        if self.shuffle:
            order = torch.randperm(len(all_batches)).tolist()
            all_batches = [all_batches[i] for i in order]

        yield from all_batches

    def __len__(self):
        total = 0
        for indices in self._buckets.values():
            total += len(indices) // self.batch_size
        return total

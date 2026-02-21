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


def _windows_for_length(length: int, block_size: int) -> int:
    # Matches indexing logic below:
    # windows = max(0, length - block_size - 1) + 1
    if length > block_size:
        return length - block_size
    return 1


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


class ChessMoveDataset(Dataset):
    def __init__(self, games, block_size, target_windows=None):
        self.block_size = block_size
        self.sequences = []
        loaded_windows = 0
        for game in tqdm(games, desc="vectorize games", unit="game"):
            feature_seq = {name: [] for name in FEATURE_IDS}
            for feat in game.features:
                for name, value in feat.items():
                    feature_seq[name].append(FEATURE_IDS[name].get(value, 0))
            self.sequences.append(feature_seq)
            length = len(feature_seq[next(iter(FEATURE_IDS))])
            loaded_windows += _windows_for_length(length, block_size)
            if target_windows is not None and loaded_windows >= target_windows:
                break

        self.index = []
        for g_idx, feature_seq in enumerate(
            tqdm(self.sequences, desc="build windows", unit="game")
        ):
            length = len(feature_seq[next(iter(FEATURE_IDS))])
            max_start = max(0, length - block_size - 1)
            for start in range(max_start + 1):
                self.index.append((g_idx, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        g_idx, start = self.index[idx]
        feature_seq = self.sequences[g_idx]
        feature_chunk = {
            name: values[start : start + self.block_size + 1]
            for name, values in feature_seq.items()
        }
        current_len = len(next(iter(feature_chunk.values())))
        if current_len < self.block_size + 1:
            pad_len = self.block_size + 1 - current_len
            for name in feature_chunk:
                feature_chunk[name] = feature_chunk[name] + [0] * pad_len
        feat_x = {
            name: torch.tensor(values[:-1], dtype=torch.long)
            for name, values in feature_chunk.items()
        }
        feat_y = {
            name: torch.tensor(values[1:], dtype=torch.long)
            for name, values in feature_chunk.items()
        }
        return feat_x, feat_y


class CachedChessDataset(Dataset):
    def __init__(self, cache_dir, block_size=128, max_games=None, target_windows=None,
                 uci_plain=False, n_buckets=8, drop_above_percentile=99):
        self.block_size = block_size
        self.uci_plain = uci_plain
        self.sequences = []
        self._target_windows = target_windows
        self._loaded_windows = 0
        self.loaded_shards = 0
        self.loaded_games = 0
        cache_path = Path(cache_dir)
        pt_shards = sorted(cache_path.glob("shard-*.pt"))
        parquet_shards = sorted(cache_path.glob("shard-*.parquet"))

        if pt_shards:
            self._load_pt_shards(pt_shards, max_games=max_games)
        elif parquet_shards:
            self._load_parquet_shards(parquet_shards, max_games=max_games)
        else:
            raise ValueError("no shard-*.pt or shard-*.parquet files found in cache dir")

        if uci_plain:
            # Full-game mode: bucket by length, no sliding windows
            self._build_buckets(n_buckets, drop_above_percentile)
        else:
            # Window mode: sliding windows over feature sequences
            self.index = []
            for g_idx, feature_seq in enumerate(
                tqdm(self.sequences, desc="build windows", unit="game")
            ):
                length = len(feature_seq[next(iter(FEATURE_IDS))])
                max_start = max(0, length - block_size - 1)
                for start in range(max_start + 1):
                    self.index.append((g_idx, start))

    def _record_windows(self, length):
        self._loaded_windows += _windows_for_length(length, self.block_size)
        if self._target_windows is None:
            return False
        return self._loaded_windows >= self._target_windows

    def _load_pt_shards(self, shard_paths, max_games=None):
        count = 0
        pbar = tqdm(desc="load .pt shards", unit="shard")
        try:
            for shard_path in shard_paths:
                payload = torch.load(shard_path, map_location="cpu")
                features = payload["features"]
                length = len(next(iter(features.values())))
                self.loaded_shards += 1
                for idx in range(length):
                    feature_seq = {name: features[name][idx] for name in FEATURE_IDS}
                    self.sequences.append(feature_seq)
                    count += 1
                    self.loaded_games += 1
                    seq_len = len(feature_seq[next(iter(FEATURE_IDS))])
                    if self._record_windows(seq_len):
                        pbar.update(1)
                        pbar.set_postfix(
                            games=self.loaded_games,
                            windows=self._loaded_windows,
                            target=self._target_windows,
                            stop="target",
                        )
                        return
                    if max_games is not None and count >= max_games:
                        pbar.update(1)
                        pbar.set_postfix(
                            games=self.loaded_games,
                            windows=self._loaded_windows,
                            target=self._target_windows,
                            stop="max_games",
                        )
                        return
                pbar.update(1)
                pbar.set_postfix(
                    games=self.loaded_games,
                    windows=self._loaded_windows,
                    target=self._target_windows,
                )
        finally:
            pbar.close()

    def _load_parquet_shards(self, shard_paths, max_games=None):
        import pandas as pd

        feature_names = list(FEATURE_IDS.keys())
        pbar = tqdm(desc="load .parquet shards", unit="shard")
        try:
            for shard_path in shard_paths:
                df = pd.read_parquet(shard_path)
                has_uci = "uci_move" in df.columns
                cols = feature_names + (["uci_move"] if has_uci else [])
                # .values gives object array of numpy arrays — no Python int conversion
                col_arrays = {name: df[name].values for name in cols}
                n_rows = len(df)
                self.loaded_shards += 1
                for idx in range(n_rows):
                    feature_seq = {}
                    for name in feature_names:
                        val = col_arrays[name][idx]
                        feature_seq[name] = np.asarray(val, dtype=np.int32) if val is not None else np.empty(0, dtype=np.int32)
                    if has_uci:
                        val = col_arrays["uci_move"][idx]
                        feature_seq["uci_move"] = (np.asarray(val, dtype=np.int32) - 1) if val is not None else np.empty(0, dtype=np.int32)
                    self.sequences.append(feature_seq)
                    self.loaded_games += 1
                    seq_len = len(feature_seq[feature_names[0]])
                    if self._record_windows(seq_len):
                        pbar.update(1)
                        pbar.set_postfix(games=self.loaded_games, windows=self._loaded_windows, target=self._target_windows, stop="target")
                        return
                    if max_games is not None and self.loaded_games >= max_games:
                        pbar.update(1)
                        pbar.set_postfix(games=self.loaded_games, windows=self._loaded_windows, target=self._target_windows, stop="max_games")
                        return
                pbar.update(1)
                pbar.set_postfix(games=self.loaded_games, windows=self._loaded_windows, target=self._target_windows)
        finally:
            pbar.close()

    def _build_buckets(self, n_buckets, drop_above_percentile):
        """Compute bucket boundaries and per-game metadata for full-game mode."""
        tok_lens = np.array([len(s["uci_move"]) + 2 for s in self.sequences])
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

    def split_eval(self, eval_fraction=0.1):
        """Split into train/eval. Returns (train_ds, eval_ds)."""
        if self.uci_plain:
            n = len(self._kept_indices)
        else:
            n = len(self.sequences)
        n_eval = max(1, int(n * eval_fraction))
        n_train = n - n_eval

        # Shallow-copy and slice
        import copy
        train_ds = copy.copy(self)
        eval_ds = copy.copy(self)

        if self.uci_plain:
            train_ds._kept_indices = self._kept_indices[:n_train]
            train_ds.bucket_ids = self.bucket_ids[:n_train]
            train_ds.padded_lens = self.padded_lens[:n_train]
            train_ds.loaded_games = n_train
            eval_ds._kept_indices = self._kept_indices[n_train:]
            eval_ds.bucket_ids = self.bucket_ids[n_train:]
            eval_ds.padded_lens = self.padded_lens[n_train:]
            eval_ds.loaded_games = n_eval
        else:
            eval_seqs = self.sequences[n_train:]
            train_ds.sequences = self.sequences[:n_train]
            train_ds.loaded_games = n_train
            eval_ds.sequences = eval_seqs
            eval_ds.loaded_games = n_eval
            # Rebuild window indices
            train_ds.index = [(g, s) for g, s in self.index if g < n_train]
            eval_ds.index = [(g - n_train, s) for g, s in self.index if g >= n_train]

        return train_ds, eval_ds

    def __len__(self):
        if self.uci_plain:
            return len(self._kept_indices)
        return len(self.index)

    def __getitem__(self, idx):
        if self.uci_plain:
            return self._getitem_uci_plain(idx)
        return self._getitem_windowed(idx)

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

    def _getitem_windowed(self, idx):
        g_idx, start = self.index[idx]
        feature_seq = self.sequences[g_idx]
        end = start + self.block_size + 1
        feat_x = {}
        feat_y = {}
        for name, values in feature_seq.items():
            chunk = values[start:end]
            pad_needed = self.block_size + 1 - len(chunk)
            if pad_needed > 0:
                chunk = np.pad(chunk, (0, pad_needed))
            t = torch.from_numpy(chunk.astype(np.int64))
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
        """
        Args:
            bucket_ids: list[int] — bucket index per dataset sample.
            batch_size: int.
            shuffle: bool — shuffle within buckets each epoch.
        """
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Group sample indices by bucket
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

        # Shuffle batch order so training alternates between buckets
        if self.shuffle:
            order = torch.randperm(len(all_batches)).tolist()
            all_batches = [all_batches[i] for i in order]

        yield from all_batches

    def __len__(self):
        total = 0
        for indices in self._buckets.values():
            total += len(indices) // self.batch_size
        return total




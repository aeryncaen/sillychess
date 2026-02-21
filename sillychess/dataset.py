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
    def __init__(self, cache_dir, block_size, max_games=None, target_windows=None):
        self.block_size = block_size
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

        count = 0
        feature_names = list(FEATURE_IDS.keys())
        pbar = tqdm(desc="load .parquet shards", unit="shard")
        try:
            for shard_path in shard_paths:
                df = pd.read_parquet(shard_path)
                has_uci = "uci_move" in df.columns
                self.loaded_shards += 1
                for idx in range(len(df)):
                    feature_seq = {}
                    for name in feature_names:
                        val = df[name].iloc[idx]
                        feature_seq[name] = np.asarray(val, dtype=np.int32) if val is not None else np.empty(0, dtype=np.int32)
                    if has_uci:
                        val = df["uci_move"].iloc[idx]
                        # Parquet stores 1-indexed; convert to 0-indexed
                        feature_seq["uci_move"] = (np.asarray(val, dtype=np.int32) - 1) if val is not None else np.empty(0, dtype=np.int32)
                    self.sequences.append(feature_seq)
                    count += 1
                    self.loaded_games += 1
                    seq_len = len(feature_seq[feature_names[0]])
                    if self._record_windows(seq_len):
                        pbar.update(1)
                        pbar.set_postfix(games=self.loaded_games, windows=self._loaded_windows, target=self._target_windows, stop="target")
                        return
                    if max_games is not None and count >= max_games:
                        pbar.update(1)
                        pbar.set_postfix(games=self.loaded_games, windows=self._loaded_windows, target=self._target_windows, stop="max_games")
                        return
                pbar.update(1)
                pbar.set_postfix(games=self.loaded_games, windows=self._loaded_windows, target=self._target_windows)
        finally:
            pbar.close()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
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
# Full-game dataset + bucket sampler for --uci-plain mode
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


class FullGameDataset(Dataset):
    """Full-game dataset for --uci-plain mode.

    Each sample is one complete game tokenized as:
        [BOG, move_0+OFFSET, move_1+OFFSET, ..., move_N+OFFSET, EOG, PAD...]

    Input = tokens[:-1], target = tokens[1:].

    Games are bucketed by length for efficient batching (all games in a
    batch are padded to the same bucket length).
    """

    def __init__(self, cache_dir, max_games=None, n_buckets=8, max_drop_pct=1.0):
        """
        Args:
            cache_dir: path to directory with shard-*.parquet files.
            max_games: optional cap on number of games loaded.
            n_buckets: number of length buckets (determined from data).
            max_drop_pct: drop games above this percentile (default 1.0 = p99).
        """
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "pyarrow is required for FullGameDataset. "
                "Install with: pip install pyarrow"
            ) from exc

        cache_path = Path(cache_dir)
        parquet_shards = sorted(cache_path.glob("shard-*.parquet"))
        if not parquet_shards:
            raise ValueError(f"no shard-*.parquet files found in {cache_dir}")

        # --- Pass 1: load all games, compute token lengths ---
        import pandas as pd
        all_moves = []   # list of np.ndarray (int32, 0-indexed)
        for shard_path in parquet_shards:
            df = pd.read_parquet(shard_path, columns=["uci_move"])
            for val in df["uci_move"]:
                if val is None:
                    continue
                arr = np.asarray(val, dtype=np.int32)
                mask = arr > 0
                if not mask.any():
                    continue
                # 1-indexed parquet → 0-indexed
                all_moves.append(arr[mask] - 1)
                if max_games is not None and len(all_moves) >= max_games:
                    break
            if max_games is not None and len(all_moves) >= max_games:
                break

        # Token length per game: BOG + moves + EOG
        tok_lens = [len(m) + 2 for m in all_moves]

        # --- Compute bucket boundaries from data ---
        sorted_lens = sorted(tok_lens)
        n_total = len(sorted_lens)
        # Drop above max_drop_pct percentile
        cutoff_idx = max(1, int(n_total * (1.0 - max_drop_pct / 100.0)))
        max_len = sorted_lens[-cutoff_idx] if cutoff_idx <= n_total else sorted_lens[-1]
        # Quantile-based bucket boundaries so each bucket has ~equal games
        bucket_boundaries = []
        for i in range(1, n_buckets + 1):
            pct_idx = int(n_total * i / n_buckets) - 1
            pct_idx = min(pct_idx, n_total - 1)
            boundary = sorted_lens[pct_idx]
            if boundary > max_len:
                boundary = max_len
            if not bucket_boundaries or boundary > bucket_boundaries[-1]:
                bucket_boundaries.append(boundary)
        self.buckets = bucket_boundaries

        # --- Pass 2: assign games to buckets ---
        self.games = []
        self.bucket_ids = []
        self.padded_lens = []
        dropped = 0
        for moves in all_moves:
            tok_len = len(moves) + 2
            bid = None
            for i, bmax in enumerate(self.buckets):
                if tok_len <= bmax:
                    bid = i
                    break
            if bid is None:
                dropped += 1
                continue
            self.games.append(moves)
            self.bucket_ids.append(bid)
            self.padded_lens.append(self.buckets[bid])

        self.loaded_games = len(self.games)
        self.dropped_games = dropped
        self.sequences = None  # not used in full-game mode

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        moves = self.games[idx]
        padded_len = self.padded_lens[idx]

        # Build token sequence: [BOG, move+OFFSET, ..., EOG, PAD...]
        tokens = np.empty(padded_len, dtype=np.int64)
        tokens[0] = BOG_ID
        n = len(moves)
        tokens[1:n + 1] = moves + MOVE_OFFSET
        tokens[n + 1] = EOG_ID
        if n + 2 < padded_len:
            tokens[n + 2:] = PAD_ID

        t = torch.from_numpy(tokens)
        return {"uci_move": t[:-1]}, {"uci_move": t[1:]}

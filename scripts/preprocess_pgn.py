import argparse
import io
import json
from pathlib import Path

import chess.pgn
import torch
from tqdm import tqdm

from sillychess.san_features import FEATURE_IDS, move_features


def winner_color(result):
    if result == "1-0":
        return "white"
    if result == "0-1":
        return "black"
    return None


def open_pgn(path):
    if path.suffix == ".zst":
        import zstandard as zstd

        dctx = zstd.ZstdDecompressor()
        stream = dctx.stream_reader(path.open("rb"))
        return io.TextIOWrapper(stream, encoding="utf-8")
    return path.open("r", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-size", type=int, default=10000)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--perspective", choices=["winner", "white", "black"], default="winner")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "feature_ids": {name: mapping for name, mapping in FEATURE_IDS.items()},
        "shard_size": args.shard_size,
    }
    with (output_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    shard_idx = 0
    features_shard = {name: [] for name in FEATURE_IDS}
    count = 0

    pgn_path = Path(args.pgn)
    handle = open_pgn(pgn_path)
    try:
        total = args.max_games
        pbar = tqdm(total=total, unit="game")
        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                break

            if args.perspective == "winner":
                winner = winner_color(game.headers.get("Result", ""))
                if winner is None:
                    continue
                self_color = winner
            else:
                self_color = args.perspective

            board = game.board()
            features = {name: [] for name in FEATURE_IDS}
            step = 1
            for move in game.mainline_moves():
                feat = move_features(board, move, step, self_color)
                for name, value in feat.items():
                    features[name].append(FEATURE_IDS[name].get(value, 0))
                board.push(move)
                step += 1

            if not features["step"]:
                continue

            for name in FEATURE_IDS:
                features_shard[name].append(features[name])

            count += 1
            pbar.update(1)
            if args.max_games is not None and count >= args.max_games:
                break

            if len(features_shard["step"]) >= args.shard_size:
                shard_path = output_dir / f"shard-{shard_idx:05d}.pt"
                torch.save({"features": features_shard}, shard_path)
                shard_idx += 1
                features_shard = {name: [] for name in FEATURE_IDS}

        pbar.close()
        if features_shard["step"]:
            shard_path = output_dir / f"shard-{shard_idx:05d}.pt"
            torch.save({"features": features_shard}, shard_path)
    finally:
        handle.close()


if __name__ == "__main__":
    main()

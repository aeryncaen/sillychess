"""Analyze game lengths in parquet shards to inform bucket sizes."""

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_dir", help="directory containing shard-*.parquet files")
    args = parser.parse_args()

    lengths = []
    shards = sorted(Path(args.cache_dir).glob("shard-*.parquet"))
    if not shards:
        print(f"no shard-*.parquet files in {args.cache_dir}")
        return

    for shard in shards:
        table = pq.read_table(shard, columns=["piece"])
        for game in table["piece"].to_pylist():
            if game:
                # +2 for BOG/EOG tokens that will be added
                lengths.append(len(game) + 2)

    lengths = np.array(lengths)
    print(f"shards: {len(shards)}")
    print(f"games:  {len(lengths)}")
    print(f"min: {lengths.min()}  max: {lengths.max()}  "
          f"mean: {lengths.mean():.1f}  median: {np.median(lengths):.0f}  "
          f"std: {lengths.std():.1f}")
    print()
    print("percentiles (token length = moves + BOG + EOG):")
    for p in [5, 10, 25, 50, 75, 90, 95, 99, 100]:
        print(f"  p{p:3d}: {np.percentile(lengths, p):6.0f}")
    print()
    print("length distribution:")
    edges = list(range(0, 52, 4)) + list(range(52, 102, 10)) + [120, 140, 160, 200, 250, 300, 400, 600, 1200]
    counts, _ = np.histogram(lengths, bins=edges)
    cum = 0
    for i, count in enumerate(counts):
        if count == 0:
            continue
        cum += count
        pct = 100 * count / len(lengths)
        cum_pct = 100 * cum / len(lengths)
        bar = "#" * int(pct / 2)
        print(f"  [{edges[i]:4d},{edges[i+1]:4d})  {count:8d}  {pct:5.1f}%  cum {cum_pct:5.1f}%  {bar}")


if __name__ == "__main__":
    main()

# sillychess

Transformer next-move prediction over SAN (algebraic notation) tokens for imitation learning.
Each token is embedded as a 2D matrix via a composite lookup over SAN features (piece, from-square, to-square, capture, promotion, check/mate, castling, move step, player perspective).

## Data format

Training expects one game per line in JSONL:

```
{"moves": ["e4", "e5", "Nf3", "Nc6"]}
```

- `moves` is a list of SAN tokens.

PGN and SAN are replayed into feature sequences (piece/from/to/capture/promo/check/castle/step/player).

## Quick start

Train directly from PGN:

```
python scripts/train.py --data data/games.pgn --format pgn
```

Tune the 2D embedding shape and attention depth:

```
python scripts/train.py --data data/games.pgn --format pgn --h-rows 4 --w-dim 64 --n-intra-layer 2 --n-inter-layer 4
```

Use winner perspective (self = winner, draws dropped, default):

```
python scripts/train.py --data data/games.pgn --format pgn --perspective winner
```

Pre-tokenize and cache PGN to disk:

```
python scripts/preprocess_pgn.py --pgn /path/to/lichess_db_standard_rated_2014-07.pgn.zst --output-dir data/cache/lichess-2014-07
```

Train from cache:

```
python scripts/train.py --cache-dir data/cache/lichess-2014-07
```

## Notes

- The model uses two-stage attention: intra-token attention over the embedding rows, then causal attention over the move sequence.
- Output head is feature-only and uses a small ML-Decoder style query decoder with one query per class (`--feature-decoder-layers`).
- For large datasets, consider sharding JSONL and streaming loaders.
- Planned next step: outcome-based RL by playing against Stockfish at increasing strength levels (win/draw/loss reward).
- Feature embeddings use the true from/to squares computed by replaying each game, so SAN sequences must be legal.
- The `player` feature is derived from a per-game perspective; use `--perspective white`, `--perspective black`, `--perspective both`, or `--perspective winner` (default).
- Features use a shared `NULL` value for non-applicable fields and padding positions.
- When `--perspective winner` is set, draws are skipped and JSONL inputs must include a `result` field (`1-0` or `0-1`).

# sillychess

Transformer chess move prediction trained on Lichess games. Two model variants:

- **2D feature-attention model** (`--uci`): Composite SAN feature embeddings (piece, from/to square, capture, promotion, check, castle, step, player) with per-feature attention heads and cross-feature attention MLP.
- **Plain 1D transformer** (`--uci-plain`): Standard autoregressive transformer over a 1968-token UCI move vocabulary. Supports full-game training with `--full-games`.

Both variants predict UCI moves (e.g. `e2e4`, `e7e8q`). The UCI vocabulary is 1968 geometrically valid moves with no disambiguation ambiguity.

## Setup

### Python

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Dependencies: `torch`, `python-chess`, `pyarrow`, `zstandard`, `tqdm`.

### Go preprocessor

Requires Go 1.21+.

```bash
go build -o preprocess ./cmd/preprocess
go build -o vocabbuilder ./cmd/vocabbuilder
```

This produces two binaries in the project root:
- `preprocess` — converts PGN files to parquet shards
- `vocabbuilder` — enumerates the 1968-move UCI vocabulary

## 1. Download data

Lichess publishes monthly game databases at https://database.lichess.org/

```bash
# Example: July 2014 (~1.5M games, ~300MB compressed)
wget https://database.lichess.org/standard/lichess_db_standard_rated_2014-07.pgn.zst
```

Any `.pgn` or `.pgn.zst` file works. The preprocessor only keeps decisive games (1-0 or 0-1); draws are skipped.

## 2. Build UCI vocab

```bash
./vocabbuilder > uci_vocab.txt
```

This enumerates all 1968 geometrically valid UCI moves and writes them sorted to `uci_vocab.txt`. Only needs to be done once (the file is already in the repo).

## 3. Preprocess

```bash
./preprocess \
  -pgn lichess_db_standard_rated_2014-07.pgn.zst \
  -out data/lichess-2014-07 \
  -vocab uci_vocab.txt
```

This streams the PGN, replays each decisive game, and writes parquet shards (`shard-00000.parquet`, etc.) to the output directory. Each shard holds up to 100K games. Uses all CPU cores.

Output columns (per game, variable-length int32 arrays): `piece`, `from_square`, `to_square`, `capture`, `promotion`, `check`, `castle`, `step`, `player`, `uci_move`.

The `uci_move` column stores 1-indexed vocab IDs (0 = null/padding).

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-pgn` | (required) | Input PGN file (`.pgn` or `.pgn.zst`) |
| `-out` | (required) | Output directory for parquet shards |
| `-vocab` | `uci_vocab.txt` | Path to UCI vocab file |
| `-max-games` | 0 (unlimited) | Cap on number of games to process |

## 4. Analyze game lengths (for bucket sizing)

Before full-game training, check the length distribution:

```bash
python scripts/analyze_shards.py --cache-dir data/lichess-2014-07
```

This reports percentiles and a histogram of game lengths (in plies) to help choose `--buckets`.

## 5. Train

### Plain 1D transformer (full-game mode, recommended)

```bash
python scripts/train.py \
  --cache-dir data/lichess-2014-07 \
  --uci-plain \
  --full-games \
  --buckets 40,80,120,200,350 \
  --w-dim 128 \
  --n-head 4 \
  --n-layer 8 \
  --batch-size 64 \
  --steps 10000 \
  --lr 3e-4 \
  --lr-schedule cosine \
  --eval-every 200
```

Each sample is a complete game tokenized as `[BOG, move+3, ..., EOG, PAD...]`. Games are bucketed by length for efficient batching. Loss uses `ignore_index=0` (PAD).

### Plain 1D transformer (window mode)

```bash
python scripts/train.py \
  --cache-dir data/lichess-2014-07 \
  --uci-plain \
  --block-size 128 \
  --w-dim 128 \
  --n-head 4 \
  --n-layer 8 \
  --batch-size 64 \
  --steps 10000 \
  --lr 3e-4
```

### 2D feature-attention model

```bash
python scripts/train.py \
  --cache-dir data/lichess-2014-07 \
  --uci \
  --block-size 128 \
  --w-dim 48 \
  --rows-per-feature 4 \
  --n-layer 6 \
  --batch-size 32 \
  --steps 5000 \
  --lr 1e-3
```

### Key training flags

| Flag | Default | Description |
|------|---------|-------------|
| `--cache-dir` | | Directory with parquet shards |
| `--uci` | off | Single UCI move head on 2D model |
| `--uci-plain` | off | Plain 1D transformer (implies `--uci`) |
| `--full-games` | off | Full-game training (requires `--uci-plain`) |
| `--buckets` | `40,80,120,200,350` | Bucket max-lengths for `--full-games` |
| `--block-size` | 128 | Context window (window mode only) |
| `--w-dim` | 48 | Model dimension (`d_model` for plain, descriptor width for 2D) |
| `--n-head` | 4 | Attention heads (plain mode) |
| `--rows-per-feature` | 4 | Embedding rows per feature (2D mode) |
| `--n-layer` | 6 | Transformer layers |
| `--batch-size` | 32 | Batch size |
| `--steps` | 2000 | Training steps |
| `--lr` | 1e-3 | Peak learning rate |
| `--lr-schedule` | `constant` | `constant`, `cosine`, or `wsd` |
| `--warmup-steps` | 200 | LR warmup steps |
| `--dropout` | 0.1 | Dropout rate |
| `--grad-clip` | 1.0 | Gradient clipping norm |
| `--save-model` | `model.pt` | Output path for saved model |
| `--device` | `auto` | `auto`, `cuda`, `mps`, or `cpu` |

## Architecture notes

- **UCI vocabulary**: 1968 moves. Full-game mode adds 3 special tokens (PAD=0, BOG=1, EOG=2, moves at offset 3) for a total vocab of 1971.
- **2D model**: 9 features each get `rows_per_feature` embedding rows. Sequence attention uses rows as heads. Cross-feature mixing via softmax feature-attention in the MLP (replaces SwiGLU gating). Per-feature weights via batched matmul, no cross-feature mixing except through attention.
- **Plain model**: Standard GPT-style with RoPE, SwiGLU MLP, RMSNorm, weight-tied embedding/head.
- **Embedding**: Single `nn.Embedding` table with per-feature offsets (1156 total classes across all features).
- The preprocessor only keeps decisive games. The `player` feature encodes the relationship to the winner (self-as-white, self-as-black, opponent-as-white, opponent-as-black).

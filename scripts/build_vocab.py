import argparse

from sillychess.dataset import iter_jsonl_games, iter_pgn_games
from sillychess.vocab import MoveVocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--format", choices=["jsonl", "pgn"], default="jsonl")
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.format == "jsonl":
        games = iter_jsonl_games(args.data, max_games=args.max_games)
    else:
        games = iter_pgn_games(args.data, max_games=args.max_games)

    vocab = MoveVocab.build((game.moves for game in games), min_freq=args.min_freq)
    vocab.save(args.output)


if __name__ == "__main__":
    main()

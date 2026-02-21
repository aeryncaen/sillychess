from __future__ import annotations

import chess


FILES = "abcdefgh"
RANKS = "12345678"
PROMOS = "NBRQ"
NULL = "NULL"

SQUARES = [NULL] + [f + r for r in RANKS for f in FILES]

PIECE_MAP = {
    chess.PAWN: "P",
    chess.KNIGHT: "N",
    chess.BISHOP: "B",
    chess.ROOK: "R",
    chess.QUEEN: "Q",
    chess.KING: "K",
}


FEATURE_SPECS = {
    "piece": [NULL] + list(PIECE_MAP.values()),
    "from_square": SQUARES,
    "to_square": SQUARES,
    "capture": [NULL, "x"],
    "promotion": [NULL] + list(PROMOS),
    "check": [NULL, "+", "#"],
    "castle": [NULL, "O-O", "O-O-O"],
    "step": [NULL] + [str(i) for i in range(1, 1001)],
    "player": [NULL, "self-white", "self-black", "opponent-white", "opponent-black"],
}


FEATURE_SIZES = {name: len(values) for name, values in FEATURE_SPECS.items()}
FEATURE_IDS = {
    name: {value: idx for idx, value in enumerate(values)}
    for name, values in FEATURE_SPECS.items()
}
FEATURE_ORDER = list(FEATURE_SPECS.keys())


def _player_feature(turn: bool, self_color: str) -> str:
    if self_color == "white":
        return "self-white" if turn == chess.WHITE else "opponent-black"
    if self_color == "black":
        return "self-black" if turn == chess.BLACK else "opponent-white"
    raise ValueError("self_color must be 'white' or 'black'")


def move_features(
    board: chess.Board, move: chess.Move, step: int, self_color: str
) -> dict[str, str]:
    piece_type = board.piece_type_at(move.from_square)
    piece = PIECE_MAP.get(piece_type, NULL)
    from_sq = chess.square_name(move.from_square)
    to_sq = chess.square_name(move.to_square)
    capture = "x" if board.is_capture(move) else NULL
    promotion = PIECE_MAP.get(move.promotion, NULL) if move.promotion else NULL
    if board.is_castling(move):
        castle = "O-O" if chess.square_file(move.to_square) == 6 else "O-O-O"
    else:
        castle = NULL

    board.push(move)
    if board.is_checkmate():
        check = "#"
    elif board.is_check():
        check = "+"
    else:
        check = NULL
    board.pop()

    return {
        "piece": piece,
        "from_square": from_sq,
        "to_square": to_sq,
        "capture": capture,
        "promotion": promotion,
        "check": check,
        "castle": castle,
        "step": str(min(step, 1000)),
        "player": _player_feature(board.turn, self_color),
    }


def empty_feature_values() -> dict[str, str]:
    return {name: "" for name in FEATURE_SPECS}

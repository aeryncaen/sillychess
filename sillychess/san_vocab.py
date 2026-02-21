FILES = "abcdefgh"
RANKS = "12345678"
PIECES = "NBRQK"
PROMOS = "NBRQ"
CHECK_SUFFIXES = ["", "+", "#"]


def generate_san_tokens():
    tokens = set()

    for suffix in CHECK_SUFFIXES:
        tokens.add(f"O-O{suffix}")
        tokens.add(f"O-O-O{suffix}")

    disambigs = [""] + list(FILES) + list(RANKS) + [f + r for f in FILES for r in RANKS]
    for piece in PIECES:
        for dis in disambigs:
            for capture in ["", "x"]:
                for file in FILES:
                    for rank in RANKS:
                        square = f"{file}{rank}"
                        for suffix in CHECK_SUFFIXES:
                            tokens.add(f"{piece}{dis}{capture}{square}{suffix}")

    for file in FILES:
        for rank in RANKS:
            square = f"{file}{rank}"
            if rank in "18":
                for promo in PROMOS:
                    for suffix in CHECK_SUFFIXES:
                        tokens.add(f"{square}={promo}{suffix}")
            else:
                for suffix in CHECK_SUFFIXES:
                    tokens.add(f"{square}{suffix}")

    for origin_file in FILES:
        for file in FILES:
            for rank in RANKS:
                square = f"{file}{rank}"
                if rank in "18":
                    for promo in PROMOS:
                        for suffix in CHECK_SUFFIXES:
                            tokens.add(f"{origin_file}x{square}={promo}{suffix}")
                else:
                    for suffix in CHECK_SUFFIXES:
                        tokens.add(f"{origin_file}x{square}{suffix}")

    return sorted(tokens)

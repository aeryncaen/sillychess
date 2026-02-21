import json
from collections import Counter


class MoveVocab:
    def __init__(self, tokens):
        self.itos = list(tokens)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    @classmethod
    def build(cls, token_iter, min_freq=1):
        counter = Counter()
        for tokens in token_iter:
            counter.update(tokens)
        special = ["<pad>", "<unk>"]
        vocab = special + [tok for tok, n in counter.items() if n >= min_freq]
        return cls(vocab)

    @property
    def pad_id(self):
        return self.stoi["<pad>"]

    @property
    def unk_id(self):
        return self.stoi["<unk>"]

    def encode(self, tokens):
        ids = []
        for tok in tokens:
            ids.append(self.stoi.get(tok, self.unk_id))
        return ids

    def decode(self, ids, drop_special=True):
        tokens = []
        for idx in ids:
            tok = self.itos[idx]
            if drop_special and tok.startswith("<") and tok.endswith(">"):
                continue
            tokens.append(tok)
        return tokens

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"itos": self.itos}, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["itos"])

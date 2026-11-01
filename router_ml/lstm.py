import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from router_ml import data, metrics

TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")
PAD, UNK = 0, 1

def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())

class Vocab:
    def __init__(self, words: list[str]):
        self.itos = ["<pad>", "<unk>", *words]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def build(cls, texts, min_freq: int = 2) -> "Vocab":
        counts = Counter(t for text in texts for t in tokenize(text))
        return cls(sorted(w for w, n in counts.items() if n >= min_freq))

    def encode(self, text: str, max_len: int = 64) -> list[int]:
        return [self.stoi.get(t, UNK) for t in tokenize(text)][:max_len] or [UNK]

def batchify(vocab: Vocab, texts, labels=None):
    ids = [vocab.encode(t) for t in texts]
    width = max(len(x) for x in ids)
    x = torch.full((len(ids), width), PAD, dtype=torch.long)
    for row, seq in enumerate(ids):
        x[row, :len(seq)] = torch.tensor(seq)
    y = None if labels is None else torch.tensor(labels, dtype=torch.long)
    return x, y

class BiLSTMClassifier(nn.Module):
    
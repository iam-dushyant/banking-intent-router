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
    def __init__(self, vocab_size: int, n_classes: int, dim: int = 128, hidden: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.lstm = nn.LSTM(dim, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(4 * hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = (x != PAD).unsqueeze(-1)
        lengths = mask.sum(1).squeeze(-1).cpu()
        packed = pack_padded_sequence(self.embed(x), lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=x.shape[1])
        mean = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        biggest = h.masked_fill(~mask, -1e4).max(1).values
        return self.out(self.drop(torch.cat([mean, biggest], dim=1)))

@torch.no_grad()
def predict_proba(model, vocab, texts, device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        x, _ = batchify(vocab, texts[i: i + batch_size])
        out.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
    return np.concatenate(out)

def train(ds, epochs: int, device: str, batch_size: int = 64, seed: int = 42):
    torch.manual_seed(seed)
    texts, labels = ds["train"]["text"], ds["train"]["label"]
    vocab = Vocab.build(texts)
    model = BiLSTMClassifier(len(vocab.itos), len(data.label_names(ds))).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    val_y = np.array(ds["validation"]["label"])
    best, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(texts)).tolist()
        total = 0.0
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            x, y = batchify(vocab, [texts[j] for j in idx], [labels[j] for j in idx])
            loss = loss_fn(model(x.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        probs = predict_proba(model, vocab, ds["validation"]["text"], device)
        val_acc = float((probs.argmax(1) == val_y).mean())
        print(f"epoch {epoch:>2} loss {total / len(texts):.3f} val_acc {val_acc:.4f}")
        if val_acc > best:
            best = val_acc
            best_state = {k : v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, vocab, best

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--device", default=metrics.pick_device())
    p.add_argument("--out", type=Path, default=Path("artifacts/lstm"))
    args = p.parse_args()
    ds = data.load()
    start = time.perf_counter()
    model, vocab, val_acc = train(ds, args.epochs, args.device)
    train_s = time.perf_counter() - start
    probs = predict_proba(model, vocab, ds["test"]["text"], args.device)
    card = metrics.scorecard(ds["test"]["label"], probs, data.label_names(ds))
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out / "model.pt")
    (args.out / "vocab.json").write_text(json.dumps(vocab.itos))
    params = sum(p.numel() for p in model.parameters())
    metrics.save(
        "bilstm",
        card,
        val_accuracy=round(val_acc, 4),
        train_s=train_s,
        device=args.device,
        params=params,
    )

if __name__ == "__main__":
    main()
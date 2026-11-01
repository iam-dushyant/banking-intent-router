import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

RESULTS = Path("results")

def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def selective(probs: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    conf = probs.max(axis=1)
    answered = conf >= threshold
    acc = float((probs.argmax(axis=1) == y)[answered].mean()) if answered.any() else 0.0
    return {
        "threshold": threshold,
        "coverage": round(float(answered.mean()), 4),
        "accuracy": round(acc, 4),
    }

def scorecard(y_true, probs: np.ndarray, names: list[str]) -> dict:
    y = np.asarray(y_true)
    pred = probs.argmax(axis=1)
    pairs = Counter(
        (names[t], names[p]) for t, p in zip(y, pred, strict=True) if t != p
    )
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "at_0.5": selective(probs, y, 0.5),
        "at_0.9": selective(probs, y, 0.9),
        "top_confusions": [f"{t} -> {p} ({n})" for (t,p), n in pairs.most_common(5)],
    }

def save(name: str, card: dict, **extra) -> Path:
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{name}.json"
    record = {"model": name, **card, **extra, "machine": platform.platform()}
    out.write_text(json.dumps(record, indent=2))
    print(json.dumps({k: record[k] for k in ("model", "accuracy", "macro_f1")}))
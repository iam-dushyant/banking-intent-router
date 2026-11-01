import argparse
from pathlib import Path

from datasets import ClassLabel, DatasetDict, load_dataset, load_from_disk

RAW = Path("data/raw")
OUT = Path("data/banking77")
SEED = 42

def clean_label(name: str) -> str:
    return name.strip().lower().rstrip("?")

def build(raw: Path = RAW, val_size: float = 0.1) -> DatasetDict:
    ds = load_dataset(
        "csv",
        data_files = {"train": str(raw / "train.csv"), "test": str(raw / "test.csv")},
    )
    ds = ds.rename_column("category", "intent")
    ds = ds.map(lambda row: {"intent": clean_label(row["intent"])})
    names = sorted(set(ds["train"]["intent"]))
    labels = ClassLabel(names=names)
    ds = ds.map(lambda row: {"label": labels.str2int(row["intent"])})
    ds = ds.cast_column("label", labels)
    split = ds["train"].train_test_split(
        test_size=val_size, stratify_by_column="label", seed=SEED
    )
    return DatasetDict(train=split["train"], validation=split["test"], test=ds["test"])

def load(path: Path = OUT) -> DatasetDict:
    return load_from_disk(str(path))

def label_names(ds: DatasetDict) -> list[str]:
    return ds["train"].features["label"].names

def sample_per_class(split, k: int, seed: int = SEED):
    shuffled = split.shuffle(seed=seed)
    seen: dict[int, int] = {}
    keep = []
    for i, label in enumerate(shuffled['label']):
        if seen.get(label, 0) < k:
            seen[label] = seen.get(label, 0) + 1
            keep.append(i)
    return shuffled.select(keep)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=RAW)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    ds = build(args.raw)
    ds.save_to_disk(str(args.out))
    for name, split in ds.items():
        print(f"{name:<11} {split.num_rows:>6} rows")
        # Consistent presentation with column padding
        # e.g. :<11, :>6
    print(f"intents {len(label_names(ds)):>6}")

if __name__ == "__main__":
    main()
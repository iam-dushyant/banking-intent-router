set -euo pipefail
SHA=57ec275d8078af65b7731c2a98be812d844a6d6b
REPO=https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets
BASE=$REPO/$SHA/banking_data
mkdir -p data/raw
for f in train.csv test.csv categories.json; do
  curl -fsS -o "data/raw/$f" "$BASE/$f"
done
python - <<'PY'
import csv
for split in ("train", "test"):
    with open(f"data/raw/{split}.csv", newline="") as f:
        print(split, sum(1 for _ in csv.DictReader(f)), "rows")
PY
import json

from router_ml.metrics import RESULTS

def main() -> None:
    rows = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json"))]
    rows.sort(key=lambda r: -r["accuracy"])
    print(
        f"{'model':<16} {'test acc':>8} {'macro F1':>8} {'cover@0.9':>9} "
        f"{'acc@0.9':>7} {'train_s':>8} device"
    )
    for r in rows:
        print(
            f"{r['model']:<16} {r['accuracy']:>8.3f} {r['macro_f1']:>8.3f} "
            f"{r['at_0.9']['coverage']:>9.3f} {r['at_0.9']['accuracy']:>7.3f} "
            f"{r.get('train_s', 0):>8.0f} {r.get('device','-')}"
        )

if __name__ == "__main__":
    main()
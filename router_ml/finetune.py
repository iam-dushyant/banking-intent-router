import argparse
import time
from pathlib import Path

import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from router_ml import data, metrics

BASE = "distilbert/distilbert-base-uncased"

def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.exp(logits - logits.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=BASE, help="Hub id or local folder")
    p.add_argument("--epochs", type=float, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--out", type=Path, default=Path("artifacts/distilbert"))
    p.add_argument("--name", default="distilbert")
    args = p.parse_args()

    ds = data.load()
    names = data.label_names(ds)
    tok = AutoTokenizer.from_pretrained(args.model)
    enc = ds.map(lambda b: tok(b["text"], truncation=True, max_length=64), batched=True)
    enc = enc.remove_columns(["text", "intent"])
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(names),
        id2label=dict(enumerate(names)),
        label2id={n: i for i, n in enumerate(names)}
    )
    device = metrics.pick_device()
    targs = TrainingArguments(
        output_dir="runs/finetune",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=128,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        fp16=device == "cuda",
        report_to="none",
        seed=42,
    )

    def accuracy(pred) -> dict:
        return {
            "accuracy": float((pred.predictions.argmax(1) == pred.label_ids).mean())
        }

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=enc["train"],
        eval_dataset=enc["validation"],
        processing_class=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=accuracy,
    )
    start = time.perf_counter()
    trainer.train()
    train_s = time.perf_counter() - start
    val_acc = trainer.evaluate()["eval_accuracy"]
    probs = softmax(trainer.predict(enc["test"]).predictions)
    card = metrics.scorecard(ds["test"]["label"], probs, names)
    trainer.save_model(str(args.out))
    params = sum(p.numel() for p in model.parameters())
    metrics.save(
        args.name,
        card,
        val_accuracy=round(val_acc, 4),
        train_s=train_s,
        device=device,
        params=params,
        base=args.model,
    )

if __name__ == "__main__":
    main()
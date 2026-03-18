"""
Fine-tune DeBERTa-v3-large for binary misinformation classification (GPU required).

Memory-optimised for T4 (15 GB VRAM):
  - fp16=True
  - gradient_checkpointing=True
  - batch_size=4, gradient_accumulation=4 (effective batch=16)

OOM fallback: set MODEL_ID = "microsoft/deberta-v3-base" (86M params).

Expected runtime: ~2–3 hrs on T4.
"""
import os
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd

MODEL_ID      = "microsoft/deberta-v3-large"
MODEL_ID_BASE = "microsoft/deberta-v3-base"   # OOM fallback


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


def load_data(project_dir: str):
    """Load normalised (texts, labels); uses augmented data if available."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.data_loader import load_combined_data, records_to_texts_labels

    aug_path = Path(project_dir) / "data" / "augmented" / "augmented_train.json"
    if aug_path.exists():
        import json
        with open(aug_path, encoding="utf-8") as f:
            records = json.load(f)
        texts  = [r["text"]  for r in records]
        labels = [int(r["label"]) for r in records]
        print(f"Loaded augmented data: {len(texts)} samples")
    else:
        records = load_combined_data(project_dir)
        texts, labels = records_to_texts_labels(records)
    return texts, labels


def build_dataset(texts: list[str], labels: list[int],
                  tokenizer, max_length: int = 512) -> Dataset:
    df = pd.DataFrame({"text": texts, "label": labels})
    ds = Dataset.from_pandas(df)
    return ds.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=max_length,
                             padding="max_length"),
        batched=True,
    )


def train(project_dir: str, val_split: float = 0.15, use_base: bool = False) -> None:
    model_id  = MODEL_ID_BASE if use_base else MODEL_ID
    texts, labels = load_data(project_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    tr_texts, vl_texts, tr_labels, vl_labels = train_test_split(
        texts, labels, test_size=val_split, stratify=labels, random_state=42,
    )
    train_ds = build_dataset(tr_texts, tr_labels, tokenizer)
    val_ds   = build_dataset(vl_texts, vl_labels, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=2, ignore_mismatched_sizes=True,
    )

    output_dir = os.path.join(project_dir, "models", "deberta_finetuned")
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,   # effective batch = 16
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        fp16=True,
        gradient_checkpointing=True,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    preds = trainer.predict(val_ds)
    pred_labels = preds.predictions.argmax(axis=1)
    print("\n" + classification_report(vl_labels, pred_labels,
                                        target_names=["False", "True"]))
    print(f"✅ DeBERTa saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--base", action="store_true",
                        help="Use deberta-v3-base instead of large (OOM fallback)")
    args = parser.parse_args()
    train(args.project_dir, use_base=args.base)

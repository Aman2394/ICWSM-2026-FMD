"""
Tier 2 — Fine-tuned Encoder Out-of-Fold Predictions (GPU required).

Generates OOF softmax predictions from fine-tuned FinBERT and DeBERTa classifiers
using 5-fold stratified CV to prevent data leakage into the meta-classifier.

Expected runtime: ~30 min (FinBERT) + ~2–3 hrs (DeBERTa) per run on T4.
"""
import os
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm

FINBERT_MODEL_ID  = "ProsusAI/finbert"
DEBERTA_MODEL_ID  = "microsoft/deberta-v3-large"
DEBERTA_BASE_ID   = "microsoft/deberta-v3-base"   # OOM fallback
N_FOLDS           = 5


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


def tokenize_dataset(texts: list[str], labels: list[int],
                     tokenizer, max_length: int = 512) -> Dataset:
    df = pd.DataFrame({"text": texts, "label": labels})
    ds = Dataset.from_pandas(df)
    return ds.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=max_length,
                             padding="max_length"),
        batched=True,
    )


def get_softmax_preds(trainer: Trainer, dataset: Dataset) -> np.ndarray:
    """Returns softmax probabilities shape (N, 2)."""
    output = trainer.predict(dataset)
    logits = output.predictions
    probs  = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()
    return probs  # (N, 2) → [P(False), P(True)]


def run_oof_finetuning(
    texts: list[str],
    labels: list[int],
    config: dict,
) -> np.ndarray:
    """
    5-fold OOF fine-tuning. Returns OOF softmax predictions shape (N, 2).

    IMPORTANT: These OOF preds prevent leakage when used as meta-classifier features.

    config keys:
        model_name       : HuggingFace model ID (required)
        output_dir       : directory to save fold checkpoints (required)
        n_splits         : number of CV folds (default 5)
        random_state     : random seed (default 42)
        + any TrainingArguments kwargs (num_train_epochs, learning_rate, fp16, etc.)
    """
    config = dict(config)  # copy so we don't mutate caller's dict
    model_id   = config.pop("model_name")
    output_dir = config.pop("output_dir")
    n_splits   = config.pop("n_splits", N_FOLDS)
    random_state = config.pop("random_state", 42)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_preds = np.zeros((len(texts), 2), dtype=np.float32)

    # Remaining config keys are passed directly to TrainingArguments
    training_args_kwargs = dict(
        save_total_limit=1,
        **config,
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        print(f"\n── Fold {fold + 1}/{n_splits} ──")
        train_texts  = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts    = [texts[i] for i in val_idx]
        val_labels   = [labels[i] for i in val_idx]

        train_ds = tokenize_dataset(train_texts, train_labels, tokenizer)
        val_ds   = tokenize_dataset(val_texts,   val_labels,   tokenizer)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=2, ignore_mismatched_sizes=True,
        )

        fold_output_dir = os.path.join(output_dir, f"fold{fold}")
        args = TrainingArguments(output_dir=fold_output_dir, **training_args_kwargs)

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
        )
        trainer.train()

        fold_preds = get_softmax_preds(trainer, val_ds)
        oof_preds[val_idx] = fold_preds

        acc = accuracy_score(val_labels, fold_preds.argmax(axis=1))
        print(f"   Fold {fold + 1} val accuracy: {acc:.4f}")

        # Free GPU memory
        del model, trainer
        torch.cuda.empty_cache()

    overall_acc = accuracy_score(labels, oof_preds.argmax(axis=1))
    print(f"\n✅ OOF accuracy ({model_id}): {overall_acc:.4f}")
    return oof_preds


def inference_softmax_preds(texts: list[str], model_dir: str,
                             device: str = "cuda",
                             batch_size: int = 32) -> np.ndarray:
    """
    Run inference with a saved fine-tuned model on blind/test data.
    Returns softmax probabilities shape (N, 2) → [P(False), P(True)].

    Use this at inference time instead of run_oof_finetuning().
    model_dir: path to saved model (e.g. models/finbert_finetuned/).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device).eval()

    all_probs = []
    from tqdm import tqdm
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Inference {model_dir.split('/')[-1]}"):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                           max_length=512, padding=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)

    torch.cuda.empty_cache()
    return np.vstack(all_probs).astype(np.float32)

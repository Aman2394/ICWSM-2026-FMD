"""
Blind-set inference pipeline.

Runs all three tiers on unseen test data and outputs predictions using
the saved meta-model. All fitted objects (PCA extractor, fine-tuned models,
meta-model) are loaded from disk — nothing is refit on test data.

Usage:
    python src/predict.py \
        --project-dir /path/to/fmd \
        --input data/raw/blind_test.json \
        --output results/predictions/blind_predictions.csv

Input JSON schema (same as training data):
    [{"index": 0, "Open-ended Verifiable Question": "...", ...}, ...]
    OR already-normalised: [{"id": 0, "text": "...", "label": null}, ...]
"""
import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.data_loader import load_raw_file, extract_paragraph
from features.tier1_nli import extract_nli_feature_matrix, FEATURE_NAMES as NLI_NAMES
from features.tier1_embeddings import extract_all_embeddings, EmbeddingDistanceExtractor, FEATURE_NAMES as EMB_NAMES
from features.tier1_coherence import extract_coherence_feature_matrix, FEATURE_NAMES as COH_NAMES
from features.tier1_epistemic import extract_epistemic_feature_matrix, FEATURE_NAMES as EPI_NAMES
from features.tier2_encoder import inference_softmax_preds
from features.tier3_perplexity import extract_perplexity_feature_matrix, FEATURE_NAMES as PP_NAMES


def load_blind_data(input_path: str) -> tuple[list[str], list]:
    """
    Load blind test data. Handles both raw RFC-BENCH schema and
    pre-normalised {text, id} schema. Returns (texts, ids).
    """
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    if "Open-ended Verifiable Question" in data[0]:
        # Raw RFC-BENCH schema — strip prompt prefix
        texts = [extract_paragraph(d["Open-ended Verifiable Question"]) for d in data]
        ids   = [d.get("index", i) for i, d in enumerate(data)]
    else:
        # Pre-normalised schema
        texts = [d["text"] for d in data]
        ids   = [d.get("id", i) for i, d in enumerate(data)]

    print(f"Loaded {len(texts)} blind samples from {input_path}")
    return texts, ids


def run_inference(project_dir: str, input_path: str, output_path: str,
                  device: str = "cuda") -> pd.DataFrame:
    """
    Full inference pipeline for blind test data.

    Steps:
      1. Load blind texts
      2. Tier 1: NLI + Embeddings (using saved extractor) + Coherence + Epistemic
      3. Tier 2: FinBERT + DeBERTa direct inference (using saved models)
      4. Tier 3: MLM perplexity
      5. Stack features → LightGBM meta-model → predictions
    """
    texts, ids = load_blind_data(input_path)
    models_dir   = Path(project_dir) / "models"
    features_dir = Path(project_dir) / "feature_cache"
    results_dir  = Path(project_dir) / "results" / "predictions"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Tier 1 ────────────────────────────────────────────────────────────────
    print("\n── Tier 1: NLI features ──")
    t1_nli = extract_nli_feature_matrix(texts, device=device)

    print("\n── Tier 1: FinBERT embeddings ──")
    embeddings = extract_all_embeddings(texts, device=device)

    # Load fitted PCA extractor from training — do NOT refit on blind data
    extractor_path = str(models_dir / "emb_extractor.pkl")
    extractor = EmbeddingDistanceExtractor.load(extractor_path)
    t1_emb = extractor.transform(embeddings)

    print("\n── Tier 1: Coherence features ──")
    t1_coh = extract_coherence_feature_matrix(texts, device=device)

    print("\n── Tier 1: Epistemic features ──")
    t1_epi = extract_epistemic_feature_matrix(texts)

    t1 = np.hstack([t1_nli, t1_emb, t1_coh, t1_epi])
    print(f"Tier 1 shape: {t1.shape}")

    # ── Tier 2 ────────────────────────────────────────────────────────────────
    print("\n── Tier 2: FinBERT inference ──")
    finbert_probs = inference_softmax_preds(
        texts,
        model_dir=str(models_dir / "finbert_finetuned"),
        device=device,
    )

    print("\n── Tier 2: DeBERTa inference ──")
    deberta_probs = inference_softmax_preds(
        texts,
        model_dir=str(models_dir / "deberta_finetuned"),
        device=device,
    )

    t2 = np.hstack([finbert_probs, deberta_probs])   # (N, 4)
    print(f"Tier 2 shape: {t2.shape}")

    # ── Tier 3 ────────────────────────────────────────────────────────────────
    print("\n── Tier 3: MLM perplexity ──")
    t3 = extract_perplexity_feature_matrix(texts, device=device)
    print(f"Tier 3 shape: {t3.shape}")

    # ── Meta-classifier ───────────────────────────────────────────────────────
    X = np.hstack([t1, t2, t3])
    print(f"\n── Meta-classifier: combined features {X.shape} ──")

    with open(models_dir / "meta_model.pkl", "rb") as f:
        meta_model = pickle.load(f)

    preds      = meta_model.predict(X)          # 0 = False, 1 = True
    pred_probs = meta_model.predict_proba(X)    # (N, 2) → [P(False), P(True)]

    # ── Save predictions ──────────────────────────────────────────────────────
    df = pd.DataFrame({
        "id":           ids,
        "prediction":   preds,
        "label":        ["True" if p == 1 else "False" for p in preds],
        "prob_true":    pred_probs[:, 1].round(4),
        "prob_false":   pred_probs[:, 0].round(4),
    })

    # ── Metrics (only when ground-truth labels are available) ─────────────────
    true_labels = [r.get("label") for r in _load_records_for_metrics(input_path)]
    if all(l is not None for l in true_labels):
        df = _add_metrics(df, preds, pred_probs, true_labels, output_path)
    else:
        print("ℹ️  No ground-truth labels found — skipping metrics.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    n_true  = (preds == 1).sum()
    n_false = (preds == 0).sum()
    print(f"\n✅ Predictions saved to {output_path}")
    print(f"   True:  {n_true} ({n_true/len(preds):.1%})")
    print(f"   False: {n_false} ({n_false/len(preds):.1%})")
    return df


def _load_records_for_metrics(input_path: str) -> list[dict]:
    """Return raw records so we can check for ground-truth labels."""
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    if "Ground-True Answer" in data[0]:
        from utils.data_loader import parse_label
        return [{"label": parse_label(d["Ground-True Answer"])} for d in data]
    return [{"label": d.get("label")} for d in data]


def _add_metrics(df: pd.DataFrame, preds: np.ndarray, pred_probs: np.ndarray,
                 true_labels: list, output_path) -> pd.DataFrame:
    """Compute and print full metrics; add true_label column to df."""
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, classification_report, roc_auc_score,
                                  confusion_matrix)

    y_true = np.array([int(l) for l in true_labels])
    df["true_label"] = y_true
    df["correct"]    = (preds == y_true).astype(int)

    acc  = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec  = recall_score(y_true, preds, zero_division=0)
    f1   = f1_score(y_true, preds, zero_division=0)
    f1_macro = f1_score(y_true, preds, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(y_true, pred_probs[:, 1])
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(y_true, preds)

    print("\n── Evaluation Metrics ──────────────────────────────────────────")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Precision       : {prec:.4f}  (True class)")
    print(f"  Recall          : {rec:.4f}  (True class)")
    print(f"  F1 (True class) : {f1:.4f}")
    print(f"  F1 (macro)      : {f1_macro:.4f}")
    print(f"  ROC-AUC         : {auc:.4f}")
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"              Pred False  Pred True")
    print(f"  Actual False    {cm[0,0]:5d}      {cm[0,1]:5d}")
    print(f"  Actual True     {cm[1,0]:5d}      {cm[1,1]:5d}")
    print()
    print(classification_report(y_true, preds, target_names=["False", "True"]))

    # Save metrics JSON alongside predictions
    metrics = {
        "accuracy": round(acc, 4), "precision": round(prec, 4),
        "recall": round(rec, 4), "f1": round(f1, 4),
        "f1_macro": round(f1_macro, 4), "roc_auc": round(auc, 4),
        "n_samples": len(y_true),
        "confusion_matrix": cm.tolist(),
    }
    metrics_path = Path(str(output_path).replace(".csv", "_metrics.json"))
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved to {metrics_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on blind test data.")
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--input",  required=True, help="Path to test/blind JSON")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    df = run_inference(
        project_dir=args.project_dir,
        input_path=args.input,
        output_path=args.output,
        device=args.device,
    )
    print(df.head(10).to_string(index=False))

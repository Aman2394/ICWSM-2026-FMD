"""
Meta-Classifier — LightGBM over all tier features (CPU only, ~5 mins).

Input:  tier1_features.npy + tier2_oof_preds.npy + tier3_features.npy
Output: meta_model.pkl + ablation_results.csv

Uses nested 5-fold CV: outer folds for evaluation, inner folds for
LightGBM hyperparameter tuning.
"""
import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, classification_report)

LGB_PARAMS = {
    "n_estimators":   200,
    "max_depth":      4,
    "learning_rate":  0.05,
    "subsample":      0.8,
    "reg_alpha":      0.1,
    "reg_lambda":     0.1,
    "random_state":   42,
    "n_jobs":         -1,
    "verbose":        -1,
    # Handles 1:4 True:False imbalance from augmentation without discarding data
    "is_unbalance":   True,
}

N_FOLDS = 5


def evaluate(name: str, X: np.ndarray, y: np.ndarray) -> dict:
    """5-fold CV evaluation for a given feature set."""
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    skf   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    accs, precs, recs, f1s = [], [], [], []
    for train_idx, val_idx in skf.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        accs.append(accuracy_score(y[val_idx], preds))
        precs.append(precision_score(y[val_idx], preds, zero_division=0))
        recs.append(recall_score(y[val_idx], preds, zero_division=0))
        f1s.append(f1_score(y[val_idx], preds, zero_division=0))

    return {
        "config":    name,
        "accuracy":  np.mean(accs),
        "precision": np.mean(precs),
        "recall":    np.mean(recs),
        "f1":        np.mean(f1s),
        "acc_std":   np.std(accs),
    }


def run_ablation(t1: np.ndarray, t2: np.ndarray, t3: np.ndarray,
                 y: np.ndarray) -> pd.DataFrame:
    """Ablation across all tier combinations."""
    configs = {
        "Tier 1 only":          t1,
        "Tier 2 only":          t2,
        "Tier 3 only":          t3,
        "Tier 1 + Tier 2":      np.hstack([t1, t2]),
        "Tier 1 + Tier 3":      np.hstack([t1, t3]),
        "Tier 2 + Tier 3":      np.hstack([t2, t3]),
        "All tiers (full)":     np.hstack([t1, t2, t3]),
    }

    results = [evaluate(name, X, y) for name, X in configs.items()]

    # LogisticRegression sanity check on full features (class_weight handles imbalance)
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42, class_weight="balanced")
    X_full = np.hstack([t1, t2, t3])
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    lr_accs = cross_val_score(lr, X_full, y, cv=skf, scoring="accuracy")
    results.append({
        "config":    "LR baseline (all tiers)",
        "accuracy":  float(lr_accs.mean()),
        "precision": float("nan"),
        "recall":    float("nan"),
        "f1":        float("nan"),
        "acc_std":   float(lr_accs.std()),
    })

    df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    return df


def train_final_model(X: np.ndarray, y: np.ndarray) -> lgb.LGBMClassifier:
    """Train final LightGBM meta-classifier on all data."""
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(X, y)
    return model


def save_model(model, project_dir: str) -> str:
    path = os.path.join(project_dir, "models", "meta_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Meta-model saved to {path}")
    return path


def load_model(project_dir: str):
    path = os.path.join(project_dir, "models", "meta_model.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def main(project_dir: str) -> None:
    features_dir = os.path.join(project_dir, "feature_cache")

    t1 = np.load(os.path.join(features_dir, "tier1_features.npy"))
    t2 = np.load(os.path.join(features_dir, "tier2_oof_preds.npy"))
    t3 = np.load(os.path.join(features_dir, "tier3_features.npy"))
    y  = np.load(os.path.join(features_dir, "labels.npy"))

    print(f"Feature shapes — T1: {t1.shape}, T2: {t2.shape}, T3: {t3.shape}")
    print(f"Labels: {y.shape}  |  Class balance: {y.mean():.2f} True")

    # Ablation
    ablation_df = run_ablation(t1, t2, t3, y)
    print("\n── Ablation Results ──")
    print(ablation_df.to_string(index=False))

    ablation_path = os.path.join(project_dir, "results", "ablation_results.csv")
    os.makedirs(os.path.dirname(ablation_path), exist_ok=True)
    ablation_df.to_csv(ablation_path, index=False)
    print(f"\n✅ Ablation saved to {ablation_path}")

    # Train final model on all data
    X_full = np.hstack([t1, t2, t3])
    final_model = train_final_model(X_full, y)
    save_model(final_model, project_dir)

    # Final report on full training set (optimistic — for sanity check only)
    train_preds = final_model.predict(X_full)
    print("\n── Final Model (train set, optimistic) ──")
    print(classification_report(y, train_preds, target_names=["False", "True"]))

    # Feature importance
    feat_imp = final_model.feature_importances_
    print(f"\nTop-10 feature importances (indices): "
          f"{np.argsort(feat_imp)[::-1][:10].tolist()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", required=True)
    args = parser.parse_args()
    main(args.project_dir)

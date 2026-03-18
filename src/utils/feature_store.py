"""
Persistent feature store: save/load .npy feature arrays and labels.
Always saves to Drive in Colab to survive session disconnects.
"""
import os
import numpy as np


def save_features(array: np.ndarray, name: str, project_dir: str) -> str:
    """Save a numpy array to features/ and print confirmation."""
    features_dir = os.path.join(project_dir, "feature_cache")
    os.makedirs(features_dir, exist_ok=True)
    path = os.path.join(features_dir, f"{name}.npy")
    np.save(path, array)
    print(f"✅ Saved {name}.npy — shape {array.shape} — {path}")
    return path


def load_features(name: str, project_dir: str) -> np.ndarray:
    """Load a numpy array from features/."""
    path = os.path.join(project_dir, "feature_cache", f"{name}.npy")
    arr = np.load(path)
    print(f"📂 Loaded {name}.npy — shape {arr.shape}")
    return arr


def save_all_tiers(t1: np.ndarray, t2: np.ndarray, t3: np.ndarray,
                   labels: np.ndarray, project_dir: str) -> np.ndarray:
    """Concatenate all tiers, save individually and combined."""
    save_features(t1, "tier1_features", project_dir)
    save_features(t2, "tier2_oof_preds", project_dir)
    save_features(t3, "tier3_features", project_dir)
    save_features(labels, "labels", project_dir)
    X = np.hstack([t1, t2, t3])
    save_features(X, "X_combined", project_dir)
    print(f"✅ Combined feature matrix: {X.shape}")
    return X


def load_all_tiers(project_dir: str):
    """Load all tier feature arrays and labels. Returns (X, y)."""
    t1 = load_features("tier1_features", project_dir)
    t2 = load_features("tier2_oof_preds", project_dir)
    t3 = load_features("tier3_features", project_dir)
    y  = load_features("labels", project_dir)
    X  = np.hstack([t1, t2, t3])
    print(f"✅ Feature matrix: {X.shape}, labels: {y.shape}")
    return X, y

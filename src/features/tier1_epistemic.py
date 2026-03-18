"""
Tier 1 — Epistemic Calibration Features (CPU only).

Rule-based: measures certainty vs. hedging language balance.
"""
import re
import numpy as np

HIGH_CERTAINTY = frozenset({
    "will", "certain", "guaranteed", "definitely", "absolutely",
    "clearly", "undoubtedly", "must", "always", "never",
})
HEDGES = frozenset({
    "may", "might", "could", "would", "perhaps", "possibly", "suggest",
    "indicate", "appear", "seem", "likely", "unlikely", "some", "often",
})

FEATURE_NAMES = [
    "epi_certainty_ratio",
    "epi_hedge_density",
    "epi_certainty_evidence_mismatch",
    "epi_epistemic_density",
]


def extract_epistemic_features(text: str) -> dict:
    """
    Compute epistemic calibration features from word-level lexicon matching.
    """
    words = re.findall(r"\b\w+\b", text.lower())
    n = len(words) + 1e-8

    high_cert = sum(1 for w in words if w in HIGH_CERTAINTY)
    hedge     = sum(1 for w in words if w in HEDGES)
    epistemic = high_cert + hedge

    return {
        "epi_certainty_ratio":             float(high_cert / (epistemic + 1e-8)),
        "epi_hedge_density":               float(hedge / n),
        "epi_certainty_evidence_mismatch": float(int(high_cert > 2 and hedge < 1)),
        "epi_epistemic_density":           float(epistemic / n),
    }


def extract_epistemic_feature_matrix(texts: list[str]) -> np.ndarray:
    """
    Returns ndarray of shape (N, len(FEATURE_NAMES)).
    CPU only — very fast.
    """
    rows = [
        [extract_epistemic_features(t)[k] for k in FEATURE_NAMES]
        for t in texts
    ]
    return np.array(rows, dtype=np.float32)

"""
Tier 1 — NLI Internal Consistency Features (GPU required).

Uses cross-encoder/nli-deberta-v3-large to score all sentence pairs
within a paragraph and derive contradiction / entailment signals.

Expected runtime: ~2–3 hrs for 2,000 samples on T4.
"""
import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import CrossEncoder

NLI_MODEL_ID = "cross-encoder/nli-deberta-v3-large"
FEATURE_NAMES = [
    "nli_contradiction_ratio",
    "nli_max_contradiction_score",
    "nli_entailment_ratio",
    "nli_coherence_score",
    "nli_weighted_contradiction",
    "nli_sentence_count",       # explicit count so model knows when NLI is meaningful
    "nli_has_multiple_sents",   # binary flag: 1 if ≥2 sentences, 0 if single
]


def load_nli_model(device: str = "cuda") -> CrossEncoder:
    return CrossEncoder(NLI_MODEL_ID, device=device)


def extract_nli_features(text: str, model: CrossEncoder) -> dict:
    """
    Scores all sentence pairs in a paragraph using NLI.

    Returns dict with keys matching FEATURE_NAMES.
    scores shape from model.predict: (n_pairs, 3)
        col 0 → contradiction
        col 1 → entailment
        col 2 → neutral
    """
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        # Single sentence: no pairs to score. Contradiction = 0 (none found),
        # coherence = 1.0 (a single sentence is trivially self-consistent).
        return {
            "nli_contradiction_ratio":      0.0,
            "nli_max_contradiction_score":  0.0,
            "nli_entailment_ratio":         0.0,
            "nli_coherence_score":          1.0,  # not 0.0 — single sentence is coherent
            "nli_weighted_contradiction":   0.0,
            "nli_sentence_count":           float(len(sentences)),
            "nli_has_multiple_sents":       0.0,
        }

    pairs = [
        (sentences[i], sentences[j])
        for i in range(len(sentences))
        for j in range(i + 1, len(sentences))
    ]

    scores = model.predict(pairs, apply_softmax=True)
    contra = scores[:, 0]
    entail = scores[:, 1]

    return {
        "nli_contradiction_ratio":      float((contra > 0.5).mean()),
        "nli_max_contradiction_score":  float(contra.max()),
        "nli_entailment_ratio":         float((entail > 0.5).mean()),
        "nli_coherence_score":          float(entail.sum() / (entail.sum() + contra.sum() + 1e-8)),
        "nli_weighted_contradiction":   float(contra.sum()),
        "nli_sentence_count":           float(len(sentences)),
        "nli_has_multiple_sents":       1.0,
    }


def extract_nli_feature_matrix(texts: list[str], device: str = "cuda",
                                batch_size: int = 32) -> np.ndarray:
    """
    Extract NLI features for a list of texts.

    Returns ndarray of shape (N, len(FEATURE_NAMES)).
    """
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    model = load_nli_model(device=device)
    rows = []
    from tqdm import tqdm
    for text in tqdm(texts, desc="NLI features"):
        feat = extract_nli_features(text, model)
        rows.append([feat[k] for k in FEATURE_NAMES])

    return np.array(rows, dtype=np.float32)

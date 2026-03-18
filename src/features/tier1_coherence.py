"""
Tier 1 — Discourse Coherence Features (GPU required).

Uses sentence-transformers/all-mpnet-base-v2 to embed individual sentences
and measure sequential similarity within a paragraph.

Expected runtime: ~5 mins on T4.
"""
import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

SENT_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"

FEATURE_NAMES = [
    "coh_consecutive_similarity_min",
    "coh_consecutive_similarity_std",
    "coh_first_last_similarity",
    "coh_mean_coherence",
]


def load_sent_model(device: str = "cuda") -> SentenceTransformer:
    return SentenceTransformer(SENT_MODEL_ID, device=device)


def extract_coherence_features(text: str, model: SentenceTransformer) -> dict:
    """
    Measures discourse coherence via sentence embedding similarities.
    Single-sentence paragraphs receive max coherence scores (no incoherence).
    """
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return {
            "coh_consecutive_similarity_min": 1.0,
            "coh_consecutive_similarity_std": 0.0,
            "coh_first_last_similarity":      1.0,
            "coh_mean_coherence":             1.0,
        }

    embs = model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)

    consec_sims = [
        float(cosine_similarity([embs[i]], [embs[i + 1]])[0][0])
        for i in range(len(embs) - 1)
    ]

    return {
        "coh_consecutive_similarity_min": float(min(consec_sims)),
        "coh_consecutive_similarity_std": float(np.std(consec_sims)),
        "coh_first_last_similarity":      float(cosine_similarity([embs[0]], [embs[-1]])[0][0]),
        "coh_mean_coherence":             float(np.mean(consec_sims)),
    }


def extract_coherence_feature_matrix(texts: list[str],
                                      device: str = "cuda") -> np.ndarray:
    """
    Returns ndarray of shape (N, len(FEATURE_NAMES)).
    """
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    model = load_sent_model(device=device)
    rows = []
    from tqdm import tqdm
    for text in tqdm(texts, desc="Coherence features"):
        feat = extract_coherence_features(text, model)
        rows.append([feat[k] for k in FEATURE_NAMES])
    return np.array(rows, dtype=np.float32)

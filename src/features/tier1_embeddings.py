"""
Tier 1 — FinBERT CLS Embeddings + Distance Features (GPU required).

Pipeline:
1. Extract CLS embeddings for all samples.
2. Fit PCA(64) on True-class training embeddings.
3. Compute centroid of True-class embeddings in PCA space.
4. For each sample: cosine distance + Mahalanobis distance from True centroid.

The fitted EmbeddingDistanceExtractor must be saved after training and loaded
at inference time — never refit on blind/test data.

Expected runtime: ~10 mins on T4.
"""
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist, mahalanobis

FINBERT_MODEL_ID = "ProsusAI/finbert"
PCA_COMPONENTS    = 64

FEATURE_NAMES = [
    "emb_cosine_dist_from_true",
    "emb_mahalanobis_dist_from_true",
]


def load_finbert(device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_ID)
    model = AutoModel.from_pretrained(FINBERT_MODEL_ID).to(device).eval()
    return tokenizer, model


def get_cls_embedding(text: str, tokenizer, model, device: str = "cuda") -> np.ndarray:
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=512, padding=True,
    ).to(device)
    with torch.no_grad():
        out = model(**inputs)
    return out.last_hidden_state[:, 0, :].cpu().numpy().squeeze()  # (768,)


def extract_all_embeddings(texts: list[str], device: str = "cuda") -> np.ndarray:
    """Returns raw CLS embeddings shape (N, 768)."""
    tokenizer, model = load_finbert(device)
    embeddings = []
    from tqdm import tqdm
    for text in tqdm(texts, desc="FinBERT CLS embeddings"):
        emb = get_cls_embedding(text, tokenizer, model, device)
        embeddings.append(emb)
    return np.array(embeddings, dtype=np.float32)


class EmbeddingDistanceExtractor:
    """
    Fits on True-class training embeddings, then transforms any split.

    IMPORTANT: Save this object after training with save() and load it
    at inference time with load() — never refit on blind/test data.
    """

    def __init__(self, n_components: int = PCA_COMPONENTS):
        self.pca = PCA(n_components=n_components)
        self.centroid: np.ndarray | None = None
        self.cov_inv: np.ndarray | None = None

    def fit(self, true_embeddings: np.ndarray) -> "EmbeddingDistanceExtractor":
        """Fit PCA and compute centroid + inverse covariance on True-class embs."""
        pca_embs = self.pca.fit_transform(true_embeddings)   # (N_true, 64)
        self.centroid = pca_embs.mean(axis=0)                # (64,)
        cov = np.cov(pca_embs, rowvar=False) + np.eye(PCA_COMPONENTS) * 1e-6
        self.cov_inv = np.linalg.inv(cov)
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Returns shape (N, 2): [cosine_dist, mahalanobis_dist]."""
        assert self.centroid is not None, "Call fit() first."
        pca_embs = self.pca.transform(embeddings)   # (N, 64)

        cosine_dists = cdist(pca_embs, self.centroid[np.newaxis, :],
                             metric="cosine").squeeze()
        maha_dists = np.array([
            mahalanobis(e, self.centroid, self.cov_inv)
            for e in pca_embs
        ])

        return np.stack([cosine_dists, maha_dists], axis=1).astype(np.float32)

    def fit_transform(self, embeddings: np.ndarray,
                      labels: np.ndarray) -> np.ndarray:
        """Convenience: fit on True-class subset, transform all."""
        true_mask = labels == 1
        self.fit(embeddings[true_mask])
        return self.transform(embeddings)

    def save(self, path: str) -> None:
        """Persist fitted extractor to disk."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"✅ EmbeddingDistanceExtractor saved to {path}")

    @classmethod
    def load(cls, path: str) -> "EmbeddingDistanceExtractor":
        """Load a previously fitted extractor — use this at inference time."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"📂 EmbeddingDistanceExtractor loaded from {path}")
        return obj

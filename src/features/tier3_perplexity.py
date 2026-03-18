"""
Tier 3 — MLM Perplexity Features via FinBERT (GPU required).

Masks each token one at a time and records the negative log-probability
of the original token — a proxy for how "expected" the text is under
the financial language model.

Expected runtime: ~2 hrs on T4 (token-by-token, sequential).
"""
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm

FINBERT_MODEL_ID = "ProsusAI/finbert"

FEATURE_NAMES = [
    "pp_mean_perplexity",
    "pp_max_perplexity",
    "pp_std_perplexity",
    "pp_top10pct_ratio",
]


def load_mlm_model(device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_ID)
    model = AutoModelForMaskedLM.from_pretrained(FINBERT_MODEL_ID).to(device).eval()
    return tokenizer, model


def mlm_perplexity_features(text: str, tokenizer, model,
                             device: str = "cuda") -> dict:
    """
    Token-by-token MLM perplexity.
    Skips [CLS] and [SEP] tokens.
    """
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512,
    ).to(device)
    input_ids = inputs["input_ids"].squeeze()   # (seq_len,)

    if len(input_ids) < 3:   # [CLS] text [SEP] — needs at least 1 real token
        return {k: 0.0 for k in FEATURE_NAMES}

    token_nlls = []
    for i in range(1, len(input_ids) - 1):   # skip [CLS] and [SEP]
        masked_ids = input_ids.clone()
        masked_ids[i] = tokenizer.mask_token_id

        with torch.no_grad():
            logits = model(masked_ids.unsqueeze(0)).logits   # (1, seq, vocab)

        probs     = torch.softmax(logits[0, i], dim=-1)
        token_prob = probs[input_ids[i]].item()
        token_nlls.append(-np.log(token_prob + 1e-10))

    pp = np.array(token_nlls, dtype=np.float32)

    p90 = np.percentile(pp, 90)
    top10 = pp[pp > p90]
    top10_ratio = float(top10.mean() / (pp.mean() + 1e-8)) if len(top10) > 0 else 1.0

    return {
        "pp_mean_perplexity": float(pp.mean()),
        "pp_max_perplexity":  float(pp.max()),
        "pp_std_perplexity":  float(pp.std()),
        "pp_top10pct_ratio":  top10_ratio,
    }


def extract_perplexity_feature_matrix(texts: list[str],
                                       device: str = "cuda") -> np.ndarray:
    """
    Returns ndarray of shape (N, len(FEATURE_NAMES)).
    Saves a checkpoint every 100 samples to stderr progress.
    """
    tokenizer, model = load_mlm_model(device=device)
    rows = []
    for text in tqdm(texts, desc="MLM perplexity"):
        feat = mlm_perplexity_features(text, tokenizer, model, device)
        rows.append([feat[k] for k in FEATURE_NAMES])
    return np.array(rows, dtype=np.float32)

# CLAUDE.md — Financial Misinformation Detection (FMD)
## ICWSM 2026 Shared Task

> **Framing:** This is NOT "detect GPT-4.1 editing artifacts."  
> It is: *"Does this paragraph exhibit properties of well-formed, internally consistent financial journalism?"*

---

## Project Overview

**Task:** Binary classification of standalone financial news paragraphs → `True` (original) or `False` (misinformation).  
**Dataset:** RFC-BENCH — 1,000 SFT + 1,000 RL samples, balanced 50/50.  
**Eval Metrics:** Accuracy, Precision, Recall, F1.  
**Constraint:** Single T4 GPU (Google Colab compatible).

---

## Architecture: Three-Tier Feature Ensemble + LightGBM Meta-Classifier

```
Input Paragraph
      │
      ├──► Tier 1: Perturbation-Agnostic Features   (generalizes to ANY misinformation)
      │         ├── NLI Consistency (DeBERTa NLI)
      │         ├── FinBERT CLS Embeddings + Distances
      │         ├── Discourse Coherence (sentence-transformers)
      │         └── Epistemic Calibration (rule-based)
      │
      ├──► Tier 2: Learned Encoder Representations  (generalizes to most misinformation)
      │         ├── Fine-tuned FinBERT classifier logits
      │         └── Fine-tuned DeBERTa-v3-large classifier logits
      │
      └──► Tier 3: Dataset-Specific Auxiliary Features  (this benchmark only)
                ├── MLM Perplexity (FinBERT masked LM)
                └── Perturbation-type features (rule-based)

All tiers (96-dim vector) → LightGBM Meta-Classifier (nested 5-fold CV)
```

---

## Repository Structure

```
fmd/
├── CLAUDE.md                        # This file
├── data/
│   ├── raw/
│   │   ├── train_sft.json
│   │   └── train_rl.json
│   └── augmented/
│       └── augmented_train.json     # Output of augmentation pipeline
├── features/
│   ├── tier1_features.npy
│   ├── tier2_oof_preds.npy
│   └── tier3_features.npy
├── models/
│   ├── finbert_finetuned/           # Saved fine-tuned FinBERT
│   ├── deberta_finetuned/           # Saved fine-tuned DeBERTa
│   └── meta_model.pkl               # LightGBM meta-classifier
├── notebooks/
│   ├── 01_data_augmentation.ipynb   # Colab: CPU, API calls only
│   ├── 02_tier1_features.ipynb      # Colab: GPU required (NLI inference)
│   ├── 03_tier2_finetuning.ipynb    # Colab: GPU required
│   ├── 04_tier3_features.ipynb      # Colab: GPU required (MLM perplexity)
│   ├── 05_meta_classifier.ipynb     # Colab: CPU, LightGBM only
│   ├── 06_ablation.ipynb            # Colab: CPU (uses saved features)
│   └── check_regeneration.ipynb     # Quality check for augmented data
├── src/
│   ├── augmentation/
│   │   ├── call_llm.py
│   │   └── perturbation_prompts.py
│   ├── features/
│   │   ├── tier1_nli.py
│   │   ├── tier1_embeddings.py
│   │   ├── tier1_coherence.py
│   │   ├── tier1_epistemic.py
│   │   ├── tier2_encoder.py
│   │   └── tier3_perplexity.py
│   ├── models/
│   │   ├── finetune_finbert.py
│   │   ├── finetune_deberta.py
│   │   └── meta_classifier.py
│   └── utils/
│       ├── colab_setup.py           # Drive mounting, GPU checks, installs
│       └── feature_store.py         # Save/load .npy feature arrays
├── results/
│   ├── ablation_results.csv
│   └── predictions/
└── requirements.txt
```

---

## Environment Setup

### Google Colab — Standard Header (all GPU notebooks)

```python
# ── Cell 1: Colab Setup ──────────────────────────────────────────────────────
import subprocess, sys, os

# Mount Drive for persistent storage of models & features
from google.colab import drive
drive.mount('/content/drive')

PROJECT_DIR = "/content/drive/MyDrive/fmd"
os.makedirs(PROJECT_DIR, exist_ok=True)
os.chdir(PROJECT_DIR)

# Verify GPU
import torch
assert torch.cuda.is_available(), "⚠️  No GPU detected. Runtime → Change runtime type → T4 GPU"
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Install dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.40", "sentence-transformers", "lightgbm",
    "datasets", "accelerate", "bitsandbytes", "scikit-learn",
    "pandas", "numpy", "tqdm"
])
```

### Local (non-Colab)

```bash
conda create -n fmd python=3.11
conda activate fmd
pip install -r requirements.txt
```

### `requirements.txt`

```
transformers>=4.40.0
sentence-transformers>=2.7.0
lightgbm>=4.3.0
datasets>=2.19.0
accelerate>=0.30.0
scikit-learn>=1.4.0
pandas>=2.2.0
numpy>=1.26.0
torch>=2.2.0
tqdm
scipy
```

---

## GPU Components — Colab Notes

All GPU-bound steps are isolated to specific notebooks. Use **T4 GPU runtime** for these.

| Notebook | GPU? | Est. Time (T4) | Key Models |
|---|---|---|---|
| `01_data_augmentation` | ❌ CPU | ~2–4 hrs (API) | LLM API calls only |
| `02_tier1_features` | ✅ GPU | ~2–3 hrs | `cross-encoder/nli-deberta-v3-large` |
| `03_tier2_finetuning` | ✅ GPU | ~3–4 hrs | FinBERT + DeBERTa-v3-large |
| `04_tier3_features` | ✅ GPU | ~2 hrs | FinBERT MLM head |
| `05_meta_classifier` | ❌ CPU | ~5 mins | LightGBM |
| `06_ablation` | ❌ CPU | ~10 mins | Uses saved `.npy` files |

### Colab Session Management

> ⚠️ **Colab disconnects lose in-memory state.** Always save checkpoints to Drive after each major step.

```python
# Save features immediately after extraction
import numpy as np
np.save(f"{PROJECT_DIR}/features/tier1_features.npy", tier1_features)
np.save(f"{PROJECT_DIR}/features/tier1_labels.npy", labels)
print("✅ Features saved to Drive")
```

---

## Tier 1 Features — Implementation Guide

### 1.1 NLI Internal Consistency (`tier1_nli.py`)

```python
# GPU required — ~2–3 hrs for full dataset on T4
from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/nli-deberta-v3-large", device="cuda")

def extract_nli_features(text: str) -> dict:
    sentences = sent_tokenize(text)
    pairs = [(sentences[i], sentences[j])
             for i in range(len(sentences))
             for j in range(i+1, len(sentences))]
    if not pairs:
        return {k: 0.0 for k in ["contradiction_ratio", "max_contradiction_score",
                                   "entailment_ratio", "coherence_score", "weighted_contradiction"]}
    scores = model.predict(pairs, apply_softmax=True)
    # scores shape: (n_pairs, 3) → [contradiction, entailment, neutral]
    contra = scores[:, 0]
    entail = scores[:, 1]
    return {
        "contradiction_ratio":      (contra > 0.5).mean(),
        "max_contradiction_score":  contra.max(),
        "entailment_ratio":         (entail > 0.5).mean(),
        "coherence_score":          entail.sum() / (entail.sum() + contra.sum() + 1e-8),
        "weighted_contradiction":   contra.sum(),
    }
```

### 1.2 FinBERT Embeddings + Distances (`tier1_embeddings.py`)

```python
# GPU required — fast (~10 mins on T4)
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
import torch, numpy as np

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModel.from_pretrained("ProsusAI/finbert").cuda().eval()

def get_cls_embedding(text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=512, padding=True).to("cuda")
    with torch.no_grad():
        out = model(**inputs)
    return out.last_hidden_state[:, 0, :].cpu().numpy().squeeze()

# After extracting all embeddings:
# 1. Fit PCA(n_components=64) on training set True-class embeddings
# 2. Compute centroid of True embeddings
# 3. For each sample: cosine distance + Mahalanobis distance from True centroid
```

### 1.3 Discourse Coherence (`tier1_coherence.py`)

```python
# GPU required — fast (~5 mins on T4)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

sent_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")

def extract_coherence_features(text: str) -> dict:
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return {"consecutive_similarity_min": 1.0, "consecutive_similarity_std": 0.0,
                "first_last_similarity": 1.0, "mean_coherence": 1.0}
    embs = sent_model.encode(sentences, convert_to_numpy=True)
    consec_sims = [cosine_similarity([embs[i]], [embs[i+1]])[0][0]
                   for i in range(len(embs)-1)]
    return {
        "consecutive_similarity_min": min(consec_sims),
        "consecutive_similarity_std": np.std(consec_sims),
        "first_last_similarity":      cosine_similarity([embs[0]], [embs[-1]])[0][0],
        "mean_coherence":             np.mean(consec_sims),
    }
```

### 1.4 Epistemic Calibration (`tier1_epistemic.py`)

```python
# CPU only
HIGH_CERTAINTY = {"will", "certain", "guaranteed", "definitely", "absolutely",
                  "clearly", "undoubtedly", "must", "always", "never"}
HEDGES = {"may", "might", "could", "would", "perhaps", "possibly", "suggest",
          "indicate", "appear", "seem", "likely", "unlikely", "some", "often"}

def extract_epistemic_features(text: str) -> dict:
    words = text.lower().split()
    n = len(words) + 1e-8
    high_cert = sum(w in HIGH_CERTAINTY for w in words)
    hedge = sum(w in HEDGES for w in words)
    epistemic = high_cert + hedge
    return {
        "certainty_ratio":              high_cert / (epistemic + 1e-8),
        "hedge_density":                hedge / n,
        "certainty_evidence_mismatch":  int(high_cert > 2 and hedge < 1),
    }
```

---

## Tier 2 Fine-Tuning — Colab Configs

### FinBERT (`finetune_finbert.py`)

```python
# T4 GPU — ~30 min/run
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                           TrainingArguments, Trainer)

MODEL_NAME = "ProsusAI/finbert"
# Full fine-tune: 110M params fits comfortably on T4

training_args = TrainingArguments(
    output_dir=f"{PROJECT_DIR}/models/finbert_finetuned",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    fp16=True,                        # Required for T4
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    report_to="none",
)
```

### DeBERTa-v3-large (`finetune_deberta.py`)

```python
# T4 GPU — ~2–3 hrs/run
# 304M params — needs memory optimizations

training_args = TrainingArguments(
    output_dir=f"{PROJECT_DIR}/models/deberta_finetuned",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,   # Effective batch = 16
    learning_rate=1e-5,
    fp16=True,                        # Required for T4
    gradient_checkpointing=True,      # Required for T4 with 304M params
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    report_to="none",
)

# OOM fallback: if DeBERTa-v3-large OOMs, switch to:
# MODEL_NAME = "microsoft/deberta-v3-base"  # 86M params
```

### Out-of-Fold Predictions (prevent leakage)

```python
from sklearn.model_selection import StratifiedKFold

# CRITICAL: Tier 2 features fed to meta-classifier must be OOF predictions
# to prevent data leakage. Never use in-fold predictions.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(train_df), 2))  # softmax probs [P(False), P(True)]

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    # train on train_idx, predict on val_idx
    # store predictions in oof_preds[val_idx]
    ...

np.save(f"{PROJECT_DIR}/features/tier2_oof_preds.npy", oof_preds)
```

---

## Tier 3 Features — MLM Perplexity

```python
# GPU required — slow (~2 hrs on T4, token-by-token)
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch, numpy as np

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
mlm_model = AutoModelForMaskedLM.from_pretrained("ProsusAI/finbert").cuda().eval()

def mlm_perplexity_features(text: str) -> dict:
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=512).to("cuda")
    input_ids = inputs["input_ids"].squeeze()
    token_perplexities = []

    for i in range(1, len(input_ids) - 1):  # Skip [CLS] and [SEP]
        masked_ids = input_ids.clone()
        masked_ids[i] = tokenizer.mask_token_id
        with torch.no_grad():
            logits = mlm_model(masked_ids.unsqueeze(0)).logits
        probs = torch.softmax(logits[0, i], dim=-1)
        token_prob = probs[input_ids[i]].item()
        token_perplexities.append(-np.log(token_prob + 1e-10))

    pp = np.array(token_perplexities)
    return {
        "mean_perplexity":          pp.mean(),
        "max_perplexity":           pp.max(),
        "std_perplexity":           pp.std(),
        "top_10pct_perplexity_ratio": pp[pp > np.percentile(pp, 90)].mean() / (pp.mean() + 1e-8),
    }
```

---

## Meta-Classifier

```python
# CPU only — fast (~5 mins)
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load all feature arrays
t1 = np.load(f"{PROJECT_DIR}/features/tier1_features.npy")
t2 = np.load(f"{PROJECT_DIR}/features/tier2_oof_preds.npy")
t3 = np.load(f"{PROJECT_DIR}/features/tier3_features.npy")
X  = np.hstack([t1, t2, t3])   # ~96 dimensions
y  = np.load(f"{PROJECT_DIR}/features/labels.npy")

lgb_params = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
}

# Nested CV: outer=eval, inner=hyperparam tuning
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Sanity check: also run LogisticRegression on same features
# If LGB >> LR, non-linear interactions are genuine
lr = LogisticRegression(max_iter=1000, C=1.0)
```

---

## Data Augmentation

```python
# CPU — uses LLM API (no GPU needed)
# Run in 01_data_augmentation.ipynb

PERTURBATION_PROMPTS = {
    "misleading_framing":     "Rewrite using alarming/negative word choice, but change NO facts.",
    "cherry_picked_context":  "Add one true-sounding sentence from a different time period/sector that misleads.",
    "false_attribution":      "Change who said or caused something. Keep the original claim unchanged.",
    "correlation_as_causation": "Rewrite to imply causation where the original only implies coincidence.",
    "omission_bias":          "Remove one qualifying sentence (e.g., 'however...') that changes the overall meaning.",
    "scale_distortion":       "Change magnitude words: 'some'→'most', 'one analyst'→'analysts'.",
    "hedge_manipulation":     "Flip certainty: 'will increase'→'might increase' or vice versa.",
    "detail_hallucination":   "Add one specific but fabricated detail: a fake quote, invented metric, or made-up comparison.",
    "composite":              "Apply two subtle changes simultaneously (e.g., a numerical tweak + a slight sentiment shift).",
}

# Target: 500 True × 3–4 perturbations = ~2,000 False examples
# Final balanced dataset: ~2,000 True + 2,000 False
```

---

## Training Timeline

| Days | Task | Compute | Output |
|---|---|---|---|
| 1–2 | Data augmentation (`01_`) | CPU / API | `augmented_train.json` |
| 3 | Tier 1 feature extraction (`02_`) | T4 GPU ~2–3 hrs | `tier1_features.npy` |
| 4–5 | Tier 2 fine-tuning (`03_`) | T4 GPU ~3–4 hrs | `tier2_oof_preds.npy` |
| 6 | Tier 3 MLM perplexity (`04_`) | T4 GPU ~2 hrs | `tier3_features.npy` |
| 7 | Meta-classifier + ablation (`05_`, `06_`) | CPU | `meta_model.pkl`, `ablation_results.csv` |
| 8–14 | Paper writing | — | `paper.pdf` |

---

## Ablation Targets

| Config | Expected Acc. | Generalizes? |
|---|---|---|
| Random baseline | 50.0% | — |
| FinBERT zero-shot | ~52–55% | ✅ Yes |
| Tier 1 only | ~62–68% | ✅ Yes |
| Tier 1 + Tier 2 | ~70–75% | ✅ Mostly |
| Tier 1 + 2 + augmentation | ~74–78% | ✅ Mostly |
| **Full system (all tiers)** | **~77–82%** | ⚠️ Partially |
| Tier 3 only | ~58–63% | ❌ No |
| Best causal LLM baseline (Qwen3-8B SFT) | ~55–65% | ⚠️ Partially |

---

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| Blind set uses unknown perturbation types | Tier 1+2 carry weight; meta-classifier auto-downweights Tier 3 |
| Different LLM used for generation | NLI consistency + discourse coherence are generation-agnostic |
| Overfitting on 1,000 samples | Nested CV + augmentation + early stopping + Tier 1 inductive bias |
| DeBERTa OOM on T4 | `gradient_checkpointing=True` + `fp16=True` + `batch_size=2`; fallback to `deberta-v3-base` |
| LLM submission required by organizers | Keep `Qwen2.5-7B QLoRA` fallback; use Tier 1 features as prompt context |
| NLI noisy on financial text | Average multiple NLI models; optionally fine-tune on financial pairs |

---

## Key Models Reference

| Model | HuggingFace ID | Use | GPU? |
|---|---|---|---|
| NLI Cross-Encoder | `cross-encoder/nli-deberta-v3-large` | Tier 1 consistency | ✅ |
| FinBERT | `ProsusAI/finbert` | Tier 1 embeddings, Tier 2 classifier, Tier 3 MLM | ✅ |
| Sentence Transformer | `sentence-transformers/all-mpnet-base-v2` | Tier 1 coherence | ✅ |
| DeBERTa-v3-large | `microsoft/deberta-v3-large` | Tier 2 classifier | ✅ |
| LightGBM | — | Meta-classifier | ❌ |

---

## Paper Details

**Venue:** ICWSM 2026 Workshop · **Format:** AAAI 4 pages + appendix  
**Workshop Date:** May 26, 2026

**Title Template:**  
*[TeamName] at the Financial Misinformation Detection Challenge Task: Generalizable Coherence-Based Detection via Multi-Tier Feature Ensembles*

**Core novelty claims:**
1. First system framing reference-free FMD as multi-tier coherence detection with explicit generalizable vs. dataset-specific signal separation.
2. First application of NLI-based internal consistency scoring for financial misinformation.
3. Domain-adapted encoders outperform causal LLMs 3–5× their size on this classification task.
4. Diverse augmentation beyond the four documented perturbation types using real-world misinformation patterns.

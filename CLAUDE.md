# CLAUDE.md — Financial Misinformation Detection (FMD)
## ICWSM 2026 Shared Task

> **Framing:** This is NOT "detect GPT-4.1 editing artifacts."
> It is: *"Does this paragraph exhibit properties of well-formed, internally consistent financial journalism?"*

---

## Project Status

### ✅ Done
- [x] Full project scaffold (`src/`, `notebooks/`, `feature_cache/`, `models/`, `results/`)
- [x] `requirements.txt` + `.venv` created and verified
- [x] GitHub repo: https://github.com/Aman2394/ICWSM-2026-FMD (private)
- [x] **Data loader** (`src/utils/data_loader.py`) — parses RFC-BENCH schema (`Open-ended Verifiable Question` / `Ground-True Answer`), strips prompt prefix, unique IDs (`sft_0`, `rl_0`, ...)
- [x] **Data splitter** (`src/utils/data_splitter.py`) — stratified 70/15/15 train/val/test split on original 2,000 samples; `get_split_records()` filters augmented samples to train split only
- [x] **Augmentation pipeline** (`src/augmentation/`) — 9 perturbation types, Azure OpenAI / OpenAI / Anthropic support, resume/checkpoint, fast-fail on missing API key
- [x] **Data augmentation running** (`01_data_augmentation.ipynb`) — ~35% complete (~350/1000 sources done, ~1,157 augmented samples so far); est. ~81 mins remaining
- [x] **Tier 1 feature extractors** (`src/features/tier1_*.py`) — NLI (5 features), FinBERT embeddings + PCA/distances (2 features), coherence (4 features), epistemic (4 features)
- [x] **`EmbeddingDistanceExtractor`** — fitted on True-class training embeddings; `save()` / `load()` for inference-time reuse (never refit on blind data)
- [x] **Tier 2 OOF fine-tuning** (`src/features/tier2_encoder.py`) — 5-fold stratified CV for FinBERT + DeBERTa; `inference_softmax_preds()` for blind-set inference
- [x] **Tier 3 MLM perplexity** (`src/features/tier3_perplexity.py`) — token-by-token FinBERT MLM (4 features)
- [x] **Fine-tuners** (`src/models/finetune_finbert.py`, `finetune_deberta.py`) — class-weighted loss for 1:4 imbalance, OOM fallback to `deberta-v3-base`
- [x] **Meta-classifier** (`src/models/meta_classifier.py`) — LightGBM (`is_unbalance=True`), ablation across all tier combos, LR sanity check (`class_weight='balanced'`)
- [x] **Inference pipeline** (`src/predict.py`) — full blind-set inference; auto-computes metrics (Accuracy, Precision, Recall, F1, F1-macro, ROC-AUC, confusion matrix) when ground-truth labels are present; saves `_metrics.json` alongside predictions CSV
- [x] **Notebooks 01–07** — Colab-ready, Drive mounting, GPU checks, checkpointing after each step
- [x] **Notebook 07** (`07_predict_blind.ipynb`) — blind-set inference with optional metrics

### 🔄 In Progress
- [ ] **Data augmentation** — ~65% remaining (~81 mins); output → `data/augmented/augmented_train.json`

### ⏳ Pending (in order)
- [ ] **Train/val/test split** — run `data_splitter.make_splits()` after augmentation completes
- [ ] **Notebook 02** — Tier 1 feature extraction on train split (T4 GPU, ~2–3 hrs); saves `feature_cache/tier1_features.npy` + `models/emb_extractor.pkl`
- [ ] **Notebook 03** — Tier 2 OOF fine-tuning FinBERT + DeBERTa (T4 GPU, ~3–4 hrs); saves `feature_cache/tier2_oof_preds.npy`
- [ ] **Notebook 04** — Tier 3 MLM perplexity (T4 GPU, ~2 hrs); saves `feature_cache/tier3_features.npy`
- [ ] **Notebook 05** — Meta-classifier training + ablation (CPU, ~5 mins); saves `models/meta_model.pkl`
- [ ] **Notebook 06** — Detailed ablation analysis + paper-ready numbers (CPU, ~10 mins)
- [ ] **Notebook 07** — Blind-set inference + metrics on held-out test split
- [ ] **Paper writing** (deadline: ICWSM 2026 Workshop, May 26 2026)

---

## Important Implementation Decisions

| Decision | Detail |
|---|---|
| **Class imbalance** | Dataset is 1:4 True:False after augmentation. Handled via `is_unbalance=True` (LightGBM), `class_weight='balanced'` (LR), `CrossEntropyLoss(weight=...)` (fine-tuners). Do NOT downsample. |
| **No leakage** | Train/val/test split on original 2,000 samples first. Augmented samples only go into train. Tier 2 OOF predictions prevent leakage into meta-classifier. |
| **EmbeddingDistanceExtractor** | Fitted on True-class train embeddings. Saved to `models/emb_extractor.pkl`. Must be loaded (not refit) at inference time. |
| **feature_cache/ vs src/features/** | `feature_cache/` = `.npy` data files. `src/features/` = Python source. Never confuse. |
| **Augmented data schema** | `augmented_train.json` contains 2,000 originals + ~3,000 LLM-generated False samples. `perturbation_type=None` marks originals. `source_id` links augmented to original. |
| **Blind set inference** | `predict.py` auto-detects labels. Metrics computed if present (test split eval), skipped if absent (official blind set). |
| **Azure OpenAI** | Use `AzureOpenAI` client with deployment name as model. Env vars: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION`. |

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
      │         ├── Fine-tuned FinBERT classifier logits (OOF)
      │         └── Fine-tuned DeBERTa-v3-large classifier logits (OOF)
      │
      └──► Tier 3: Dataset-Specific Auxiliary Features  (this benchmark only)
                ├── MLM Perplexity (FinBERT masked LM)
                └── Perturbation-type features (rule-based)

All tiers (~15 features) → LightGBM Meta-Classifier (nested 5-fold CV)
```

---

## Repository Structure

```
ICWSM-2026-FMD/
├── CLAUDE.md                        # This file
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── misinfo_SFT_train_for_cot.json   # 1,000 samples (500 True + 500 False)
│   │   └── misinfo_RL_train_for_cot.json    # 1,000 samples (500 True + 500 False)
│   ├── augmented/
│   │   └── augmented_train.json     # 2,000 originals + ~3,000 LLM-augmented False
│   └── splits.json                  # Train/val/test IDs (created after augmentation)
├── feature_cache/                   # .npy feature arrays (NOT src/features/)
│   ├── tier1_features.npy
│   ├── tier2_oof_preds.npy
│   ├── tier3_features.npy
│   ├── labels.npy
│   ├── blind_tier1.npy              # Blind-set features (notebook 07)
│   ├── blind_tier2.npy
│   └── blind_tier3.npy
├── models/
│   ├── emb_extractor.pkl            # Fitted PCA + centroid — load at inference time
│   ├── finbert_finetuned/
│   ├── deberta_finetuned/
│   └── meta_model.pkl
├── notebooks/
│   ├── 01_data_augmentation.ipynb   # CPU — Azure/OpenAI API calls
│   ├── 02_tier1_features.ipynb      # GPU — NLI + embeddings + coherence + epistemic
│   ├── 03_tier2_finetuning.ipynb    # GPU — OOF fine-tuning FinBERT + DeBERTa
│   ├── 04_tier3_features.ipynb      # GPU — MLM perplexity
│   ├── 05_meta_classifier.ipynb     # CPU — LightGBM + ablation
│   ├── 06_ablation.ipynb            # CPU — detailed ablation + paper numbers
│   ├── 07_predict_blind.ipynb       # GPU — blind-set inference + metrics
│   └── check_regeneration.ipynb     # CPU — augmentation quality check
├── src/
│   ├── predict.py                   # Full inference pipeline (CLI + importable)
│   ├── augmentation/
│   │   ├── call_llm.py              # LLM API augmentation (Azure/OpenAI/Anthropic)
│   │   └── perturbation_prompts.py  # 9 perturbation type prompts
│   ├── features/                    # Python source only (NOT .npy data)
│   │   ├── tier1_nli.py
│   │   ├── tier1_embeddings.py      # Includes EmbeddingDistanceExtractor (save/load)
│   │   ├── tier1_coherence.py
│   │   ├── tier1_epistemic.py
│   │   ├── tier2_encoder.py         # OOF training + inference_softmax_preds()
│   │   └── tier3_perplexity.py
│   ├── models/
│   │   ├── finetune_finbert.py
│   │   ├── finetune_deberta.py
│   │   └── meta_classifier.py
│   └── utils/
│       ├── colab_setup.py
│       ├── data_loader.py           # RFC-BENCH schema parser + unique IDs
│       ├── data_splitter.py         # 70/15/15 stratified split + augment filtering
│       └── feature_store.py         # save/load .npy to feature_cache/
└── results/
    ├── ablation_results.csv
    └── predictions/
        ├── blind_predictions.csv
        └── blind_predictions_metrics.json
```

---

## Data Schema (RFC-BENCH)

```json
{
  "index": 0,
  "Open-ended Verifiable Question": "You are a financial misinformation detector.\nPlease check whether the following information is true or false and output the answer [true/false].\n\n\n<PARAGRAPH TEXT>",
  "Ground-True Answer": "The provided information is true.",
  "Instruction": "No"
}
```

- `extract_paragraph()` in `data_loader.py` strips the prompt prefix to get bare paragraph text
- Unique IDs: `sft_0` … `sft_999`, `rl_0` … `rl_999`
- Labels: `"true"` → 1, `"false"` → 0

---

## Environment Setup

### Local
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Jupyter kernel registered as "Python (fmd)"
```

### Google Colab
```python
# Clone repo and mount Drive
!git clone https://<token>@github.com/Aman2394/ICWSM-2026-FMD.git /content/drive/MyDrive/fmd

# Upload raw data files manually to:
# /content/drive/MyDrive/fmd/data/raw/misinfo_SFT_train_for_cot.json
# /content/drive/MyDrive/fmd/data/raw/misinfo_RL_train_for_cot.json
```

### Azure OpenAI Setup (for augmentation)
```python
os.environ['LLM_PROVIDER']             = 'azure'
os.environ['AZURE_OPENAI_API_KEY']     = '<32-char hex key>'
os.environ['AZURE_OPENAI_ENDPOINT']    = 'https://<resource>.openai.azure.com/'
os.environ['AZURE_OPENAI_DEPLOYMENT']  = '<deployment-name>'
os.environ['AZURE_OPENAI_API_VERSION'] = '<YOUR_API_VERSION>'
```

---

## GPU Components — Colab Notes

| Notebook | GPU? | Est. Time (T4) | Status |
|---|---|---|---|
| `01_data_augmentation` | ❌ CPU | ~2–4 hrs (API) | 🔄 ~35% done |
| `02_tier1_features` | ✅ GPU | ~2–3 hrs | ⏳ Pending |
| `03_tier2_finetuning` | ✅ GPU | ~3–4 hrs | ⏳ Pending |
| `04_tier3_features` | ✅ GPU | ~2 hrs | ⏳ Pending |
| `05_meta_classifier` | ❌ CPU | ~5 mins | ⏳ Pending |
| `06_ablation` | ❌ CPU | ~10 mins | ⏳ Pending |
| `07_predict_blind` | ✅ GPU | ~2–3 hrs | ⏳ Pending |

> ⚠️ **Colab disconnects lose in-memory state.** Each notebook saves checkpoints to Drive after every major step. Safe to re-run — all steps are idempotent.

---

## Training Timeline

| Days | Task | Compute | Output |
|---|---|---|---|
| 1–2 | Data augmentation (`01_`) | CPU / API | `augmented_train.json` |
| 2 | Train/val/test split | CPU | `data/splits.json` |
| 3 | Tier 1 feature extraction (`02_`) | T4 GPU ~2–3 hrs | `feature_cache/tier1_features.npy` + `models/emb_extractor.pkl` |
| 4–5 | Tier 2 fine-tuning (`03_`) | T4 GPU ~3–4 hrs | `feature_cache/tier2_oof_preds.npy` |
| 6 | Tier 3 MLM perplexity (`04_`) | T4 GPU ~2 hrs | `feature_cache/tier3_features.npy` |
| 7 | Meta-classifier + ablation (`05_`, `06_`) | CPU | `models/meta_model.pkl`, `results/ablation_results.csv` |
| 7 | Blind-set inference + eval (`07_`) | T4 GPU | `results/predictions/blind_predictions.csv` |
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
| DeBERTa OOM on T4 | `gradient_checkpointing=True` + `fp16=True` + `batch_size=4`; fallback to `deberta-v3-base` |
| Class imbalance (1:4 after augmentation) | `is_unbalance=True` (LGB), `CrossEntropyLoss(weight=...)` (fine-tuners) |
| Augmented samples leaking into val/test | Split on original 2,000 first; `get_split_records()` filters augmented to train only |
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

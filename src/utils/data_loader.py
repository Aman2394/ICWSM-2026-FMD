"""
Data loading utilities for RFC-BENCH dataset.

Schema (both SFT and RL JSON files):
  {
    "index": int,
    "Open-ended Verifiable Question": "You are a financial misinformation detector.\\n...\\n\\n\\n<PARAGRAPH>",
    "Ground-True Answer": "The provided information is true." | "The provided information is false.",
    "Instruction": "No"
  }

Usage:
    from utils.data_loader import load_raw_data, load_combined_data
"""
import json
import re
from pathlib import Path

# Prompt prefix to strip to recover the bare paragraph text
_PROMPT_PREFIX_PATTERN = re.compile(
    r"You are a financial misinformation detector\.\s*"
    r"Please check whether the following information is true or false "
    r"and output the answer \[true/false\]\.\s*",
    re.IGNORECASE,
)


def extract_paragraph(question_field: str) -> str:
    """Strip the task prompt prefix and return the raw paragraph text."""
    text = _PROMPT_PREFIX_PATTERN.sub("", question_field)
    return text.strip()


def parse_label(answer_field: str) -> int:
    """
    'The provided information is true.'  → 1
    'The provided information is false.' → 0
    """
    lower = answer_field.lower()
    if "true" in lower:
        return 1
    if "false" in lower:
        return 0
    raise ValueError(f"Cannot parse label from: {answer_field!r}")


def load_raw_file(path: str, split_tag: str | None = None) -> list[dict]:
    """
    Load one JSON file and return a list of normalised records:
      {"id": int, "text": str, "label": int, "split": str, "original_question": str}
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for item in raw:
        records.append({
            "id":                item["index"],
            "text":              extract_paragraph(item["Open-ended Verifiable Question"]),
            "label":             parse_label(item["Ground-True Answer"]),
            "split":             split_tag or "unknown",
            "original_question": item["Open-ended Verifiable Question"],
        })
    return records


def load_combined_data(project_dir: str) -> list[dict]:
    """
    Load and merge both train_sft and train_rl files.
    Returns list of normalised records (SFT first, then RL).
    """
    raw_dir = Path(project_dir) / "data" / "raw"
    sft_path = raw_dir / "misinfo_SFT_train_for_cot.json"
    rl_path  = raw_dir / "misinfo_RL_train_for_cot.json"

    records = []
    if sft_path.exists():
        sft = load_raw_file(str(sft_path), split_tag="sft")
        records.extend(sft)
        print(f"SFT: {len(sft)} samples  "
              f"(true={sum(r['label']==1 for r in sft)}, "
              f"false={sum(r['label']==0 for r in sft)})")

    if rl_path.exists():
        rl = load_raw_file(str(rl_path), split_tag="rl")
        records.extend(rl)
        print(f"RL:  {len(rl)} samples  "
              f"(true={sum(r['label']==1 for r in rl)}, "
              f"false={sum(r['label']==0 for r in rl)})")

    n_true  = sum(r["label"] == 1 for r in records)
    n_false = sum(r["label"] == 0 for r in records)
    print(f"\nTotal: {len(records)} samples  "
          f"(true={n_true}, false={n_false}, "
          f"balance={n_true/len(records):.1%} true)")
    return records


def records_to_texts_labels(records: list[dict]) -> tuple[list[str], list[int]]:
    """Unpack records into parallel (texts, labels) lists."""
    texts  = [r["text"]  for r in records]
    labels = [r["label"] for r in records]
    return texts, labels

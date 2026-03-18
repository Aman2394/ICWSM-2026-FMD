"""
Train / Validation / Test split for RFC-BENCH.

CRITICAL: Split is performed on the 2,000 ORIGINAL samples ONLY, before
augmentation. Augmented samples are then assigned to the train split only,
preventing any leakage of augmented versions into val or test.

Default split: 70% train / 15% val / 15% test  (stratified by label)

Split is saved to data/splits.json so it is fixed and reproducible across
all notebooks and reruns.

Usage:
    from utils.data_splitter import make_splits, load_splits, filter_by_split
"""
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

SPLITS_FILE = "data/splits.json"


def make_splits(
    records: list[dict],
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    project_dir: str = ".",
) -> dict[str, list]:
    """
    Stratified train/val/test split on original records.

    Args:
        records: Original 2,000 records from load_combined_data().
                 Must NOT include augmented samples.
        val_size:  Fraction for validation.
        test_size: Fraction for test (held out, final eval only).

    Returns dict with keys 'train', 'val', 'test' → lists of record IDs.
    Saves split to data/splits.json for reproducibility.
    """
    ids    = [r["id"]    for r in records]
    labels = [r["label"] for r in records]

    # First carve out test set
    ids_trainval, ids_test, y_trainval, _ = train_test_split(
        ids, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )

    # Then split remaining into train / val
    val_fraction = val_size / (1 - test_size)
    ids_train, ids_val, _, _ = train_test_split(
        ids_trainval, y_trainval,
        test_size=val_fraction,
        stratify=y_trainval,
        random_state=random_state,
    )

    splits = {"train": ids_train, "val": ids_val, "test": ids_test}

    # Print stats
    label_map = {r["id"]: r["label"] for r in records}
    for name, split_ids in splits.items():
        n_true  = sum(label_map[i] == 1 for i in split_ids)
        n_false = sum(label_map[i] == 0 for i in split_ids)
        print(f"{name:6s}: {len(split_ids):5d} samples  "
              f"(true={n_true}, false={n_false})")

    # Save for reproducibility
    splits_path = Path(project_dir) / SPLITS_FILE
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    print(f"\n✅ Splits saved to {splits_path}")
    return splits


def load_splits(project_dir: str = ".") -> dict[str, list]:
    """Load previously saved splits from data/splits.json."""
    splits_path = Path(project_dir) / SPLITS_FILE
    if not splits_path.exists():
        raise FileNotFoundError(
            f"Splits file not found: {splits_path}\n"
            "Run make_splits() first (notebook 00 or 01 Cell 2)."
        )
    with open(splits_path, encoding="utf-8") as f:
        splits = json.load(f)
    print(f"Loaded splits — train: {len(splits['train'])}, "
          f"val: {len(splits['val'])}, test: {len(splits['test'])}")
    return splits


def filter_by_split(records: list[dict], split_ids: list) -> list[dict]:
    """Return only the records whose id is in split_ids."""
    id_set = set(split_ids)
    return [r for r in records if r["id"] in id_set]


def get_split_records(
    all_records: list[dict],
    augmented_path: str | None,
    project_dir: str = ".",
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Load splits and return (train_records, val_records, test_records).

    Train set includes augmented samples (filtered to train IDs only).
    Val and test sets contain ONLY original samples — no augmented data.

    Args:
        all_records:    Original 2,000 records from load_combined_data().
        augmented_path: Path to augmented_train.json, or None.
    """
    splits = load_splits(project_dir)

    val_records  = filter_by_split(all_records, splits["val"])
    test_records = filter_by_split(all_records, splits["test"])

    # Train: original train records + augmented versions of train True samples
    train_orig = filter_by_split(all_records, splits["train"])

    if augmented_path and Path(augmented_path).exists():
        with open(augmented_path, encoding="utf-8") as f:
            aug_data = json.load(f)

        # Augmented source_ids may be raw integers (e.g. 42) if augmentation ran
        # before IDs were updated to "sft_42"/"rl_42" format. Since indices 0-999
        # exist in both SFT and RL files, we cannot safely map an integer source_id
        # to a specific split. Conservative safe rule:
        #   Include augmented sample only if its integer source_id is NOT in
        #   val_numeric or test_numeric — guarantees zero leakage at the cost
        #   of excluding a small number of ambiguous train samples (~5-10%).
        def _numeric(sid) -> int | None:
            try:
                return int(sid) if not isinstance(sid, str) else int(sid.split("_")[1])
            except (TypeError, ValueError, IndexError):
                return None

        val_numeric  = {_numeric(s) for s in splits["val"]  if _numeric(s) is not None}
        test_numeric = {_numeric(s) for s in splits["test"] if _numeric(s) is not None}
        safe_exclude = val_numeric | test_numeric   # any index touching val or test

        aug_train = [
            r for r in aug_data
            if r.get("perturbation_type") is not None         # exclude originals
            and _numeric(r.get("source_id")) not in safe_exclude  # safe train only
        ]
        train_records = train_orig + aug_train
        print(f"\ntrain : {len(train_orig)} original + {len(aug_train)} augmented "
              f"= {len(train_records)} total")
    else:
        train_records = train_orig
        print(f"\ntrain : {len(train_orig)} original (no augmented data found)")

    n_true  = sum(r["label"] == 1 for r in train_records)
    n_false = sum(r["label"] == 0 for r in train_records)
    print(f"         (true={n_true}, false={n_false})")
    print(f"val   : {len(val_records)} original only")
    print(f"test  : {len(test_records)} original only")

    return train_records, val_records, test_records

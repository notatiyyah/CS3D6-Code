"""
Train/Test Split with Multi-Label Stratification and Group-Leakage Prevention
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# Relative imports
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.utils import make_binary_label_matrix


# --- CONSTANTS ---
FULL_DATASET_PATH = "data/output/gold_standard.json"
TAXONOMY_PATH = "data/output/taxonomy_autogen_v3.csv"

TRAIN_OUTPUT_PATH = "data/output/train_data.json"
VAL_OUTPUT_PATH = "data/output/val_data.json"
TEST_OUTPUT_PATH = "data/output/test_data.json"

TEST_SIZE = 0.15
VALIDATION_SIZE = 0.176  # (17.6% of 85% = ~15% total)
RANDOM_STATE = 42


# 1. Load Annotated Records & Taxonomy
print(f"Loading data from {FULL_DATASET_PATH} and {TAXONOMY_PATH}.")

with open(FULL_DATASET_PATH, "r", encoding="utf-8") as f:
    records = json.load(f)

taxonomy = pd.read_csv(TAXONOMY_PATH)


# 2. Build Label Matrix
print("Building multi-label matrix on ANs categories.")

df_labels = make_binary_label_matrix(records, taxonomy)
labels = df_labels.to_numpy()


# 3. Group Record Indices by Unique Text
print("Grouping records by text content to prevent data leakage...")

text_to_indices = defaultdict(list)

for idx, record in enumerate(records):
    text_content = record.get("text", "").strip()

    if text_content:
        text_to_indices[text_content].append(idx)


# Create unique text references and aggregate labels
unique_texts = list(text_to_indices.keys())

grouped_labels = []

for text in unique_texts:
    member_indices = text_to_indices[text]

    # Merge duplicate text labels
    aggregated_label = labels[member_indices].max(axis=0)

    grouped_labels.append(aggregated_label)


unique_texts = np.array(unique_texts)
grouped_labels = np.array(grouped_labels)


# 4. Add pseudo-label for zero-label examples
#
# MultilabelStratifiedShuffleSplit struggles with rows like:
# [0,0,0,0,...]
#
# This column is ONLY used for splitting.
print("Adding pseudo-label for zero-label stratification...")

no_label_rows = grouped_labels.sum(axis=1) == 0

stratify_labels = np.column_stack(
    [
        grouped_labels,
        no_label_rows.astype(int)
    ]
)

print(
    f"Zero-label groups preserved: {no_label_rows.sum()}"
)


# 5. First Split: Separate Test Groups
print("Applying group split 1 (Train+Val | Test).")

first_split = MultilabelStratifiedShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

temp_train_grp_idx, test_grp_idx = next(
    first_split.split(
        np.zeros(len(unique_texts)),
        stratify_labels,
    )
)


X_temp_train_grp = unique_texts[temp_train_grp_idx]
y_temp_train_grp = stratify_labels[temp_train_grp_idx]


# 6. Second Split: Separate Validation Groups
print("Applying group split 2 (Train | Val).")

second_split = MultilabelStratifiedShuffleSplit(
    n_splits=1,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
)


train_grp_idx_relative, val_grp_idx_relative = next(
    second_split.split(
        np.zeros(len(X_temp_train_grp)),
        y_temp_train_grp,
    )
)


# Map relative indices back
train_grp_idx = temp_train_grp_idx[train_grp_idx_relative]
val_grp_idx = temp_train_grp_idx[val_grp_idx_relative]


# 7. Explode groups back into records
def collect_records_from_groups(group_indices):
    records_out = []

    for grp_idx in group_indices:
        target_text = unique_texts[grp_idx]

        for row_idx in text_to_indices[target_text]:
            records_out.append(records[row_idx])

    return records_out


train_records = collect_records_from_groups(train_grp_idx)
val_records = collect_records_from_groups(val_grp_idx)
test_records = collect_records_from_groups(test_grp_idx)


# 8. Export JSON
print(
    f"Exporting split datasets to "
    f"{TRAIN_OUTPUT_PATH}, {VAL_OUTPUT_PATH}, {TEST_OUTPUT_PATH}."
)

for path, dataset in [
    (TRAIN_OUTPUT_PATH, train_records),
    (VAL_OUTPUT_PATH, val_records),
    (TEST_OUTPUT_PATH, test_records),
]:

    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)


# 9. Sanity checks
train_ids = {r["id"] for r in train_records}
val_ids = {r["id"] for r in val_records}
test_ids = {r["id"] for r in test_records}


train_texts = {r["text"].strip() for r in train_records}
val_texts = {r["text"].strip() for r in val_records}
test_texts = {r["text"].strip() for r in test_records}


print("\n--- Leakage Verification Report ---")

print(f"Train samples: n={len(train_records)}")
print(f"Val samples:   n={len(val_records)}")
print(f"Test samples:  n={len(test_records)}")


print(
    "ID Overlaps (Should be 0) -> "
    f"Train∩Val: {len(train_ids & val_ids)} | "
    f"Train∩Test: {len(train_ids & test_ids)} | "
    f"Val∩Test: {len(val_ids & test_ids)}"
)


print(
    "TEXT Leakage (Must be 0) -> "
    f"Train∩Val: {len(train_texts & val_texts)} | "
    f"Train∩Test: {len(train_texts & test_texts)} | "
    f"Val∩Test: {len(val_texts & test_texts)}"
)


# Check zero-label distribution
def count_zero_labels(dataset):
    return sum(
        1 for r in dataset
        if len(r.get("needs", [])) == 0
    )


print("\n--- Zero-label Distribution ---")
print(f"Train: {count_zero_labels(train_records)}")
print(f"Val:   {count_zero_labels(val_records)}")
print(f"Test:  {count_zero_labels(test_records)}")

import matplotlib.pyplot as plt
import pandas as pd


print("\n--- Label Distribution Report ---")


def get_label_counts(dataset, taxonomy):
    counts = {}

    for label in taxonomy["cat_label"]:
        counts[label] = 0

    for record in dataset:
        record_labels = {
            need["label"]
            for need in record.get("needs", [])
            if "label" in need
        }

        for label in record_labels:
            if label in counts:
                counts[label] += 1

    return counts


train_counts = get_label_counts(train_records, taxonomy)
val_counts = get_label_counts(val_records, taxonomy)
test_counts = get_label_counts(test_records, taxonomy)

distribution_df = pd.DataFrame(
    {
        "train": train_counts,
        "val": val_counts,
        "test": test_counts,
    }
)

distribution_df["total"] = (
    distribution_df["train"]
    + distribution_df["val"]
    + distribution_df["test"]
)

distribution_df["train_pct"] = (
    distribution_df["train"]
    / distribution_df["total"]
    * 100
).round(1)

distribution_df["val_pct"] = (
    distribution_df["val"]
    / distribution_df["total"]
    * 100
).round(1)

distribution_df["test_pct"] = (
    distribution_df["test"]
    / distribution_df["total"]
    * 100
).round(1)

distribution_df = distribution_df.sort_values(
    "total",
    ascending=False,
)

pd.set_option("display.max_rows", None)

print(
    distribution_df[
        [
            "train",
            "val",
            "test",
            "train_pct",
            "val_pct",
            "test_pct",
        ]
    ]
)

print("\n--- Missing Labels ---")

for split_name in ["train", "val", "test"]:
    missing = distribution_df[
        distribution_df[split_name] == 0
    ].index.tolist()

    print(
        f"{split_name}: "
        f"{len(missing)} missing labels"
    )

    if missing:
        print(missing)


# Plot
plt.figure(figsize=(16, 8))

distribution_df[
    ["train", "val", "test"]
].plot(
    kind="bar",
    figsize=(18, 8),
)

plt.title("Label Distribution Across Splits")
plt.ylabel("Count")
plt.xlabel("Need ID")
plt.tight_layout()
plt.show()

print("\n--- Stratification Error ---")

distribution_df["train_error"] = (
    distribution_df["train_pct"] - 75
).abs()

distribution_df["val_error"] = (
    distribution_df["val_pct"] - 15
).abs()

distribution_df["test_error"] = (
    distribution_df["test_pct"] - 10
).abs()

print(
    distribution_df[
        [
            "train_pct",
            "val_pct",
            "test_pct",
            "train_error",
            "val_error",
            "test_error",
        ]
    ].sort_values(
        "test_error",
        ascending=False,
    )
)
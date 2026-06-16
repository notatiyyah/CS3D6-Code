"""
Train/Test Split with Multi-Label Stratification
"""
import json
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# Relative imports
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import make_binary_label_matrix

# --- CONSTANTS --- 
FULL_DATASET_PATH = "data/output/full_dataset.json"
TAXONOMY_PATH = "data/output/taxonomy_autogen_v3.csv"

TRAIN_OUTPUT_PATH = "data/output/train_data.json"
VAL_OUTPUT_PATH = "data/output/val_data.json"
TEST_OUTPUT_PATH = "data/output/test_data.json"

TEST_SIZE = 0.15
VALIDATION_SIZE = 0.17 # (17.6% of 85% = ~15% total)
RANDOM_STATE = 42

# 1. Load Annotated Records & Taxonomy
print(f"Loading data from {FULL_DATASET_PATH} and {TAXONOMY_PATH}.")
with open(FULL_DATASET_PATH, "r", encoding="utf-8") as f:
    records = json.load(f)
taxonomy = pd.read_csv(TAXONOMY_PATH)

# 2. Build Label Matrix and Data Arrays
print("Building multi-label matrix on ANs categories.")
df_labels = make_binary_label_matrix(records, taxonomy)
labels = df_labels.to_numpy()

texts = np.array([r.get("text", "") for r in records])

# 3. First Split: Separate Test Set (15%)
print("Applying split 1 (Train+Val | Test).")
first_split = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE) # type: ignore
temp_train_idx, test_idx = next(first_split.split(texts, labels))

# Subset the temporary training data
X_temp_train = texts[temp_train_idx]
y_temp_train = labels[temp_train_idx]

# 4. Second Split: Split Validation (15% of total) from the Temp Train (85%)
print("Applying split 2 (Train | Val).")
second_split = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE) # type: ignore
train_idx_relative, val_idx_relative = next(second_split.split(X_temp_train, y_temp_train))

# Map the relative indices back to the original record indices
train_idx = temp_train_idx[train_idx_relative]
val_idx = temp_train_idx[val_idx_relative]

# Extract the actual records
train_records = [records[i] for i in train_idx]
val_records = [records[i] for i in val_idx]
test_records = [records[i] for i in test_idx]

# 5. Export to JSON
print(f"Exporting split datasets to {TRAIN_OUTPUT_PATH}, {VAL_OUTPUT_PATH}, {TEST_OUTPUT_PATH}.")
with open(TRAIN_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(train_records, f, indent=2)

with open(VAL_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(val_records, f, indent=2)

with open(TEST_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(test_records, f, indent=2)

# 6. Sanity check
train_ids = {r["id"] for r in train_records}
val_ids = {r["id"] for r in val_records}
test_ids = {r["id"] for r in test_records}

print(f"Train: n={len(train_records)}")
print(f"Val:   n={len(val_records)}")
print(f"Test:  n={len(test_records)}")
print(f"Overlap check - Train∩Val: {len(train_ids & val_ids)} (should be 0)")
print(f"Overlap check - Train∩Test: {len(train_ids & test_ids)} (should be 0)")
print(f"Overlap check - Val∩Test:   {len(val_ids & test_ids)} (should be 0)")
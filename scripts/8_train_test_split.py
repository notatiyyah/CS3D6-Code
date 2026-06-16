"""
Train/Test Split with Multi-Label Stratification
Takes annotated records and splits them into stratified train and test
sets to make sure that severe label imbalances are proportionally represented.
"""

import json
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from utils.utils import make_binary_label_matrix

# --- CONSTANTS ---
FULL_DATASET_PATH = "../data/output/full_dataset.json"
TAXONOMY_PATH = "../data/output/taxonomy_autogen_v3.csv"

TRAIN_OUTPUT_PATH = "../data/output/full_train_data.json"
TEST_OUTPUT_PATH = "../data/output/full_test_data.json"

TEST_SIZE = "0.2"
RANDOM_STATE = 42


# 1. Load Annotated Records & Taxonomy
with open(FULL_DATASET_PATH, "r", encoding="utf-8") as f:
    records = json.load(f)

taxonomy = pd.read_csv(TAXONOMY_PATH)

# 2. Build Label Matrix and Data Arrays
df_labels = make_binary_label_matrix(records, taxonomy)
labels = df_labels.to_numpy()

texts = np.array([r.get("text", "") for r in records])

# 3. Apply Multi-Label Stratified Split
msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
train_idx, val_idx = next(msss.split(texts, labels))

# 4. Slice the Data (msss.split returns a list of indexes)
train_records = [records[i] for i in train_idx]
test_records = [records[i] for i in val_idx]

# 5. Export to JSON
with open(TRAIN_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(train_records, f, indent=2)

with open(TEST_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(test_records, f, indent=2)
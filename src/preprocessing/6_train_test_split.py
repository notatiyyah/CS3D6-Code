"""
Train/Test Split with Multi-Label Stratification and Group-Leakage Prevention.
"""

import json
from collections import defaultdict

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from common.logging import setup_logger
from common.paths import PROCESSED
from utils.utils import make_binary_label_matrix

class Config:
    LOGGER = setup_logger("preprocessing.split_dataset", "split_dataset.log")

    INPUT_DATA_PATH = PROCESSED / "gold_standard.json"
    TAXONOMY_PATH = PROCESSED / "taxonomy_autogen_v3.csv"
    TRAIN_OUTPUT = PROCESSED / "train_data.json"
    VAL_OUTPUT = PROCESSED / "val_data.json"
    TEST_OUTPUT = PROCESSED / "test_data.json"

    TEST_SIZE = 0.15
    VALIDATION_SIZE = 0.176  # (17.6% of 85% = ~15% total)
    RANDOM_STATE = 42


def group_by_text(records, labels):
    """Group duplicate texts so the same note can't land in two splits."""
    text_to_indices = defaultdict(list)
    for idx, record in enumerate(records):
        text = record.get("text", "").strip()
        if text:
            text_to_indices[text].append(idx)

    texts = list(text_to_indices)
    grouped_labels = np.array([labels[text_to_indices[t]].max(axis=0) for t in texts])
    return texts, grouped_labels, text_to_indices


def stratified_split(texts, labels):
    """Two-stage multilabel stratified split into train/val/test, with an
    extra column so all-zero-label rows still stratify correctly."""
    no_labels = (labels.sum(axis=1) == 0).astype(int)
    stratify_labels = np.column_stack([labels, no_labels])

    test_splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
    )
    train_val_idx, test_idx = next(test_splitter.split(texts, stratify_labels))

    val_splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=Config.VALIDATION_SIZE, random_state=Config.RANDOM_STATE
    )
    train_idx, val_idx = next(
        val_splitter.split(train_val_idx, stratify_labels[train_val_idx])
    )

    return train_val_idx[train_idx], train_val_idx[val_idx], test_idx


def expand(indices, texts, lookup, records):
    """Turn group indices back into the original records."""
    return [records[i] for idx in indices for i in lookup[texts[idx]]]


def save(path, data):
    Config.LOGGER.info("Saving %s records -> %s", len(data), path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def main():
    Config.LOGGER.info("Loading data from %s and taxonomy from %s.", Config.INPUT_DATA_PATH, Config.TAXONOMY_PATH)
    with open(Config.INPUT_DATA_PATH, encoding="utf-8") as f:
        records = json.load(f)
    taxonomy = pd.read_csv(Config.TAXONOMY_PATH)

    # Stratified split with duplicate records handling
    labels = make_binary_label_matrix(records, taxonomy).to_numpy()
    texts, grouped_labels, lookup = group_by_text(records, labels)
    train_idx, val_idx, test_idx = stratified_split(texts, grouped_labels)

    train = expand(train_idx, texts, lookup, records)
    val = expand(val_idx, texts, lookup, records)
    test = expand(test_idx, texts, lookup, records)

    # Save out
    save(Config.TRAIN_OUTPUT, train)
    save(Config.VAL_OUTPUT, val)
    save(Config.TEST_OUTPUT, test)


    # Check for leakage
    train_texts = {x["text"] for x in train}
    val_texts = {x["text"] for x in val}
    test_texts = {x["text"] for x in test}
    Config.LOGGER.info(
        "Leakage train/val=%s train/test=%s val/test=%s",
        len(train_texts & val_texts),
        len(train_texts & test_texts),
        len(val_texts & test_texts),
    )

    Config.LOGGER.info("Split complete: train=%s val=%s test=%s", len(train), len(val), len(test))


if __name__ == "__main__":
    main()
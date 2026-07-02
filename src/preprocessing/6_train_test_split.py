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
from common.json_helpers import load_json, save_json
from common.graph_helpers import make_binary_label_matrix

class Config:
    logger = setup_logger("preprocessing.split_dataset", "split_dataset.log")

    input_data_path = PROCESSED / "gold_standard.json"
    taxonomy_path = PROCESSED / "taxonomy_autogen_v3.csv"
    train_output = PROCESSED / "train_data.json"
    val_output = PROCESSED / "val_data.json"
    test_output = PROCESSED / "test_data.json"

    test_size = 0.15
    validation_size = 0.176  # (17.6% of 85% = ~15% total)
    random_state = 42


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


def stratified_split(texts, labels, config):
    """Two-stage multilabel stratified split into train/val/test, with an
    extra column so all-zero-label rows still stratify correctly."""
    no_labels = (labels.sum(axis=1) == 0).astype(int)
    stratify_labels = np.column_stack([labels, no_labels])

    test_splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=config.test_size, random_state=config.random_state
    )
    train_val_idx, test_idx = next(test_splitter.split(texts, stratify_labels))

    val_splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=config.validation_size, random_state=config.random_state
    )
    train_idx, val_idx = next(
        val_splitter.split(train_val_idx, stratify_labels[train_val_idx])
    )

    return train_val_idx[train_idx], train_val_idx[val_idx], test_idx


def expand(indices, texts, lookup, records):
    """Turn group indices back into the original records."""
    return [records[i] for idx in indices for i in lookup[texts[idx]]]


def main():
    config = Config()
    config.logger.info("Starting train/test/validation split (70/30/30)...")

    # Load data
    records = load_json(config.input_data_path, config.logger)
    config.logger.info("Loading taxonomy from %s.", config.taxonomy_path)
    taxonomy = pd.read_csv(config.taxonomy_path)

    # Stratified split with duplicate records handling
    labels = make_binary_label_matrix(records, taxonomy).to_numpy()
    texts, grouped_labels, lookup = group_by_text(records, labels)
    train_idx, val_idx, test_idx = stratified_split(texts, grouped_labels, config)

    train = expand(train_idx, texts, lookup, records)
    val = expand(val_idx, texts, lookup, records)
    test = expand(test_idx, texts, lookup, records)

    # Save out
    save_json(config.train_output, train, config.logger)
    save_json(config.val_output, val, config.logger)
    save_json(config.test_output, test, config.logger)

    # Check for leakage
    train_texts = {x["text"] for x in train}
    val_texts = {x["text"] for x in val}
    test_texts = {x["text"] for x in test}
    config.logger.info(
        "Leakage train/val=%s train/test=%s val/test=%s",
        len(train_texts & val_texts),
        len(train_texts & test_texts),
        len(val_texts & test_texts),
    )

    config.logger.info("Split complete: train=%s val=%s test=%s", len(train), len(val), len(test))


if __name__ == "__main__":
    main()
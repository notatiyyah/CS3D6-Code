import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Prevents silently failing without error logs

import random
from dataclasses import dataclass, asdict, field
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments,
    Trainer, EvalPrediction, TrainerCallback,
)

from common.paths import MODELS, PROCESSED
from common.logging import setup_logger, FileLogCallback
from common.json_helpers import load_json, save_json


# --- Config ---
@dataclass
class TrainingConfig:
    model_name: str       = "roberta-base"
    threshold: float      = 0.5
    learning_rate: float  = 2e-5
    train_batch_size: int = 16
    eval_batch_size: int  = 32
    epochs: int           = 10
    max_length: int       = 128
    seed: int             = 42


@dataclass
class Config:
    # Dynamically create run name from datetime
    run_name: str            = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    taxonomy_path            = PROCESSED / "taxonomy_autogen_v3.csv"
    train_path               = PROCESSED / "train_data_doc_level.json"
    val_path                 = PROCESSED / "val_data_doc_level.json"

    training: TrainingConfig = field(default_factory=TrainingConfig)
    device                   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        # New dir  & logger per run
        self.run_dir = MODELS / "hierarchical" / self.run_name
        self.parent_dir = self.run_dir / "parent-classifier"
        self.parent_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(
            f"training.hierarchical.{self.run_name}",
            f"train_hierarchical_{self.run_name}.log"
        )

    def child_dir(self, parent_name: str):
        safe_name = parent_name.replace(" ", "_").replace("&", "and")
        path = self.run_dir / f"child-{safe_name}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_training_params(self, model_dir):
        save_json(path=model_dir / "config.json", data=asdict(self.training), logger=self.logger)


class MultiLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts, self.labels, self.tokenizer, self.max_length = texts, labels, tokenizer, max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
        }


def compute_sample_weights(labels):
    labels = np.array(labels)
    label_freqs = np.where(labels.sum(axis=0) == 0, 1, labels.sum(axis=0))
    return [
        max((1.0 / label_freqs[i] for i in np.where(row == 1)[0]), default=1.0 / label_freqs.max())
        for row in labels
    ]


class WeightedTrainer(Trainer):
    def __init__(self, *args, sample_weights=None, seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_weights = sample_weights
        self.seed = seed

    def get_train_dataloader(self):
        if self.sample_weights is None:
            return super().get_train_dataloader()
        gen = torch.Generator().manual_seed(self.seed)
        sampler = WeightedRandomSampler(
            self.sample_weights, num_samples=len(self.sample_weights), replacement=True, generator=gen
        )
        return DataLoader(
            self.train_dataset, batch_size=self._train_batch_size, sampler=sampler,
            collate_fn=self.data_collator
        )


def optimize_thresholds(y_true, y_prob, label_names):
    y_true, y_prob = np.array(y_true), np.array(y_prob)
    results = {}
    for i, name in enumerate(label_names):
        c_true, c_prob = y_true[:, i], y_prob[:, i]
        best_f1, best_t = max(
            ((f1_score(c_true, (c_prob > t).astype(int), zero_division=0), t)
             for t in np.arange(0.05, 0.95, 0.05)),
            key=lambda x: x[0]
        )
        results[name] = {"threshold": float(best_t), "optimized_f1": float(best_f1)}
    return results


def build_compute_metrics_fn(label_names):
    def compute_metrics(eval_pred: EvalPrediction):
        preds = (1 / (1 + np.exp(-eval_pred.predictions)) > 0.5).astype(int)
        return {
            "f1_macro": f1_score(eval_pred.label_ids, preds, average="macro", zero_division=0),
            "f1_micro": f1_score(eval_pred.label_ids, preds, average="micro", zero_division=0),
        }
    return compute_metrics


def predict_probs(model, tokenizer, texts, device, max_length, batch_size=32):
    model.eval()
    with torch.no_grad():
        return np.vstack([
            1 / (1 + np.exp(-model(**tokenizer(
                texts[i:i + batch_size], truncation=True, padding="max_length",
                max_length=max_length, return_tensors="pt"
            ).to(device)).logits.cpu().numpy()))
            for i in range(0, len(texts), batch_size)
        ])


def map_to_parents(dataset, parent_map, parent2id):
    transformed = []
    for item in dataset:
        vec = [0] * len(parent2id)
        for name in item["label_names"]:
            if name in parent_map:
                vec[parent2id[parent_map[name]]] = 1
        transformed.append({"text": item["text"], "labels": vec})
    return transformed


def make_training_args(output_dir, config: Config):
    return TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.training.learning_rate,
        per_device_train_batch_size=config.training.train_batch_size,
        per_device_eval_batch_size=config.training.eval_batch_size,
        num_train_epochs=config.training.epochs,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=30,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        seed=config.training.seed,
    )


def train_parent_classifier(config: Config, train_data, val_data, parent_map, parent2id, unique_parents, tokenizer):
    config.logger.info("--- Training Parent Classifier (run=%s) ---", config.run_name)

    parent_train = map_to_parents(train_data, parent_map, parent2id)
    parent_val = map_to_parents(val_data, parent_map, parent2id)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.training.model_name,
        num_labels=len(unique_parents),
        problem_type="multi_label_classification",
        id2label={i: p for p, i in parent2id.items()},
        label2id=parent2id,
    ).to(config.device)

    trainer = WeightedTrainer(
        model=model,
        args=make_training_args(config.parent_dir, config),
        train_dataset=MultiLabelDataset(
            [x["text"] for x in parent_train], [x["labels"] for x in parent_train],
            tokenizer, config.training.max_length
        ),
        eval_dataset=MultiLabelDataset(
            [x["text"] for x in parent_val], [x["labels"] for x in parent_val],
            tokenizer, config.training.max_length
        ),
        compute_metrics=build_compute_metrics_fn(unique_parents),
        sample_weights=compute_sample_weights([x["labels"] for x in parent_train]),
        seed=config.training.seed,
        callbacks=[FileLogCallback(config.logger, prefix="[parent] ")],
    )
    trainer.train()

    # Test different thresholds
    val_texts = [x["text"] for x in parent_val]
    val_labels = [x["labels"] for x in parent_val]
    probs = predict_probs(model, tokenizer, val_texts, config.device, config.training.max_length)
    thresholds = optimize_thresholds(val_labels, probs, unique_parents)

    # Save
    final_dir = config.parent_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    save_json(path=final_dir / "best_thresholds.json", data=thresholds, logger=config.logger)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    config.save_training_params(config.parent_dir)

    config.logger.info("Parent classifier saved to %s", final_dir)
    return parent_train, parent_val


def train_child_classifiers(config: Config, train_data, val_data, parent_train, parent_val, parent_map, parent2id, tokenizer):
    parent_to_children = defaultdict(list)
    for child, parent in parent_map.items():
        parent_to_children[parent].append(child)

    for parent_name, children in parent_to_children.items():
        children = sorted(children)
        if len(children) < 2:
            config.logger.info("Skipping child classifier for '%s' (fewer than 2 children)", parent_name)
            continue

        sub_train = [x for x, p in zip(train_data, parent_train) if p["labels"][parent2id[parent_name]] == 1]
        sub_val = [x for x, p in zip(val_data, parent_val) if p["labels"][parent2id[parent_name]] == 1]
        if not sub_train:
            config.logger.info("Skipping child classifier for '%s' (no training examples)", parent_name)
            continue

        config.logger.info("--- Training Child Classifier: %s ---", parent_name)

        child2id = {c: i for i, c in enumerate(children)}
        t_labels = [[1 if n in item["label_names"] else 0 for n in children] for item in sub_train]
        v_labels = [[1 if n in item["label_names"] else 0 for n in children] for item in sub_val]

        c_out = config.child_dir(parent_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            config.training.model_name,
            num_labels=len(children),
            problem_type="multi_label_classification",
            id2label={i: c for c, i in child2id.items()},
            label2id=child2id,
        ).to(config.device)

        trainer = WeightedTrainer(
            model=model,
            args=make_training_args(c_out, config),
            train_dataset=MultiLabelDataset(
                [x["text"] for x in sub_train], t_labels, tokenizer, config.training.max_length
            ),
            eval_dataset=MultiLabelDataset(
                [x["text"] for x in sub_val], v_labels, tokenizer, config.training.max_length
            ),
            compute_metrics=build_compute_metrics_fn(children),
            sample_weights=compute_sample_weights(t_labels),
            seed=config.training.seed,
            callbacks=[FileLogCallback(config.logger, prefix=f"[child:{parent_name}] ")],
        )
        trainer.train()

        # Test different thresholds
        c_texts = [x["text"] for x in sub_val]
        c_probs = predict_probs(model, tokenizer, c_texts, config.device, config.training.max_length)
        thresholds = optimize_thresholds(v_labels, c_probs, children)

        # Save
        final_dir = c_out / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        save_json(path=final_dir / "best_thresholds.json", data=thresholds, logger=config.logger)
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        config.save_training_params(c_out)

        config.logger.info("Child classifier '%s' saved to %s", parent_name, final_dir)


def main():
    config = Config()
    random.seed(config.training.seed)
    np.random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)

    config.logger.info("Initializing Hierarchical Training Pipeline on %s (run=%s)", config.device, config.run_name)

    # Load Data
    train_data = load_json(config.train_path, config.logger)
    val_data = load_json(config.val_path, config.logger)

    # Map children to parents
    config.logger.info("Importing taxonomy from %s", config.taxonomy_path)
    taxonomy_df = pd.read_csv(config.taxonomy_path)
    parent_map = pd.Series(taxonomy_df.high_level_category.values, index=taxonomy_df.cat_label).to_dict()
    unique_parents = sorted(set(parent_map.values()))
    parent2id = {p: i for i, p in enumerate(unique_parents)}

    # Set up and train
    tokenizer = AutoTokenizer.from_pretrained(config.training.model_name)

    parent_train, parent_val = train_parent_classifier(
        config, train_data, val_data, parent_map, parent2id, unique_parents, tokenizer
    )
    train_child_classifiers(
        config, train_data, val_data, parent_train, parent_val, parent_map, parent2id, tokenizer
    )

    config.logger.info("Pipeline execution sequence complete (run=%s)", config.run_name)


if __name__ == "__main__":
    main()
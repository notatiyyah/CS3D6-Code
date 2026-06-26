import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Prevents silently failing without error logs

from dataclasses import dataclass, asdict, field
from datetime import datetime
import json
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                          TrainingArguments, Trainer, EvalPrediction)

from common.paths import MODELS, LOGS, PROCESSED
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
    weight_decay: float   = 0.01
    max_length: int       = 128
    seed: int             = 42
    fp16_enabled: bool    = torch.cuda.is_available()

@dataclass
class Config:
    # Dynamically create run name from datetime
    run_name: str            = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    label_mapping_path       = PROCESSED / "label_mapping.json"
    train_path               = PROCESSED / "train_data_doc_level.json"
    val_path                 = PROCESSED / "val_data_doc_level.json"
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device                   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger                   = setup_logger("training.doc_level.train_flat", "train_flat.log")

    def __post_init__(self):
        # New dir  & logger per run
        self.model_dir = MODELS / "doc-classifier-flat" / self.run_name
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(
            f"training.doc_level.train_flat.{self.run_name}",
            f"train_doc_flat_{self.run_name}.log"
        )

    def save_training_params(self):
        save_json(
            path=self.model_dir / "config.json",
            data=asdict(self.training),
            logger=self.logger
        )


class MultiLabelDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["text"], truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(item["labels"], dtype=torch.float),
        }

def compute_metrics(eval_pred: EvalPrediction, config: Config):
    probs = 1 / (1 + np.exp(-eval_pred.predictions))
    preds = (probs > config.training.threshold).astype(int)
    labels = eval_pred.label_ids
    return {
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "precision_micro": precision_score(labels, preds, average="micro", zero_division=0),
        "recall_micro": recall_score(labels, preds, average="micro", zero_division=0),
    }


def main():
    config = Config()
    config.logger.info("Beginning training script - Flat Document Classifier (run=%s)", config.run_name)

    # Load mappings
    label2id = load_json(config.label_mapping_path, config.logger)
    id2label = {int(v): k for k, v in label2id.items()}

    # Load data
    train_data = load_json(config.train_path, config.logger)
    val_data = load_json(config.val_path, config.logger)

    # Model settings
    config.logger.info("Initializing Flat Doc Classifier on %s...", config.device)

    tokenizer = AutoTokenizer.from_pretrained(config.training.model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.training.model_name,
        num_labels=len(label2id),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id
    )

    args = TrainingArguments(
        output_dir=config.model_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.training.learning_rate,
        per_device_train_batch_size=config.training.train_batch_size,
        per_device_eval_batch_size=config.training.eval_batch_size,
        num_train_epochs=config.training.epochs,
        weight_decay=config.training.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=30,
        save_total_limit=1,  # keep only the best checkpoint on disk
        fp16=config.training.fp16_enabled,
        seed=config.training.seed
    )

    def metrics_wrapper(eval_pred: EvalPrediction):
        return compute_metrics(eval_pred=eval_pred, config=config)

    # Train
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=MultiLabelDataset(train_data, tokenizer),
        eval_dataset=MultiLabelDataset(val_data, tokenizer),
        compute_metrics=metrics_wrapper,
        callbacks=[FileLogCallback(config.logger)],
    )

    config.logger.info("Training Flat Doc Classifier on %s...", config.device)
    trainer.train()

    # Save output
    final_dir = config.model_dir / "final_model"
    config.logger.info("Saving Flat Doc Classifier to %s...", final_dir)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    config.save_training_params()


if __name__ == "__main__":
    main()
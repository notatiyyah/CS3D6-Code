"""
RoBERTa BIO Token Classifier — Needs extraction via sequence labeling.

Standard BIO-tagged token classification baseline. Superseded by the
candidate-based span classifier (span_model.py / train_span_v3.py) for the
main pipeline, but kept and reported as a comparison architecture.
"""

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Prevents silently failing without error logs

import json
from collections import Counter
from dataclasses import dataclass, asdict, field
from datetime import datetime

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForTokenClassification,
)

from common.paths import MODELS, PROCESSED
from common.logging import setup_logger
from common.json_helpers import load_json, save_json


# --- Config ---
@dataclass
class TrainingConfig:
    base_model: str = "roberta-base"
    learning_rate: float = 2e-5
    train_batch_size: int = 8
    eval_batch_size: int = 8
    epochs: int = 4
    weight_decay: float = 0.01
    max_length: int = 512
    logging_steps: int = 10
    o_label_weight: float = 0.1
    weight_cap: float = 5.0
    seed: int = 42


@dataclass
class Config:
    run_name: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    train_path = PROCESSED / "train_data.json"
    val_path = PROCESSED / "val_data.json"

    training: TrainingConfig = field(default_factory=TrainingConfig)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        self.model_dir = MODELS / "needs-bio-classifier" / self.run_name
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(
            f"training.bio_classifier.{self.run_name}",
            f"train_bio_classifier_{self.run_name}.log"
        )

    def save_training_params(self):
        save_json(path=self.model_dir / "config.json", data=asdict(self.training), logger=self.logger)


def load_records(path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [{"text": item["text"], "entities": item.get("needs", [])} for item in json.load(f)]


def build_label_list(train_records, val_records):
    unique_labels = {e["label"] for r in train_records + val_records for e in r["entities"]}
    label_list = ["O"] + [tag for name in sorted(unique_labels) for tag in (f"B-{name}", f"I-{name}")]
    return label_list, {l: i for i, l in enumerate(label_list)}, {i: l for i, l in enumerate(label_list)}


def build_align_labels_fn(tokenizer, label2id, max_length):
    def align_labels(examples):
        tokenized = tokenizer(
            examples["text"], truncation=True, padding="max_length",
            max_length=max_length, return_offsets_mapping=True
        )
        all_labels = []
        for offsets, entities in zip(tokenized["offset_mapping"], examples["entities"]):
            token_labels = [label2id["O"]] * len(offsets)
            for entity in entities:
                e_start, e_end, e_label = entity["start"], entity["end"], entity["label"]
                first_token = True
                for idx, (tok_start, tok_end) in enumerate(offsets):
                    if tok_start == tok_end:
                        token_labels[idx] = -100
                        continue
                    if not (tok_end <= e_start or tok_start >= e_end):
                        bio_tag = f"B-{e_label}" if first_token else f"I-{e_label}"
                        token_labels[idx] = label2id.get(bio_tag, label2id["O"])
                        first_token = False
            all_labels.append(token_labels)
        tokenized["labels"] = all_labels
        tokenized.pop("offset_mapping")
        return tokenized
    return align_labels


def make_weights(dataset, label_list, label2id, o_label_weight, weight_cap, device, logger):
    label_counts = Counter(l for sample in dataset for l in sample["labels"] if l != -100)
    total = sum(label_counts.values())
    weights = torch.ones(len(label_list))
    for label_id, count in label_counts.items():
        if count > 0:
            weights[label_id] = total / (len(label_list) * count)
    weights = weights.clamp(max=weight_cap)
    weights[label2id["O"]] = o_label_weight

    logger.info("Label weights (capped at %s, O fixed at %s): %s", weight_cap, o_label_weight, weights.tolist())
    return weights.to(device)


class WeightedTrainer(Trainer):
    def __init__(self, *args, loss_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_weights = loss_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.CrossEntropyLoss(weight=self.loss_weights)(
            outputs.logits.view(-1, outputs.logits.shape[-1]), labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss


class FileLogCallback(TrainerCallback):
    """Forwards Trainer's per-step/per-epoch log dicts (loss, eval f1, etc.)
    into our own file logger, since HF's progress bar + internal logger
    don't write to it by default."""
    def __init__(self, logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        entries = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in logs.items())
        self.logger.info("step=%s %s", state.global_step, entries)


def build_compute_metrics_fn(label_list):
    """Token-level F1 (not seqeval-style entity-level F1): every non-masked
    (-100) token's predicted BIO tag is compared against its gold tag.
    Reported on non-'O' tokens only, so the dominant background class
    doesn't dilute the metric. NOTE: this is not directly comparable to
    span v3's exact-span loose/strict F1 — token-level credit is more
    forgiving than exact-span matching."""
    o_id = label_list.index("O")

    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=2)
        labels = eval_pred.label_ids

        true_labels, true_preds = [], []
        for pred_row, label_row in zip(preds, labels):
            for p, l in zip(pred_row, label_row):
                if l == -100:
                    continue
                true_labels.append(l)
                true_preds.append(p)

        return {
            "f1_micro": f1_score(true_labels, true_preds, average="micro", zero_division=0, labels=[i for i in range(len(label_list)) if i != o_id]),
            "f1_macro": f1_score(true_labels, true_preds, average="macro", zero_division=0, labels=[i for i in range(len(label_list)) if i != o_id]),
            "precision_micro": precision_score(true_labels, true_preds, average="micro", zero_division=0, labels=[i for i in range(len(label_list)) if i != o_id]),
            "recall_micro": recall_score(true_labels, true_preds, average="micro", zero_division=0, labels=[i for i in range(len(label_list)) if i != o_id]),
        }
    return compute_metrics


def main():
    config = Config()
    torch.manual_seed(config.training.seed)

    config.logger.info("Beginning training script - BIO Token Classifier (run=%s)", config.run_name)
    config.save_training_params()

    train_records = load_records(config.train_path)
    val_records = load_records(config.val_path)
    label_list, label2id, id2label = build_label_list(train_records, val_records)

    config.logger.info("Initializing BIO Classifier on %s...", config.device)

    tokenizer = AutoTokenizer.from_pretrained(config.training.base_model, add_prefix_space=True)
    align_labels = build_align_labels_fn(tokenizer, label2id, config.training.max_length)

    train_ds = Dataset.from_list(train_records).map(align_labels, batched=True, remove_columns=["text", "entities"])
    val_ds = Dataset.from_list(val_records).map(align_labels, batched=True, remove_columns=["text", "entities"])

    model = AutoModelForTokenClassification.from_pretrained(
        config.training.base_model, num_labels=len(label_list), id2label=id2label, label2id=label2id
    )

    args = TrainingArguments(
        output_dir=config.model_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=config.training.learning_rate,
        per_device_train_batch_size=config.training.train_batch_size,
        per_device_eval_batch_size=config.training.eval_batch_size,
        num_train_epochs=config.training.epochs,
        weight_decay=config.training.weight_decay,
        logging_steps=config.training.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        seed=config.training.seed,
    )

    weights = make_weights(
        train_ds, label_list, label2id, config.training.o_label_weight, config.training.weight_cap,
        config.device, config.logger,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=build_compute_metrics_fn(label_list),
        loss_weights=weights,
        callbacks=[FileLogCallback(config.logger)],
    )

    config.logger.info("Training BIO Classifier on %s...", config.device)
    trainer.train()

    best_f1_macro = max((h.get("eval_f1_macro", 0.0) for h in trainer.state.log_history), default=0.0)
    config.logger.info("Best eval f1_macro (non-O tokens): %.4f", best_f1_macro)

    final_dir = config.model_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)

    config.logger.info("Saving BIO Classifier to %s...", final_dir)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    config.save_training_params()
    config.logger.info("Pipeline complete (run=%s). Saved to %s", config.run_name, final_dir)


if __name__ == "__main__":
    main()
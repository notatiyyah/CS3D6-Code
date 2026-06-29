import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Prevents silently failing without error logs

import json
import random
from dataclasses import dataclass, asdict, field
from datetime import datetime

import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, TrainerCallback

from common.paths import MODELS, PROCESSED
from common.logging import setup_logger
from common.json_helpers import load_json, save_json
from common.span_model import SpanClassifier, generate_candidates


# --- Config ---
@dataclass
class TrainingConfig:
    base_model: str = "distilbert-base-uncased"
    learning_rate: float = 2e-5
    max_steps: int = 5000
    train_batch_size: int = 1
    eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    eval_steps: int = 125
    save_steps: int = 125
    logging_steps: int = 50
    max_candidate_size: int = 20
    neg_sample_ratio: int = 5
    neg_sample_floor: int = 50
    pos_weight_cap: float = 5.0
    seed: int = 42
    fp16: bool = torch.cuda.is_available()

@dataclass
class Config:
    run_name: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    train_path = PROCESSED / "train_data.json"
    val_path = PROCESSED / "val_data.json"

    training: TrainingConfig = field(default_factory=TrainingConfig)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        self.model_dir = MODELS / "needs-span-classifier" / self.run_name
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(
            f"training.span.{self.run_name}",
            f"train_span_{self.run_name}.log"
        )

    def save_training_params(self):
        save_json(path=self.model_dir / "config.json", data=asdict(self.training), logger=self.logger)


class SpanDataset(Dataset):
    def __init__(self, records, tokenizer, label2id, max_candidate_size):
        self.examples = []
        for record in records:
            tokenized = tokenizer(record["text"], truncation=True, max_length=512, return_offsets_mapping=True)
            offsets = tokenized["offset_mapping"]
            gold_token_spans = {}
            all_entities = record.get("needs", []) + record.get("persons", [])
            for entity in all_entities:
                if entity["label"] not in label2id:
                    continue
                tok_start, tok_end = None, None
                for idx, (tok_s, tok_e) in enumerate(offsets):
                    if tok_s == tok_e:
                        continue

                    if tok_s < entity["end"] and tok_e > entity["start"]:
                        if tok_start is None:
                            tok_start = idx
                        tok_end = idx + 1
                if tok_start is not None and tok_end is not None and tok_end > tok_start:
                    gold_token_spans.setdefault((tok_start, tok_end), set()).add(label2id[entity["label"]])
            self.examples.append({
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "candidates": generate_candidates(offsets, max_candidate_size),
                "gold_token_spans": gold_token_spans,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def build_collate_fn(label_list, neg_sample_ratio, neg_sample_floor):
    def collate_fn(batch):
        item = batch[0]
        all_candidates, gold_spans = item["candidates"], item["gold_token_spans"]
        pos_idxs = [i for i, c in enumerate(all_candidates) if c in gold_spans]
        neg_idxs = [i for i, c in enumerate(all_candidates) if c not in gold_spans]
        n_keep_neg = max(len(pos_idxs) * neg_sample_ratio, neg_sample_floor)
        if len(neg_idxs) > n_keep_neg:
            neg_idxs = random.sample(neg_idxs, n_keep_neg)
        kept_idxs = sorted(pos_idxs + neg_idxs)

        labels = torch.zeros(len(kept_idxs), len(label_list), dtype=torch.float)
        for new_i, orig_i in enumerate(kept_idxs):
            cand = all_candidates[orig_i]
            if cand in gold_spans:
                for label_id in gold_spans[cand]:
                    labels[new_i, label_id] = 1.0

        return {
            "input_ids": torch.tensor([item["input_ids"]], dtype=torch.long),
            "attention_mask": torch.tensor([item["attention_mask"]], dtype=torch.long),
            "candidate_spans": torch.tensor([all_candidates[i] for i in kept_idxs], dtype=torch.long),
            "labels": labels,
        }
    return collate_fn


def compute_metrics(eval_pred, threshold=0.5):
    """Candidate-level F1 used only for checkpoint selection during training
    (metric_for_best_model), not the real evaluation. Trainer hands us logits
    and labels for whichever candidates collate_fn kept for eval (gold spans +
    sampled negatives), not the full per-document candidate set — so this is
    a binary-classification proxy over sampled candidates, not document-level
    loose/strict span F1. Use eval_span.py for the real numbers.
    """
    probs = 1 / (1 + np.exp(-eval_pred.predictions))
    preds = (probs > threshold).astype(int)
    labels = eval_pred.label_ids
    return {
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "precision_micro": precision_score(labels, preds, average="micro", zero_division=0),
        "recall_micro": recall_score(labels, preds, average="micro", zero_division=0),
    }


class SpanTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]


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


def compute_pos_weight(train_ds, num_labels, cap, device):
    pos_counts = torch.zeros(num_labels)
    total_candidates = 0
    for ex in train_ds.examples:
        total_candidates += len(ex["candidates"])
        for l_ids in ex["gold_token_spans"].values():
            for lid in l_ids:
                pos_counts[lid] += 1
    return ((total_candidates - pos_counts) / pos_counts.clamp(min=1)).clamp(max=cap).to(device)


def main():
    config = Config()
    random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)

    config.logger.info("Beginning training script - Span Classifier (run=%s)", config.run_name)
    config.save_training_params()

    train_records = load_json(config.train_path)
    val_records = load_json(config.val_path)

    label_list = sorted({e["label"] for r in train_records + val_records for e in r.get("needs", []) + r.get("persons", [])})
    label2id = {l: i for i, l in enumerate(label_list)}

    config.logger.info("Initializing Span Classifier on %s...", config.device)

    tokenizer = AutoTokenizer.from_pretrained(config.training.base_model)

    train_ds = SpanDataset(train_records, tokenizer, label2id, config.training.max_candidate_size)
    val_ds = SpanDataset(val_records, tokenizer, label2id, config.training.max_candidate_size)

    pos_weight = compute_pos_weight(train_ds, len(label_list), config.training.pos_weight_cap, config.device)
    model = SpanClassifier(config.training.base_model, len(label_list), pos_weight).to(config.device)

    collate_fn = build_collate_fn(label_list, config.training.neg_sample_ratio, config.training.neg_sample_floor)

    args = TrainingArguments(
        output_dir=config.model_dir,
        eval_strategy="steps",
        eval_steps=config.training.eval_steps,
        save_strategy="steps",
        save_steps=config.training.save_steps,
        max_steps=config.training.max_steps,
        save_total_limit=1,
        learning_rate=config.training.learning_rate,
        per_device_train_batch_size=config.training.train_batch_size,
        per_device_eval_batch_size=config.training.eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        logging_steps=config.training.logging_steps,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        seed=config.training.seed,
        fp16=config.training.fp16,
    )

    trainer = SpanTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[FileLogCallback(config.logger)],
    )

    config.logger.info("Training Span Classifier on %s...", config.device)
    trainer.train()

    final_dir = config.model_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)

    config.logger.info("Saving Span Classifier to %s...", final_dir)
    torch.save(model.state_dict(), final_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(final_dir)
    save_json(path=final_dir / "label_list.json", data=label_list, logger=config.logger)

    config.save_training_params()
    config.logger.info("Pipeline complete (run=%s). Saved to %s", config.run_name, final_dir)


if __name__ == "__main__":
    main()
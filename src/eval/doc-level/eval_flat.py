import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from common.paths import PROCESSED, METRICS
from common.logging import setup_logger
from common.json_helpers import load_json, save_json
from eval.metrics import DocLevelEvaluator


@dataclass
class Config:
    model_dir: Path          # .../doc-classifier-flat/{run_name}/final_model
    val_path: Path           = PROCESSED / "val_data_doc_level.json"
    label_mapping_path: Path = PROCESSED / "label_mapping.json"
    thresholds: tuple        = (0.1, 0.2, 0.3, 0.4, 0.5)
    batch_size: int          = 32
    max_length: int          = 128
    device                   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        self.model_dir = Path(self.model_dir)
        self.run_dir = self.model_dir.parent  # .../doc-classifier-flat/{run_name}
        self.run_name = self.run_dir.name
        self.eval_path = METRICS / f"doc_flat.{self.run_name}.json"
        self.logger = setup_logger(f"eval.doc_flat.{self.run_name}", f"eval_doc_flat_{self.run_name}.log")


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


def generate_probabilities(model, dataloader, device):
    all_probs, all_true_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            probs = torch.sigmoid(
                model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)).logits
            )
            all_probs.extend(probs.cpu().numpy())
            all_true_labels.extend(batch["labels"].cpu().numpy())
    return np.array(all_probs), np.array(all_true_labels)


def to_label_lists(binary_matrix, all_labels):
    return [[all_labels[i] for i, val in enumerate(row) if val == 1] for row in binary_matrix]


def main():
    if len(sys.argv) < 2:
        print("Usage: python eval_flat.py <path/to/final_model>")
        sys.exit(1)

    config = Config(model_dir=Path(sys.argv[1]))
    config.logger.info("Evaluating flat classifier at %s on %s", config.model_dir, config.device)

    val_data = load_json(config.val_path, config.logger)
    label_mapping = load_json(config.label_mapping_path, config.logger)
    id2label = {int(v): k for k, v in label_mapping.items()}
    all_labels = [id2label[i] for i in range(len(id2label))]

    tokenizer = AutoTokenizer.from_pretrained(config.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_dir).to(config.device).eval()

    dataset = MultiLabelDataset(val_data, tokenizer, config.max_length)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    all_probs, all_true_labels = generate_probabilities(model, dataloader, config.device)
    y_true_lists = to_label_lists(all_true_labels, all_labels)

    evaluator = DocLevelEvaluator(all_labels,config.logger)
    results_by_threshold = {}

    for thresh in config.thresholds:
        binary_preds = (all_probs >= thresh).astype(int)
        y_pred_lists = to_label_lists(binary_preds, all_labels)
        results = evaluator.evaluate(y_true_lists, y_pred_lists)
        results_by_threshold[thresh] = results
        config.logger.info("Threshold %.1f | Macro F1: %.4f", thresh, results["overall"]["macro_f1"])

    best_thresh = max(config.thresholds, key=lambda t: results_by_threshold[t]["overall"]["macro_f1"])
    evaluator.print_report(
        results_by_threshold[best_thresh],
        title=f"FLAT {len(all_labels)}-CLASS CLASSIFIER (Best Thresh: {best_thresh})"
    )

    save_json(
        path=config.eval_path,
        data={"best_threshold": best_thresh, "results": results_by_threshold[best_thresh]},
        logger=config.logger,
    )
    config.logger.info("Eval results saved to %s", config.eval_path)


if __name__ == "__main__":
    main()
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from common.paths import PROCESSED, METRICS
from common.logging import setup_logger
from common.json_helpers import load_json, save_json
from eval.metrics import DocLevelEvaluator


@dataclass
class Config:
    val_path: Path           = PROCESSED / "val_data_doc_level.json"
    label_mapping_path: Path = PROCESSED / "label_mapping.json"
    taxonomy_path: Path      = PROCESSED / "taxonomy_autogen_v3.csv"
    eval_path: Path          = METRICS / "doc_regex.json"

    def __post_init__(self):
        self.logger = setup_logger("eval.regex_baseline", "eval_regex_baseline.log")


def compile_regex_patterns(taxonomy: pd.DataFrame):
    return {
        row["cat_label"]: re.compile(row["regex"], re.IGNORECASE)
        for _, row in taxonomy.iterrows() if pd.notna(row.get("regex"))
    }


def main():
    config = Config()
    config.logger.info("Evaluating regex document-level predictions.")

    # Get validation data
    val_records = load_json(config.val_path, config.logger)
    label2id = load_json(config.label_mapping_path, config.logger)
    id2label = {int(v): k for k, v in label2id.items()}

    config.logger.info("Loading regex taxonomy...")
    taxonomy = pd.read_csv(config.taxonomy_path)
    regex_patterns = compile_regex_patterns(taxonomy)

    # Evaluate each against validation data
    evaluator = DocLevelEvaluator(all_labels=list(id2label.values()), logger=config.logger)
    y_true_lists, y_pred_lists = [], []

    for record in val_records:
        true_labels = [id2label[i] for i, val in enumerate(record.get("labels", [])) if val == 1]
        pred_labels = [lbl for lbl, pat in regex_patterns.items() if pat.search(record.get("text", ""))]

        y_true_lists.append(true_labels)
        y_pred_lists.append(pred_labels)

    results = evaluator.evaluate(y_true_lists, y_pred_lists)
    evaluator.print_report(results, title="PER-LABEL REGEX DOCUMENT CLASSIFICATION METRICS")

    save_json(path=config.eval_path, data=results, logger=config.logger)
    config.logger.info("Results saved to %s", config.eval_path)


if __name__ == "__main__":
    main()
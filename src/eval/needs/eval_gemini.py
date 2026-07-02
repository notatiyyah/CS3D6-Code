from dataclasses import dataclass
from typing import Dict
from pathlib import Path

import pandas as pd

from common.paths import PROCESSED, METRICS
from common.logging import setup_logger
from common.json_helpers import load_json, save_json

from eval.evaluators import SpanEvaluator


@dataclass
class Config:
    model_name: str          = "gemini"
    gt_span_path: Path       = PROCESSED / "val_data.json"
    predictions_path: Path   = PROCESSED / "gold_standard_gemini_pre_annotated.json"
    taxonomy_path: Path      = PROCESSED / "taxonomy_autogen_v3.csv"

    def __post_init__(self):
        self.logger = setup_logger(
            f"eval.{self.model_name}_span",
            f"eval_{self.model_name}_span.log",
        )
        self.eval_path = METRICS / f"span_{self.model_name}.json"


def load_predictions(config: Config) -> Dict[str, list]:
    """Get Gemini predictions from json. Returns a lookup dictionary by ID."""
    predictions = load_json(config.predictions_path, config.logger)
    return {
        record["data"]["id"]: record["predictions"][0]["result"]
        for record in predictions
        if record.get("data", {}).get("id") and record.get("predictions")
    }

def get_true_spans(record):
    """Span-level ground truth uses only 'needs' — 'persons' also carry
    start/end/label but are deliberately excluded from this eval."""
    return [
        (item["start"], item["end"], item["label"])
        for item in record.get("needs", []) + record.get("persons", [])
        if "label" in item
    ]

def get_predicted_spans(predictions):
    """Returns a list of (start, end, label) spans from Gemini's Label Studio
    span annotations."""
    results = []

    for item in predictions:
        if item.get("type") != "labels":
            continue
        label = item["value"]["labels"][0]

        # Clean person labels
        if label == "Person_Pronoun":
            continue # Skip
        if label in ["Person_Name", "Person_Role"]:
            label = label.lower()
        
        results.append((
            item["value"]["start"],
            item["value"]["end"],
            label
        ))

    return results


def main():
    config = Config()
    config.logger.info("Evaluating %s span-level predictions.", config.model_name)

    taxonomy = pd.read_csv(config.taxonomy_path)
    all_labels = sorted(taxonomy["cat_label"].dropna().unique())
    all_labels.append("person_ref")

    # Load ground truth (validation set)
    gt_records = load_json(config.gt_span_path, config.logger)
    y_true = []
    y_pred = []

    # Load predictions
    prediction_lookup = load_predictions(config)

    for record in gt_records:
        doc_id = record["id"]
        y_true.append(get_true_spans(record))
        y_pred.append(get_predicted_spans(prediction_lookup.get(doc_id, [])))

    evaluator = SpanEvaluator(all_labels, config.logger)
    results = evaluator.evaluate(y_true, y_pred)

    evaluator.print_report(
        results,
        title=f"{config.model_name.upper()} SPAN-LEVEL CLASSIFICATION",
    )

    save_json(
        path=config.eval_path,
        data=results,
        logger=config.logger,
    )
    config.logger.info("Results saved to %s", config.eval_path)


if __name__ == "__main__":
    main()
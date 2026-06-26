from dataclasses import dataclass, field
from typing import Dict
from pathlib import Path

from common.paths import PROCESSED, METRICS
from common.logging import setup_logger
from common.json_helpers import load_json, save_json

from eval.metrics import DocLevelEvaluator


@dataclass
class Config:
    model_name: str          = "gemini"
    gt_doc_path: Path        = PROCESSED / "val_data_doc_level.json"
    gt_span_path: Path       = PROCESSED / "val_data.json"
    predictions_path: Path   = PROCESSED / "gold_standard_gemini_pre_annotated.json"
    label_mapping_path: Path = PROCESSED / "label_mapping.json"

    def __post_init__(self):
        self.logger = setup_logger(
            f"eval.{self.model_name}",
            f"eval_{self.model_name}.log",
        )
        self.eval_path = METRICS / f"doc_{self.model_name}.json"


def load_predictions(config: Config) -> Dict[str, list[str]]:
    '''Get Gemini predictions from json. Returns a lookup dictionary by ID.'''
    predictions = load_json(config.predictions_path, config.logger)
    return {
        record["data"]["id"]: record["predictions"][0]["result"]
        for record in predictions
        if record.get("data", {}).get("id") and record.get("predictions")
    }

def get_true_labels(record):
    return list({
        item["label"]
        for item in record.get("needs", []) + record.get("persons", [])
        if "label" in item
    })

def get_predicted_labels(predictions):
    '''Returns a list of predicted doc-level labels from gemini's span annotations.'''
    return list({
        item["value"]["labels"][0]
        for item in predictions
        if item.get("type") == "labels"
        and "labels" in item.get("value", {})
    })

def main():
    config = Config()
    config.logger.info("Evaluating %s document-level predictions.", config.model_name)

    label2id = load_json(config.label_mapping_path, config.logger)
    id2label = {int(v): k for k, v in label2id.items()}
    all_labels = set(label2id.keys())

    # Load ground truth (validation set)
    gt_records = load_json(config.gt_doc_path, config.logger)
    y_true = []
    y_pred = []

    # Load predictions
    prediction_lookup = load_predictions(config)

    for record in gt_records:
        doc_id = record["id"]
        y_true.append(record["label_names"])
        y_pred.append(get_predicted_labels(prediction_lookup.get(doc_id, [])))

    evaluator = DocLevelEvaluator(all_labels,config.logger)
    results = evaluator.evaluate(y_true, y_pred)

    evaluator.print_report(
        results,
        title=f"{config.model_name.upper()} DOCUMENT-LEVEL CLASSIFICATION",
    )

    save_json(
        path=config.eval_path,
        data=results,
        logger=config.logger,
    )

    config.logger.info("Results saved to %s", config.eval_path)


if __name__ == "__main__":
    main()
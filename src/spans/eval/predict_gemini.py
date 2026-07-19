"""
Generate span predictions using Gemini baseline.
Outputs standardized predictions for unified evaluation.
"""
from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path
import uuid

import pandas as pd

from common.paths import PROCESSED, PREDICTIONS, VAL_DATA
from common.logging import setup_logger
from common.json_helpers import load_json, save_json


@dataclass
class Config:
    model_name: str = "gemini"
    val_path: Path = VAL_DATA
    predictions_path: Path = PROCESSED / "gold_standard_gemini_pre_annotated.json"
    taxonomy_path: Path = PROCESSED / "taxonomy_autogen_v3.csv"
    person_labels: List[str] = field(default_factory=lambda: ["person_role", "person_name"])

    def __post_init__(self):
        self.logger = setup_logger(
            f"predict.{self.model_name}_spans",
            f"predict_{self.model_name}_spans.log",
        )
        PREDICTIONS.mkdir(parents=True, exist_ok=True)
        self.output_path = PREDICTIONS / f"span.{self.model_name}.json"


def load_predictions(config: Config) -> Dict[str, list]:
    """Get Gemini predictions from json. Returns a lookup dictionary by ID."""
    predictions = load_json(config.predictions_path, config.logger)
    return {
        record["data"]["id"]: {"text": record["data"]["note_content"], "predictions": record["predictions"][0]["result"]}
        for record in predictions
        if record.get("data", {}).get("id") and record.get("predictions")
    }


def get_predicted_spans(doc):
    """Returns a list of span dicts from Gemini's Label Studio
    span annotations."""
    results = []

    for item in doc.get("predictions", []):
        if item.get("type") != "labels":
            continue # Skip relations

        for label in item["value"]["labels"]:
            span = {
                "id": str(uuid.uuid4())[:8],
                "text": item["value"]["text"],
                "start": item["value"]["start"],
                "end": item["value"]["end"],
                "label": label.lower(), # lowercase because some were uppercased
                "confidence": 1.0,
            }
            results.append(span)
    
    return results


def main():
    config = Config()
    config.logger.info("Generating %s span predictions...", config.model_name)

    # Load val records
    val_records = load_json(config.val_path, config.logger)

    # Generate predictions in standardized format
    prediction_lookup = load_predictions(config)
    predictions = []

    for record in val_records:
        doc_id = record["id"]
        text = record.get("text", "")

        # Get predicted spans
        doc_spans = get_predicted_spans(prediction_lookup.get(doc_id, {}))

        prediction = {
            "id": doc_id,
            "text": text,
            "model": config.model_name,
            "needs": [x for x in doc_spans if x['label'] not in config.person_labels],
            "persons": [x for x in doc_spans if x['label'] in config.person_labels],
            "tenure_ids": record.get("tenure_ids"),
            "household_members": record.get("household_members")
        }
        predictions.append(prediction)

    save_json(path=config.output_path, data=predictions, logger=config.logger)
    config.logger.info(
        "Generated predictions for %d records. Saved to %s",
        len(predictions),
        config.output_path,
    )


if __name__ == "__main__":
    main()

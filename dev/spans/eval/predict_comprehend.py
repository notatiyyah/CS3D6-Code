"""
Generate span predictions using AWS Comprehend baseline.
Outputs standardized predictions for unified evaluation.
"""
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
import uuid

from common.json_helpers import load_json, save_json
from common.paths import PROCESSED, ANNOTATIONS, PREDICTIONS, VAL_DATA, TEST_DATA
from common.logging import setup_logger


@dataclass
class Config:
    model_name: str = "comprehend"
    val_path: Path = TEST_DATA
    preds_model_a_path: Path = ANNOTATIONS / "comprehend_output_a.jsonl"
    preds_model_b_path: Path = ANNOTATIONS / "comprehend_output_b.jsonl"
    preds_model_c_path: Path = ANNOTATIONS / "comprehend_output_c.jsonl"
    person_labels: List[str] = field(default_factory=lambda: ["person_role", "person_name"])

    def __post_init__(self):
        self.logger = setup_logger(
            f"predict.{self.model_name}_spans",
            f"predict_{self.model_name}_spans.log",
        )
        PREDICTIONS.mkdir(parents=True, exist_ok=True)
        self.output_path = PREDICTIONS / f"span.{self.model_name}.json"


def load_comprehend_jsonl_to_dict(filepath: Path) -> dict:
    """Loads a JSONL file and returns a dictionary keyed by the 'Line' number."""
    preds_dict = {}
    with open(filepath, "r") as f:
        for line in f:
            record = json.loads(line)
            preds_dict[record["Line"]] = record
    return preds_dict


def extract_spans_from_comprehend(record_a, record_b, record_c, text):
    """Extract spans from Comprehend entity output."""
    spans = []

    for record in [record_a, record_b, record_c]:
        if not record or "Entities" not in record:
            continue

        for entity in record["Entities"]:
            label = entity.get("Type", "").lower() # force lowercase
            start = entity.get("BeginOffset")
            end = entity.get("EndOffset")

            if start is not None and end is not None:
                text_span = text[start:end]
                spans.append(
                    {
                        "id": str(uuid.uuid4())[:8],
                        "text": text_span,
                        "start": start, 
                        "end": end,
                        "label": label,
                        "confidence": entity.get("Score")
                    }
                )

    return spans


def main():
    config = Config()
    config.logger.info("Generating %s span predictions...", config.model_name)

    # Load val records
    val_records = load_json(config.val_path, config.logger)

    # Load Comprehend predictions
    config.logger.info("Loading Comprehend predictions...")
    preds_a = load_comprehend_jsonl_to_dict(config.preds_model_a_path)
    preds_b = load_comprehend_jsonl_to_dict(config.preds_model_b_path)
    preds_c = load_comprehend_jsonl_to_dict(config.preds_model_c_path)

    # Generate predictions in standardized format
    predictions = []

    for i, record in enumerate(val_records):
        text = record.get("text", "")

        # Get Comprehend predictions for this record (line number = index)
        comprehend_a = preds_a.get(i)
        comprehend_b = preds_b.get(i)
        comprehend_c = preds_c.get(i)

        spans = extract_spans_from_comprehend(comprehend_a, comprehend_b, comprehend_c, text)

        prediction = {
            "id": record.get("id"),
            "text": text,
            "date": record.get("note_date"),
            "model": config.model_name,
            "needs": [x for x in spans if x['label'] not in config.person_labels],
            "persons": [x for x in spans if x['label'] in config.person_labels],
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

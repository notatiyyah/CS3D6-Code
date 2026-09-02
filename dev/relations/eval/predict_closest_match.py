"""
Generate relation predictions using closest preceding match heuristic.
For each need, links to the person with the largest start offset that is
still <= the need's start offset (most recent preceding mention).
Outputs standardized predictions for unified evaluation.
"""
import sys
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
from pathlib import Path

from common.paths import PROCESSED, PREDICTIONS, VAL_DATA, TEST_DATA
from common.logging import setup_logger
from common.json_helpers import load_json, save_json


@dataclass
class Config:
    data_path: Path = TEST_DATA # Can be overridden by predicted spans
    model_name: str = "closest_match"
    output_dir: Path = PREDICTIONS
    output_filename: str = "relation.{model_names}.json"

    def __post_init__(self):
        self.data_path = Path(self.data_path)
        self.logger = setup_logger(
            f"predict.{self.model_name}_relations",
            f"predict_{self.model_name}_relations.log",
        )

def predict_pairs(doc: dict) -> Set[Tuple[str, str]]:
    """
    For each need, find the person with the largest start offset that is
    still <= the need's start offset (most recent preceding mention).
    Needs with no preceding person produce no prediction.
    Property-level needs are ignored (they don't need relations).
    """
    needs = doc.get("needs", [])
    people = doc.get("persons", [])

    predicted = set()
    for need in needs:
        # Ignore property-level labels (they never need relations)
        if need["label"].startswith("property_level"):
            continue

        # Find all people that appear before this need
        preceding = [p for p in people if p["start"] <= need["start"]]
        if not preceding:
            continue

        # Link to the closest (most recent) preceding person
        closest_person = max(preceding, key=lambda p: p["start"])
        predicted.add((str(need["id"]).strip(), str(closest_person["id"]).strip()))

    return predicted


def predict_relations(data: List[dict], config: Config) -> Dict[str, dict]:
    """Generate relation predictions using closest preceding match heuristic."""
    predictions = []

    for doc in data:
        # Predict pairs using closest preceding match
        predicted_pairs = predict_pairs(doc)

        # Convert to standardized format
        doc_relations = [
            {
                "from": need_id,
                "to": person_id,
                "confidence": 1.0
            }
            for need_id, person_id in predicted_pairs
        ]
        predictions.append({
            "id": doc["id"],
            "text": doc["text"],
            "date": doc.get("date"),
            "model": doc.get("model", "oracle") + "_" + config.model_name, # Either spanModel_relationModel or oracle_relationModel
            "needs": doc["needs"],
            "persons": doc["persons"],
            "relations": doc_relations,
            "tenure_ids": doc.get("tenure_ids", []),
            "household_members": doc.get("household_members", []),
        })

    config.logger.info(f"Generated predictions for {len(predictions)} documents")
    return predictions


def main():
    if len(sys.argv) > 1:
        config = Config(sys.argv[1]) # Override val data with other data
    else:
        config = Config()
    config.logger.info("Generating %s relation predictions...", config.model_name)

    # Load data
    data = load_json(config.data_path, config.logger)
    config.logger.info(f"Loaded {len(data)} documents")

    # Generate predictions
    predictions = predict_relations(data, config)
    model_names = predictions[0]['model']
    filename = config.output_filename.format(model_names=model_names)

    # Save predictions
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / filename
    save_json(output_path, predictions, config.logger)
    config.logger.info(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()

from dataclasses import dataclass
from typing import Dict, Set, Tuple, Optional
from pathlib import Path

from common.paths import PROCESSED, METRICS
from common.logging import setup_logger
from common.json_helpers import load_json, save_json
from eval.evaluators import RelationEvaluator


@dataclass
class Config:
    model_name: str        = "gemini"
    gt_path: Path          = PROCESSED / "val_data.json"
    predictions_path: Path = PROCESSED / "gold_standard_gemini_pre_annotated.json"

    def __post_init__(self):
        self.logger = setup_logger(
            f"eval.{self.model_name}_relation",
            f"eval_{self.model_name}_relation.log",
        )
        self.eval_path = METRICS / f"relation_{self.model_name}.json"


def load_predictions(config: Config) -> Dict[str, list]:
    predictions = load_json(config.predictions_path, config.logger)
    return {
        record["data"]["id"]: record["predictions"][0]["result"]
        for record in predictions
        if record.get("data", {}).get("id") and record.get("predictions")
    }


def find_overlapping_id(start: int, end: int, entities: list) -> str | None:
    """Return the ID of the first entity that overlaps with (start, end)."""
    for e in entities:
        if max(start, e["start"]) < min(end, e["end"]):
            return str(e["id"])
    return None


def make_predict_fn(prediction_lookup: dict):
    """Returns a predict_fn(doc) compatible with RelationEvaluator.
    Gemini span boundaries may not exactly match gold, so predicted spans
    are matched to gold entities by overlap to recover IDs."""
    def predict_fn(doc: dict) -> Set[Tuple[str, str]]:
        predictions = prediction_lookup.get(doc["id"], [])
        needs = doc.get("needs", [])
        persons = doc.get("persons", [])

        spans = {
            item["id"]: (item["value"]["start"], item["value"]["end"])
            for item in predictions
            if item.get("type") == "labels"
        }

        predicted = set()
        for item in predictions:
            if item.get("type") != "relation":
                continue
            from_span = spans.get(item.get("from_id"))
            to_span = spans.get(item.get("to_id"))
            if from_span is None or to_span is None:
                continue
            from_id = find_overlapping_id(*from_span, needs)
            to_id = find_overlapping_id(*to_span, persons)
            if from_id and to_id:
                predicted.add((from_id, to_id))

        return predicted

    return predict_fn


def main():
    config = Config()
    config.logger.info("Evaluating %s relation-level predictions.", config.model_name)

    gt_records = load_json(config.gt_path, config.logger)
    prediction_lookup = load_predictions(config)

    evaluator = RelationEvaluator(config.logger)
    results = evaluator.evaluate(gt_records, predict_fn=make_predict_fn(prediction_lookup))
    evaluator.print_report(results, title=f"{config.model_name.upper()} RELATION EXTRACTION")

    config.eval_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(path=config.eval_path, data=results, logger=config.logger)
    config.logger.info("Results saved to %s", config.eval_path)


if __name__ == "__main__":
    main()

"""
Relation Extraction baseline — closest preceding match.

No model: for each 'need' span, predict a relation to whichever 'person'
span was most recently mentioned before it in the text (by start offset).
If no person precedes a need, no relation is predicted for it (counts as
a miss against any gold relation for that need, not a fallback guess).
"""

from dataclasses import dataclass, field
from pathlib import Path

from common.paths import PROCESSED, METRICS
from common.logging import setup_logger
from common.json_helpers import load_json, save_json
from eval.evaluators import RelationEvaluator


@dataclass
class Config:
    val_path: Path = field(default_factory=lambda: PROCESSED / "val_data.json")
    eval_dir: Path = field(default_factory=lambda: METRICS)

    def __post_init__(self):
        self.logger = setup_logger("eval.relation_closest_match", "eval_relation_closest_match.log")


def predict_pairs(doc: dict) -> set:
    """For each need, find the person with the largest start offset that is
    still <= the need's start offset (most recent preceding mention).
    Needs with no preceding person produce no prediction."""
    needs = doc.get("needs", [])
    people = doc.get("persons", [])

    predicted = set()
    for need in needs:
        if need['label'].startswith('property_level'):
            continue # Ignore property level labels. Will never need relations.
        preceding = [p for p in people if p["start"] <= need["start"]]
        if not preceding:
            continue
        closest_person = max(preceding, key=lambda p: p["start"])
        predicted.add((str(need["id"]).strip(), str(closest_person["id"]).strip()))

    return predicted


def main():
    config = Config()
    config.logger.info("Evaluating closest-preceding-match relation baseline...")

    val_records = load_json(config.val_path, config.logger)

    evaluator = RelationEvaluator(config.logger)
    results = evaluator.evaluate(val_records, predict_fn=predict_pairs)
    evaluator.print_report(results, title="CLOSEST PRECEDING MATCH RELATION EXTRACTION")

    save_json(path=config.eval_dir / "relation_closest_match_results.json", data=results, logger=config.logger)
    config.logger.info("Results saved to %s", config.eval_dir / "relation_closest_match_results.json")


if __name__ == "__main__":
    main()
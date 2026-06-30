"""
Relation Extraction eval — pair-level exact match.

Scores predicted (need_id, person_id) pairs against gold relations. No
co-reference resolution layer: a person mention is scored as its exact
annotated ID, not the real-world individual it refers to (co-reference
chains were not annotated — out of scope by design, not an oversight).
"""

import sys
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from common.paths import PROCESSED, METRICS
from common.logging import setup_logger
from common.json_helpers import load_json, save_json
from shared.relation_model import insert_markers, score_documents


@dataclass
class Config:
    model_dir: Path  # .../needs-relation-extractor/{run_name}/final_model
    val_path: Path = field(default_factory=lambda: PROCESSED / "val_data.json")
    max_length: int = 512
    batch_size: int = 16

    def __post_init__(self):
        self.model_dir = Path(self.model_dir)
        self.run_dir = self.model_dir.parent  # .../needs-relation-extractor/{run_name}
        self.run_name = self.run_dir.name
        self.eval_path = METRICS / f"relations.{self.run_name}.json"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = setup_logger(f"eval.relation_extraction.{self.run_name}", f"eval_relation_extraction_{self.run_name}.log")


def predict_pairs(doc: dict, model, tokenizer, config: Config) -> set:
    """Run inference over every (need, person) pair in the document, return
    the set of (need_id, person_id) pairs predicted as linked (label=1)."""
    text = doc["text"]
    needs = doc.get("needs", [])
    people = doc.get("persons", [])

    pairs, marked_texts = [], []
    for need in needs:
        for person in people:
            pairs.append((str(need["id"]).strip(), str(person["id"]).strip()))
            marked_texts.append(insert_markers(text, need, person))

    if not marked_texts:
        return set()

    predicted = set()
    with torch.no_grad():
        for i in range(0, len(marked_texts), config.batch_size):
            batch_texts = marked_texts[i:i + config.batch_size]
            batch_pairs = pairs[i:i + config.batch_size]

            inputs = tokenizer(
                batch_texts, truncation=True, padding=True,
                max_length=config.max_length, return_tensors="pt"
            ).to(config.device)

            logits = model(**inputs).logits
            preds = torch.argmax(logits, dim=1).cpu().tolist()

            for (need_id, person_id), pred in zip(batch_pairs, preds):
                if pred == 1:
                    predicted.add((need_id, person_id))

    return predicted


def main():
    if len(sys.argv) < 2:
        print("Usage: python eval_relation_extraction.py <path/to/final_model>")
        sys.exit(1)

    config = Config(model_dir=Path(sys.argv[1]))
    config.logger.info("Evaluating relation extractor at %s on %s", config.model_dir, config.device)

    val_records = load_json(config.val_path, config.logger)

    tokenizer = AutoTokenizer.from_pretrained(config.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_dir).to(config.device).eval()

    results = score_documents(
        val_records,
        predict_fn=lambda doc: predict_pairs(doc, model, tokenizer, config),
        logger=config.logger,
    )

    save_json(path=config.eval_path, data=results, logger=config.logger)
    config.logger.info("Eval results saved to %s", config.eval_path)


if __name__ == "__main__":
    main()
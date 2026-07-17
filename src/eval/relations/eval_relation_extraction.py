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
from typing import Dict, Set, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from common.paths import PROCESSED, METRICS
from common.logging import setup_logger
from common.json_helpers import load_json, save_json
from shared.relation_model import insert_markers
from eval.evaluators import RelationEvaluator


@dataclass
class Config:
    model_dir: Path
    val_path: Path = field(default_factory=lambda: PROCESSED / "val_data.json")
    batch_size: int = 16

    def __post_init__(self):
        self.model_dir = Path(self.model_dir)
        self.run_dir = self.model_dir.parent
        self.run_name = self.run_dir.name
        self.eval_path = METRICS / f"relations.{self.run_name}.json"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = setup_logger(f"eval.relation_extraction.{self.run_name}", f"eval_relation_extraction_{self.run_name}.log")
        self._set_model_params()
    
    def _set_model_params(self):
        config_path = self.run_dir / 'config.json'
        run_config = load_json(config_path, self.logger)

        self.base_model = run_config['base_model']
        self.max_length = run_config['max_length']

def predict_relations(doc, model, tokenizer, device, max_length, batch_size=8):
    text = doc.get("text", "")
    needs = doc.get("needs", [])
    persons = doc.get("persons", [])

    if not needs or not persons:
        return []

    pairs, marked_texts = [], []
    for need in needs:
        if need['label'].startswith('property_level'):
            continue # Ignore property level labels. Will never need relations.
        for person in persons:
            pairs.append((need["id"], person["id"]))
            marked_texts.append(insert_markers(text, need, person))

    if not marked_texts:
        return []

    all_labels = []
    with torch.inference_mode():
        # Process in bite-sized chunks to save memory
        for i in range(0, len(marked_texts), batch_size):
            chunk = marked_texts[i : i + batch_size]
            batch = tokenizer(
                chunk,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            logits = model(**batch).logits
            labels = torch.argmax(logits, dim=1).cpu().tolist()
            all_labels.extend(labels)

    return [pair for pair, label in zip(pairs, all_labels) if label == 1]

def make_predict_fn(config):
    """Builds a function to return need, person pairs from inference."""

    config.logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_dir).to(config.device).eval()

    config.logger.info("Evaluating...")
    def predict_fn(doc: dict) -> Set[Tuple[str, str]]:
        return predict_relations(doc, model, tokenizer, config.device, config.max_length, config.batch_size)

    return predict_fn


def main():
    if len(sys.argv) < 2:
        print("Usage: python eval_relation_extraction.py <path/to/final_model>")
        sys.exit(1)

    config = Config(model_dir=Path(sys.argv[1]))
    config.logger.info("Evaluating relation extractor at %s on %s (base_model=%s)", 
                        config.model_dir, config.device, config.base_model)

    val_records = load_json(config.val_path, config.logger)

    evaluator = RelationEvaluator(config.logger)
    results = evaluator.evaluate(
        val_records,
        predict_fn=make_predict_fn(config),
    )

    evaluator.print_report(results, title="RELATION EXTRACTION")
    save_json(path=config.eval_path, data=results, logger=config.logger)
    config.logger.info("Eval results saved to %s", config.eval_path)


if __name__ == "__main__":
    main()
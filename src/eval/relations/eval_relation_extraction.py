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
from shared.relation_model import insert_markers
from eval.evaluators import RelationEvaluator


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


def build_corpus_pairs(val_records: list) -> tuple[list, list, list]:
    """Collect all (need, person) pairs across all documents in one pass.

    Returns:
        all_texts:    marked text for each pair
        all_pairs:    (need_id, person_id) for each pair
        doc_slices:   (start, end) index into the above lists for each document
    """
    all_texts, all_pairs, doc_slices = [], [], []
    for doc in val_records:
        start = len(all_texts)
        text = doc["text"]
        for need in doc.get("needs", []):
            for person in doc.get("persons", []):
                all_pairs.append((str(need["id"]).strip(), str(person["id"]).strip()))
                all_texts.append(insert_markers(text, need, person))
        doc_slices.append((start, len(all_texts)))
    return all_texts, all_pairs, doc_slices


def run_batched_inference(all_texts: list, all_pairs: list, model, tokenizer, config: Config) -> list[int]:
    """Single batched inference pass over all pairs. Returns per-pair predictions."""
    preds = []
    with torch.inference_mode():
        for i in range(0, len(all_texts), config.batch_size):
            inputs = tokenizer(
                all_texts[i:i + config.batch_size],
                truncation=True, padding=True,
                max_length=config.max_length, return_tensors="pt"
            ).to(config.device)
            logits = model(**inputs).logits
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
    return preds


def main():
    if len(sys.argv) < 2:
        print("Usage: python eval_relation_extraction.py <path/to/final_model>")
        sys.exit(1)

    config = Config(model_dir=Path(sys.argv[1]))
    config.logger.info("Evaluating relation extractor at %s on %s", config.model_dir, config.device)

    val_records = load_json(config.val_path, config.logger)

    config.logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_dir).to(config.device).eval()

    config.logger.info("Building pairs across %d documents...", len(val_records))
    all_texts, all_pairs, doc_slices = build_corpus_pairs(val_records)
    config.logger.info("Running inference over %d pairs...", len(all_texts))
    all_preds = run_batched_inference(all_texts, all_pairs, model, tokenizer, config)

    doc_index = {id(doc): idx for idx, doc in enumerate(val_records)}

    config.logger.info("Evaluating...")

    evaluator = RelationEvaluator(config.logger)

    def predict_doc(doc):
        start, end = doc_slices[doc_index[id(doc)]]
        return {
            pair
            for pair, pred in zip(all_pairs[start:end], all_preds[start:end])
            if pred == 1
        }

    results = evaluator.evaluate(
        val_records,
        predict_fn=predict_doc,
    )

    evaluator.print_report(results, title="RELATION EXTRACTION")
    save_json(path=config.eval_path, data=results, logger=config.logger)
    config.logger.info("Eval results saved to %s", config.eval_path)


if __name__ == "__main__":
    main()
"""
BIO Token Classifier eval — span-level, via HF's NER pipeline.

aggregation_strategy="simple" merges consecutive B-/I- tokens of the same
entity type into a single (start, end, entity_group) span, so predictions
land in the same exact-span format as span v3 and are scored with the same
SpanEvaluator (loose/strict) for a fair side-by-side comparison.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import pipeline, AutoTokenizer

from common.paths import PROCESSED, METRICS
from common.logging import setup_logger
from common.json_helpers import load_json, save_json
from eval.evaluators import SpanEvaluator


@dataclass
class Config:
    model_dir: Path  # .../needs-bio-classifier/{run_name}/final_model
    val_path: Path = field(default_factory=lambda: PROCESSED / "val_data.json")
    max_length: int = 512

    def __post_init__(self):
        self.model_dir = Path(self.model_dir)
        self.run_dir = self.model_dir.parent  # .../needs-bio-classifier/{run_name}
        self.run_name = self.run_dir.name
        self.eval_path: str = METRICS / f"span_bio_classifier.{self.run_name}.json"

        self.device = 0 if torch.cuda.is_available() else -1
        self.logger = setup_logger(f"eval.bio_classifier.{self.run_name}", f"eval_bio_classifier_{self.run_name}.log")


def truncate_to_max_length(text: str, tokenizer, max_length: int) -> str:
    """The pipeline's truncation kwarg isn't accepted by
    TokenClassificationPipeline in newer transformers versions, so truncate
    manually: tokenize with truncation, check whether truncation actually
    happened (compare against an untruncated encode), and if so, cut the
    text at the last real token's end offset so character offsets returned
    by the pipeline stay consistent with what the model actually saw."""
    untruncated_len = len(tokenizer(text, add_special_tokens=True)["input_ids"])
    if untruncated_len <= max_length:
        return text

    encoded = tokenizer(text, truncation=True, max_length=max_length, return_offsets_mapping=True)
    for tok_start, tok_end in reversed(encoded["offset_mapping"]):
        if tok_end > 0:
            return text[:tok_end]
    return text


def main():
    if len(sys.argv) < 2:
        print("Usage: python eval_bio_classifier.py <path/to/final_model>")
        sys.exit(1)

    config = Config(model_dir=Path(sys.argv[1]))
    config.logger.info("Evaluating BIO classifier at %s (device=%s)", config.model_dir, config.device)

    val_records = load_json(config.val_path, config.logger)
    all_labels = sorted({n["label"] for r in val_records for n in r.get("needs", []) if "label" in n})

    config.logger.info("Loading NER pipeline...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_dir)
    pipe = pipeline(
        "ner", model=str(config.model_dir), tokenizer=tokenizer,
        aggregation_strategy="simple", device=config.device,
    )
    evaluator = SpanEvaluator(all_labels, config.logger)

    y_true, y_pred = [], []
    for record in val_records:
        text = record.get("text", "")
        y_true.append([(n["start"], n["end"], n["label"]) for n in record.get("needs", []) if "label" in n])

        safe_text = truncate_to_max_length(text, tokenizer, config.max_length)
        entities = pipe(safe_text)
        y_pred.append([(e["start"], e["end"], e["entity_group"]) for e in entities])

    results = evaluator.evaluate(y_true, y_pred)
    evaluator.print_report(results, title="BIO-TAGGING NER METRICS")

    save_json(path=config.eval_path, data=results, logger=config.logger)
    config.logger.info("Eval results saved to %s", config.eval_path)


if __name__ == "__main__":
    main()
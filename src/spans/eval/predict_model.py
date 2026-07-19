"""
Generate span predictions using model with tuned thresholds (defaults to 0.5)
Outputs standardized predictions for unified evaluation.
"""
import sys
from dataclasses import dataclass, field
from typing import List
from pathlib import Path
import uuid

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from tqdm import tqdm

from common.paths import METRICS, PREDICTIONS, VAL_DATA, TEST_DATA
from common.logging import setup_logger
from common.json_helpers import load_json, save_json
from shared.evaluators import SpanEvaluator
from shared.span_model import SpanClassifier, generate_candidates, spans_overlap


@dataclass
class Config:
    model_dir: Path
    data_path: Path = TEST_DATA
    person_labels: List[str] = field(default_factory=lambda: ["person_role", "person_name"])

    def __post_init__(self):
        self.model_dir = Path(self.model_dir)
        self.run_name = self.model_dir.parent.name
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        self.logger = setup_logger(
            f"run_thresholded.span.{self.run_name}",
            f"run_thresholded_span_{self.run_name}.log",
        )

        PREDICTIONS.mkdir(parents=True, exist_ok=True)
        METRICS.mkdir(parents=True, exist_ok=True)

        self.output_path: Path = PREDICTIONS / f"span.{self.run_name}.json"
        self.eval_path: Path = METRICS / f"span.{self.run_name}.json"

        self.thresholds_path: Path = self.model_dir / "optimized_thresholds.json"
        config_path = self.model_dir.parent / "config.json"
        run_config = load_json(config_path, self.logger)
        self.base_model = run_config["base_model"]
        self.max_length = run_config["max_length"]
        self.max_candidate_size = run_config["max_candidate_size"]


def correct_offset(text: str, char_start: int, char_end: int) -> tuple:
    span_text = text[char_start:char_end]
    n_stripped = len(span_text) - len(span_text.lstrip())
    return char_start + n_stripped, char_end


def run_inference(text, model, tokenizer, config):
    # tokenise
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=config.max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    # generate candidates
    offsets = tokenized["offset_mapping"][0].tolist()
    candidates = generate_candidates(offsets, config.max_candidate_size)
    if not candidates:
        return [], None

    # run forward pass
    input_ids = tokenized["input_ids"].to(config.device)
    attention_mask = tokenized["attention_mask"].to(config.device)
    candidate_spans = torch.tensor(candidates, dtype=torch.long, device=config.device)

    outputs = model(input_ids, attention_mask, candidate_spans)
    probs = torch.softmax(outputs["logits"], dim=-1).cpu().numpy()

    # return predicted spans (fix offsets if sentencepiece)
    char_spans = []
    correct_leading_whitespace_offset = "deberta" in config.base_model.lower()
    for tok_start, tok_end in candidates:
        char_start = offsets[tok_start][0]
        char_end = offsets[tok_end - 1][1]

        if correct_leading_whitespace_offset:
            char_start, char_end = correct_offset(text, char_start, char_end)

        char_spans.append((char_start, char_end))

    return char_spans, probs


def deduplicate_predictions(pred_list):
    """Keep the highest-confidence prediction when spans overlap for the same label.
    pred_spans: dict of (id, text, start, end, label, confidence)."""
    if not pred_list:
        return []

    # Greedy highest-confidence first keeping, per label
    sorted_preds = sorted(pred_list, key=lambda item: item["confidence"], reverse=True)
    kept = []
    for pred in sorted_preds:
        if not any(
            pred["label"] == k["label"] and spans_overlap(pred["start"], pred["end"], k["start"], k["end"])
            for k in kept
        ):
            kept.append(pred)
    return kept

def threshold_and_format(text, char_spans, probs, label_list, thresholds):
    if probs is None or not char_spans:
        return []

    # Take most likely class & confidence
    no_entity_id = len(label_list)
    pred_classes = np.argmax(probs, axis=-1)
    pred_confs = np.max(probs, axis=-1)

    # Set up thresholds (force 1.0 for no_entity)
    thresh_arr = np.array(
        [thresholds.get(label_list[c], 0.5) if c != no_entity_id else 1.0 for c in pred_classes]
    )

    # Keep candidate if not predicted no_entity and clears threshold
    pred_span_list = [
        {
            "id": str(uuid.uuid4())[:8],
            "text": text[char_spans[i][0]:char_spans[i][1]],
            "start": char_spans[i][0],
            "end": char_spans[i][1],
            "label": label_list[pred_classes[i]],
            "confidence": float(pred_confs[i]),
        }
        for i in range(len(pred_classes))
        if pred_classes[i] != no_entity_id and pred_confs[i] >= thresh_arr[i]
    ]

    return deduplicate_predictions(pred_span_list)


def load_thresholds(config, label_list):
    if config.thresholds_path.exists():
        thresholds = load_json(config.thresholds_path, config.logger)
        config.logger.info("Loaded pre-fit thresholds from %s", config.thresholds_path)
        return thresholds

    config.logger.warning(
        "No thresholds file found at %s — defaulting to 0.5 for all %d classes",
        config.thresholds_path,
        len(label_list),
    )
    return {label: 0.5 for label in label_list}


def main():
    if len(sys.argv) < 2:
        print("Usage: python model_run_thresholded.py <path/to/final_model>")
        sys.exit(1)

    config = Config(sys.argv[1])
    config.logger.info("Running inference for %s on %s",config.model_dir, config.device)

    # Load val data
    val_records = load_json(config.data_path, config.logger)
    
    # Load model config
    label_list = load_json(config.model_dir / "label_list.json", config.logger)
    thresholds = load_thresholds(config, label_list)

    tokenizer = AutoTokenizer.from_pretrained(config.model_dir)
    model = SpanClassifier(config.base_model, num_labels=len(label_list)).to(config.device)

    state_dict = load_file(config.model_dir / "model.safetensors", device=str(config.device))
    load_result = model.load_state_dict(state_dict)
    config.logger.info("Loaded model weights: %s", load_result)
    model.eval()

    # Run inference
    predictions = []
    with torch.inference_mode():
        for record in tqdm(val_records):
            text = record.get("text", "")
            raw_spans, probs = run_inference(text, model, tokenizer, config)
            clean_spans = threshold_and_format(text, raw_spans, probs, label_list, thresholds)

            predictions.append(
                {
                    "id": record.get("id"),
                    "text": text,
                    "model": config.run_name,
                    "needs": [x for x in clean_spans if x['label'] not in config.person_labels],
                    "persons": [x for x in clean_spans if x['label'] in config.person_labels],
                    "tenure_ids": record.get("tenure_ids"),
                    "household_members": record.get("household_members")
                }
            )

    # Save out
    save_json(path=config.output_path, data=predictions, logger=config.logger)
    config.logger.info(
        "Generated predictions for %d records. Saved to %s",
        len(predictions),
        config.output_path,
    )

if __name__ == "__main__":
    main()
import sys
import uuid
from pathlib import Path
import os
from tqdm import tqdm

import numpy as np
from dataclasses import dataclass, field

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from common.paths import PROCESSED, PREDICTIONS
from common.logging import setup_logger
from common.json_helpers import load_json, save_json
from shared.span_model import SpanClassifier, generate_candidates, deduplicate_predictions, spans_overlap


@dataclass
class Config:
    model_dir: Path
    val_path: Path =  PROCESSED / "val_data.json"
    output_dir: Path = PREDICTIONS

    def __post_init__(self):
        self.model_dir = Path(self.model_dir)
        self.run_dir = self.model_dir.parent 
        self.run_name = self.run_dir.name
        self.device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        self.logger = setup_logger(f"extract.span.{self.run_name}", f"extract_span_{self.run_name}.log")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._set_model_params()

        self.correct_leading_whitespace_offset = 'deberta' in self.base_model.lower()
    
    def _set_model_params(self):
        config_path = self.run_dir / 'config.json'
        run_config = load_json(config_path, self.logger)

        self.base_model = run_config['base_model']
        self.max_length = run_config['max_length']
        self.max_candidate_size = run_config['max_candidate_size']


def correct_offset(text: str, char_start: int, char_end: int) -> tuple:
    """Remove leading whitespace from predicted spans."""
    span_text = text[char_start:char_end]
    n_stripped = len(span_text) - len(span_text.lstrip())
    return char_start + n_stripped, char_end


def run_inference(text, model, tokenizer, label_list, config):
    tokenized = tokenizer(
        text, 
        truncation=True, 
        max_length=config.max_length, 
        return_offsets_mapping=True,
        return_tensors="pt" 
    )
    
    offsets = tokenized["offset_mapping"][0].tolist()

    candidates = generate_candidates(offsets, config.max_candidate_size)
    if not candidates:
        return [], [], None

    input_ids = tokenized["input_ids"].to(config.device)
    attention_mask = tokenized["attention_mask"].to(config.device)
    
    candidate_spans = torch.tensor(candidates, dtype=torch.long, device=config.device)

    outputs = model(input_ids, attention_mask, candidate_spans)
    probs = torch.sigmoid(outputs["logits"]).cpu().numpy()

    char_spans = []
    for tok_start, tok_end in candidates:
        char_start = offsets[tok_start][0]
        char_end = offsets[tok_end - 1][1]
        
        if config.correct_leading_whitespace_offset:
            char_start, char_end = correct_offset(text, char_start, char_end)
            
        char_spans.append((char_start, char_end))

    return candidates, char_spans, probs


def apply_threshold(char_spans, probs, label_list, threshold):
    if probs is None or not char_spans:
        return []
        
    if isinstance(threshold, dict):
        thresh_arr = np.array([threshold[label] for label in label_list])
    else:
        thresh_arr = np.full(len(label_list), threshold)
        
    mask = probs >= thresh_arr
    cand_indices, label_indices = np.where(mask)
    
    pred_span_list = [
        (char_spans[c][0], char_spans[c][1], label_list[l], probs[c, l])
        for c, l in zip(cand_indices, label_indices)
    ]
    
    deduped = deduplicate_predictions(pred_span_list, spans_overlap)
    return [(s, e, l, score) for s, e, l, score in deduped]


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_span.py <path/to/final_model>")
        sys.exit(1)

    config = Config(model_dir=Path(sys.argv[1]))
    config.logger.info(
        "Extracting spans at %s on %s (base_model=%s, correct_offsets=%s)",
        config.model_dir, config.device, config.base_model, config.correct_leading_whitespace_offset,
    )

    # Load data
    val_records = load_json(config.val_path, config.logger)
    label_list = load_json(config.model_dir / "label_list.json", config.logger)

    # Load thresholds optimized during evaluation (safely)
    thresholds_path = config.model_dir / "optimized_thresholds.json"
    if thresholds_path.exists():
        optimized_thresholds = load_json(thresholds_path, config.logger)
        config.logger.info("Loaded optimized thresholds: %s", optimized_thresholds)
    else:
        config.logger.warning("optimized_thresholds.json not found. Defaulting to 0.5")
        optimized_thresholds = 0.5

    # Load model & tokenisers
    tokenizer = AutoTokenizer.from_pretrained(config.model_dir)
    model = SpanClassifier(config.base_model, num_labels=len(label_list)).to(config.device)

    model_path_safetensors = config.model_dir / "model.safetensors"
    state_dict = load_file(model_path_safetensors, device=str(config.device))

    load_result = model.load_state_dict(state_dict)
    config.logger.info("Loaded model weights: %s", load_result)
    model.eval()
    
    # Loop through each record
    config.logger.info("Running inference on %d records...", len(val_records))
    
    run_id = uuid.uuid4().hex
    output_records = []

    with torch.inference_mode():
        for record in tqdm(val_records):
            text = record.get("text", "")
            
            _, char_spans, probs = run_inference(text, model, tokenizer, label_list, config)
            spans = apply_threshold(char_spans, probs, label_list, optimized_thresholds)

            needs, persons = [], []
            for span in spans:
                item = {
                    "id": uuid.uuid4().hex,
                    "start": span[0],
                    "end": span[1],
                    "label": span[2],
                    "score": float(span[3]),
                    "text": text[span[0]:span[1]]
                }
                if span[2] in ("person_name", "person_role"):
                    persons.append(item)
                else:
                    needs.append(item)

            record["needs"] = needs
            record["persons"] = persons
            record["run_id"] = run_id
            record["model_version"] = config.run_name
            output_records.append(record)

    # Save out (JSON)
    output_file = config.output_dir / f"spans_{config.run_name}_{config.val_path.stem}.json"
    save_json(path=output_file, data=output_records, logger=config.logger)
    config.logger.info("Extraction complete. Saved to %s", output_file)


if __name__ == "__main__":
    main()
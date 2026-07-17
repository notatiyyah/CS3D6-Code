import sys
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
from tqdm import tqdm

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from common.paths import PROCESSED, METRICS
from common.logging import setup_logger
from common.json_helpers import load_json, save_json
from shared.span_model import SpanClassifier, generate_candidates, deduplicate_predictions, spans_overlap
from eval.evaluators import SpanEvaluator


@dataclass
class Config:
    model_dir: Path
    val_path: Path    = PROCESSED / "val_data.json"
    thresholds: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)

    def __post_init__(self):
        self.model_dir = Path(self.model_dir)
        self.run_dir = self.model_dir.parent 
        self.run_name = self.run_dir.name
        self.device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        self.logger = setup_logger(f"eval.span.{self.run_name}", f"eval_span_{self.run_name}.log")
        self.eval_path: Path = METRICS / f'span.{self.run_name}.json'
        
        # Ensure metrics directory exists
        self.eval_path.parent.mkdir(parents=True, exist_ok=True)
        self._set_model_params()

        # Sentencepiece tokenizers require whitespace correction post-hoc.
        self.correct_leading_whitespace_offset = 'deberta' in self.base_model.lower()
    
    def _set_model_params(self):
        config_path = self.run_dir / 'config.json'
        run_config = load_json(config_path, self.logger)

        self.base_model = run_config['base_model']
        self.max_length = run_config['max_length']
        self.max_candidate_size = run_config['max_candidate_size']



def correct_offset(text: str, char_start: int, char_end: int) -> tuple:
    """Remove leading whitespace from predicted spans to prevent strict span score drops."""
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
    # probs shape: [n_candidates, num_labels + 1], last column is background
    probs = torch.softmax(outputs["logits"], dim=-1).cpu().numpy()

    char_spans = []
    for tok_start, tok_end in candidates:
        char_start = offsets[tok_start][0]
        char_end = offsets[tok_end - 1][1]
        
        if config.correct_leading_whitespace_offset:
            char_start, char_end = correct_offset(text, char_start, char_end)
            
        char_spans.append((char_start, char_end))

    return candidates, char_spans, probs


def apply_threshold(char_spans, probs, label_list, threshold):
    """threshold gates whether we trust the argmax prediction, not per-label
    membership — single-label candidates can only carry one class."""
    if probs is None or not char_spans:
        return []

    background_id = len(label_list)
    pred_classes = np.argmax(probs, axis=-1)          # [n_candidates]
    pred_confs = np.max(probs, axis=-1)                # [n_candidates]

    if isinstance(threshold, dict):
        # per-class threshold, keyed by predicted label; background has no
        # meaningful threshold since we always discard it
        thresh_arr = np.array([threshold.get(label_list[c], 0.5) if c != background_id else 1.0
                                for c in pred_classes])
    else:
        thresh_arr = np.full(len(pred_classes), threshold)

    pred_span_list = [
        (char_spans[i][0], char_spans[i][1], label_list[pred_classes[i]], pred_confs[i])
        for i in range(len(pred_classes))
        if pred_classes[i] != background_id and pred_confs[i] >= thresh_arr[i]
    ]

    deduped = deduplicate_predictions(pred_span_list, spans_overlap)
    return [(s, e, l) for s, e, l, _ in deduped]


def get_class_f1(results_dict, label, match_strategy="strict"):
    """
    Safely extracts the F1 score from the nested evaluator output.
    Adjust `match_strategy` to 'loose', 'strict', or 'iou_0.5' as needed.
    """
    try:
        # Navigates: per_label -> care_care_setting -> strict -> f1
        return results_dict["per_label"][label][match_strategy]["f1"]
    except KeyError:
        return 0.0

def main():
    if len(sys.argv) < 2:
        print("Usage: python eval_span.py <path/to/final_model>")
        sys.exit(1)

    config = Config(model_dir=Path(sys.argv[1]))
    config.logger.info(
        "Evaluating spanclassifier at %s on %s (base_model=%s, correct_offsets=%s)",
        config.model_dir, config.device, config.base_model, config.correct_leading_whitespace_offset,
    )

    val_records = load_json(config.val_path, config.logger)
    label_list = load_json(config.model_dir / "label_list.json", config.logger)

    tokenizer = AutoTokenizer.from_pretrained(config.model_dir)
    model = SpanClassifier(config.base_model, num_labels=len(label_list)).to(config.device)

    model_path_safetensors = config.model_dir / "model.safetensors"
    state_dict = load_file(model_path_safetensors, device=str(config.device))

    load_result = model.load_state_dict(state_dict)
    config.logger.info("Loaded model weights: %s", load_result)
    model.eval()

    evaluator = SpanEvaluator(label_list, config.logger)
    results_by_threshold = {}

    config.logger.info("Running inference on %d records...", len(val_records))
    y_true = []
    y_preds = [] 
    
    # Run inference once
    with torch.inference_mode():
        for record in tqdm(val_records):
            text = record.get("text", "")
            y_true.append([(n["start"], n["end"], n["label"]) for n in record.get("needs", []) + record.get("persons", []) if "label" in n])
            _, char_spans, probs = run_inference(text, model, tokenizer, label_list, config)
            y_preds.append((char_spans, probs))

    # Apply different thresholds and evaluate
    for threshold in config.thresholds:
        config.logger.info("Applying threshold=%s", threshold)
        y_pred = [
            apply_threshold(char_spans, probs, label_list, threshold) if probs is not None else []
            for char_spans, probs in y_preds
        ]

        results = evaluator.evaluate(y_true, y_pred)
        results_by_threshold[threshold] = results
        evaluator.print_report(results, title=f"SPAN METRICS (Threshold: {threshold})")

    config.logger.info("Extracting champion thresholds per class...")
    optimized_thresholds = {}
    
    for label in label_list:
        best_f1 = -1.0
        best_thresh = 0.5
        
        for t in config.thresholds:
            global_results = results_by_threshold[t]
            # Optimise for strict match
            class_f1 = get_class_f1(global_results, label, match_strategy="iou_0.5")
            
            if class_f1 > best_f1:
                best_f1 = class_f1
                best_thresh = t
                
        optimized_thresholds[label] = best_thresh
        config.logger.info("Class Winner | %s -> Threshold: %s (F1: %.4f)", label, best_thresh, best_f1)

    thresholds_path = config.model_dir / "optimized_thresholds.json"
    save_json(path=thresholds_path, data=optimized_thresholds, logger=config.logger)

    config.logger.info("Running final evaluation pass using the optimized threshold map...")
    y_pred_final = [
        apply_threshold(char_spans, probs, label_list, optimized_thresholds)
        for char_spans, probs in y_preds
    ]

    final_results = evaluator.evaluate(y_true, y_pred_final)
    results_by_threshold["optimized_per_class"] = final_results
    evaluator.print_report(final_results, title="SPAN METRICS (Optimized Per-Class Thresholds)")

    save_json(
        path=config.eval_path,
        data={str(t): r for t, r in results_by_threshold.items()},
        logger=config.logger,
    )
    config.logger.info("All eval results successfully saved to %s", config.eval_path)


if __name__ == "__main__":
    main()
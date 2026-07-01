import sys
from dataclasses import dataclass, field
from pathlib import Path

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
    model_dir: Path  # .../needs-span-classifier-v3/{run_name}/final_model
    val_path: Path = field(default_factory=lambda: PROCESSED / "val_data.json")
    max_length: int = 256
    max_candidate_size: int = 20
    thresholds: tuple = (0.5, 0.7, 0.8, 0.9)


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
        self._get_base_model()

        # Sentencepiece tokenizers (DeBERTa, ALBERT, XLNet, etc.) have a preceding 
        # whitespace before a token, which messes up strict span calculations. 
        # This removes it post-hoc.
        self.correct_leading_whitespace_offset = 'deberta' in self.base_model.lower()
    
    def _get_base_model(self):
        config_path = self.run_dir / 'config.json'
        config = load_json(config_path)
        self.base_model = config['base_model']



def correct_offset(text: str, char_start: int, char_end: int) -> tuple:
    """Remove leading whitespace from predicted spans - 
    fixes an issue with Sentencepiece tokenisers making strict span match score drop."""
    span_text = text[char_start:char_end]
    stripped = span_text.lstrip()
    n_stripped = len(span_text) - len(stripped)
    return char_start + n_stripped, char_end


def run_inference(text, model, tokenizer, label_list, config):
    """Run candidate generation + inference for one document.
    Returns (candidates, char_spans, probs) for later threshold application."""
    tokenized = tokenizer(text, truncation=True, max_length=config.max_length, return_offsets_mapping=True)
    offsets = tokenized["offset_mapping"]

    candidates = generate_candidates(offsets, config.max_candidate_size)
    if not candidates:
        return [], [], None

    input_ids = torch.tensor([tokenized["input_ids"]], dtype=torch.long).to(config.device)
    attention_mask = torch.tensor([tokenized["attention_mask"]], dtype=torch.long).to(config.device)
    candidate_spans = torch.tensor(candidates, dtype=torch.long).to(config.device)

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
    """Apply a probability threshold to inference output.
    Returns a list of (start, end, label) predicted spans."""
    pred_span_list = [
        (char_start, char_end, label, probs[cand_idx, label_id])
        for cand_idx, (char_start, char_end) in enumerate(char_spans)
        for label_id, label in enumerate(label_list)
        if probs[cand_idx, label_id] >= threshold
    ]
    
    # Deduplicate predictions of the same label on the same span
    deduped = deduplicate_predictions(pred_span_list, spans_overlap)
    
    # Return only (start, end, label) tuples
    return [(s, e, l) for s, e, l, score in deduped]


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

    tokenizer_path = config.model_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

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
        for record in val_records:
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

    # Save results
    save_json(
        path=config.eval_path,
        data={str(t): r for t, r in results_by_threshold.items()},
        logger=config.logger,
    )
    config.logger.info("Eval results saved to %s", config.eval_path)


if __name__ == "__main__":
    main()
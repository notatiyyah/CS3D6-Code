import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from common.paths import PROCESSED, PREDICTIONS
from common.logging import setup_logger
from common.json_helpers import load_json, save_json
from shared.span_model import SpanClassifier, generate_candidates, deduplicate_predictions
from shared.relation_model import insert_markers
from shared.span_model import spans_overlap


@dataclass
class Config:
    span_model_dir: Path
    relation_model_dir: Path

    # 1. Look for SageMaker data channels, fallback to local PROCESSED dir
    val_dir = Path(os.environ.get("SM_CHANNEL_VAL", PROCESSED))
    val_path: Path = field(default_factory=lambda: Config.val_dir / "val_data.json")

    max_length: int = 256
    max_candidate_size: int = 20
    span_threshold: float = 0.9 # Note: Update based on threshold sweep.
    relation_batch_size: int = 16

    def __post_init__(self):
        # 2. Support passing direct paths or looking for SageMaker default channel injections
        self.span_model_dir = self.span_model_dir
        self.relation_model_dir = self.relation_model_dir

        self.device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        self.run_name = f"{self.span_model_dir.parent.name}_{self.relation_model_dir.parent.name}"

        # 3. Look for SageMaker designated output prediction directory, fallback to local PREDICTIONS
        sm_output_dir = os.environ.get("SM_OUTPUT_DATA_DIR")
        if sm_output_dir:
            self.output_path = Path(sm_output_dir) / f"e2e.{self.run_name}.json"
        else:
            self.output_path = PREDICTIONS / f"e2e.{self.run_name}.json"

        self.logger = setup_logger(f"eval.e2e.{self.run_name}", f"eval_e2e_{self.run_name}.log")
        
        self.logger.info("Using span model: %s", self.span_model_dir)
        self.base_model = load_json(self.span_model_dir.parent / "config.json", self.logger)["base_model"]
        self.correct_leading_whitespace_offset = "deberta" in self.base_model.lower()


def correct_offset(text: str, char_start: int, char_end: int) -> tuple[int, int]:
    span_text = text[char_start:char_end]
    stripped = span_text.lstrip()
    return char_start + (len(span_text) - len(stripped)), char_end


def predict_spans(text, model, tokenizer, labels, config):
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=config.max_length,
        return_offsets_mapping=True,
    )
    offsets = tokenized["offset_mapping"]
    candidates = generate_candidates(offsets, config.max_candidate_size)

    if not candidates:
        return []

    inputs = {
        "input_ids": torch.tensor([tokenized["input_ids"]]).to(config.device),
        "attention_mask": torch.tensor([tokenized["attention_mask"]]).to(config.device),
        "candidate_spans": torch.tensor(candidates).to(config.device),
    }

    with torch.inference_mode():
        probs = torch.sigmoid(model(**inputs)["logits"]).cpu().numpy()

    preds = []
    for idx, (start, end) in enumerate(candidates):
        for label_id, label in enumerate(labels):
            if probs[idx, label_id] >= config.span_threshold:
                char_start = offsets[start][0]
                char_end = offsets[end - 1][1]
                if config.correct_leading_whitespace_offset:
                    char_start, char_end = correct_offset(text, char_start, char_end)
                preds.append({
                    "start": char_start,
                    "end": char_end,
                    "label": label,
                    "score": float(probs[idx, label_id]),
                })

    return deduplicate_predictions(
        [(p["start"], p["end"], p["label"], p["score"]) for p in preds],
        spans_overlap,
    )


def predict_relations(doc, model, tokenizer, config):
    text = doc["text"]
    needs = doc["needs"]
    persons = doc["persons"]

    pairs = []
    marked = []
    for need in needs:
        for person in persons:
            pairs.append((need["id"], person["id"]))
            marked.append(insert_markers(text, need, person))

    preds = []
    with torch.inference_mode():
        for i in range(0, len(marked), config.relation_batch_size):
            batch = tokenizer(
                marked[i:i + config.relation_batch_size],
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            ).to(config.device)

            outputs = model(**batch)
            labels = torch.argmax(outputs.logits, dim=1).cpu().tolist()

            for pair, label in zip(pairs[i:i + config.relation_batch_size], labels):
                if label == 1:
                    preds.append(pair)

    return preds


def main():
    if "SM_CHANNEL_SPAN_MODEL" in os.environ and "SM_CHANNEL_RELATION_MODEL" in os.environ:
        span_model_dir = Path(os.environ["SM_CHANNEL_SPAN_MODEL"])
        relation_model_dir = Path(os.environ["SM_CHANNEL_RELATION_MODEL"])
    elif len(sys.argv) >= 3:
        span_model_dir = Path(sys.argv[1])
        relation_model_dir = Path(sys.argv[2])
    else:
        print("Usage: python full_e2e.py <span_model_dir> <relation_model_dir>")
        sys.exit(1)

    config = Config(
        span_model_dir=span_model_dir / "final_model",
        relation_model_dir=relation_model_dir / "final_model",
    )

    records = load_json(config.val_path, config.logger)
    span_labels = load_json(config.span_model_dir / "label_list.json", config.logger)

    span_tokenizer = AutoTokenizer.from_pretrained(config.span_model_dir)
    span_model = SpanClassifier(config.base_model, num_labels=len(span_labels)).to(config.device)
    span_model.load_state_dict(load_file(config.span_model_dir / "model.safetensors", device=str(config.device)))
    span_model.eval()

    relation_tokenizer = AutoTokenizer.from_pretrained(config.relation_model_dir)
    relation_model = AutoModelForSequenceClassification.from_pretrained(
        config.relation_model_dir
    ).to(config.device).eval()

    config.logger.info("Running E2E evaluation on %s", config.device)
    predicted_docs = []
    for doc in tqdm(records):
        text = doc["text"]
        spans = predict_spans(text, span_model, span_tokenizer, span_labels, config)

        needs, persons = [], []
        for idx, span in enumerate(spans):
            item = {
                "id": f"pred_{span[2]}_{idx}",
                "start": span[0],
                "end": span[1],
                "label": span[2],
                "score": span[3],
            }
            if span[2] in ("person_name", "person_role"):
                persons.append(item)
            else:
                needs.append(item)

        pred_doc = {"id": doc["id"], "text": text, "needs": needs, "persons": persons}
        pred_doc["relations"] = predict_relations(pred_doc, relation_model, relation_tokenizer, config)
        predicted_docs.append(pred_doc)

    save_json(path=config.output_path, data=predicted_docs, logger=config.logger)
    config.logger.info("Saved predictions to %s", config.output_path)


if __name__ == "__main__":
    main()
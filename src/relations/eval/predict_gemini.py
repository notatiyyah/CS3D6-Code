"""
Generate relation predictions from Gemini baseline.
Outputs standardized predictions for unified evaluation.
"""
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
from pathlib import Path

from common.paths import PROCESSED, PREDICTIONS, VAL_DATA
from common.logging import setup_logger
from common.json_helpers import load_json, save_json


@dataclass
class Config:
    data_path: Path = VAL_DATA
    model_name: str = "gemini"
    predictions_path: Path = PROCESSED / "gold_standard_gemini_pre_annotated.json"
    output_dir: Path = PREDICTIONS
    output_filename: str = "relation.{model_names}.json"
    person_labels: List[str] = field(default_factory=lambda: ["person_role", "person_name"])

    def __post_init__(self):
        self.logger = setup_logger(
            f"predict.{self.model_name}_relations",
            f"predict_{self.model_name}_relations.log",
        )

def normalize_label(label: str) -> str:
    return str(label).strip().lower()


def span_iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    intersection = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return intersection / union if union > 0 else 0.0


def build_id_map(pred_entities: List[dict], gold_entities: List[dict], iou_threshold: float = 0.5) -> Dict[str, str]:
    """Map Gemini annotation ids onto gold entity ids using label-aware IoU matching."""
    candidates = []

    for pred in pred_entities:
        pred_label = normalize_label(pred["label"])
        for gold in gold_entities:
            if pred_label != normalize_label(gold["label"]):
                continue

            iou = span_iou(pred["start"], pred["end"], gold["start"], gold["end"])
            if iou > 0:
                candidates.append((iou, pred["id"], gold["id"]))

    candidates.sort(key=lambda item: item[0], reverse=True)
    used_pred = set()
    used_gold = set()
    id_map = {}

    for iou, pred_id, gold_id in candidates:
        if pred_id in used_pred or gold_id in used_gold:
            continue
        if iou < iou_threshold:
            continue

        id_map[pred_id] = gold_id
        used_pred.add(pred_id)
        used_gold.add(gold_id)

    return id_map


def load_gemini_annotations(predictions_path: Path, logger) -> Dict[str, dict]:
    """
    Load Gemini annotations from Label Studio format.
    Returns mapping of doc_id -> {'needs': [...], 'persons': [...], 'relations': [...]}.
    """
    predictions = load_json(predictions_path, logger)
    relations_by_doc = {}

    for record in predictions:
        doc_id = record.get("data", {}).get("id")
        if not doc_id or not record.get("predictions"):
            continue

        # Extract spans & relations from annotations
        spans = []
        relations = []
        try:
            annotations = record["predictions"][0].get("result", [])
            for ann in annotations:
                if ann.get("type") == "labels":
                    value = ann.get("value", {})
                    labels = value.get("labels") or []
                    if not labels:
                        continue
                    spans.append(
                        {
                            "id": ann.get("id"),
                            "start": value.get("start"),
                            "end": value.get("end"),
                            "label": labels[0],
                        }
                    )
                elif ann.get("type") == "relation":
                    from_id = ann.get("from_id")
                    to_id = ann.get("to_id")
                    if from_id and to_id:
                        relations.append((str(from_id).strip(), str(to_id).strip()))
        except (KeyError, IndexError, TypeError):
            pass

        relations_by_doc[doc_id] = {"spans": spans, "relations": relations}

    logger.info(f"Loaded relations for {len(relations_by_doc)} documents from Gemini")
    return relations_by_doc


def predict_relations(data: List[dict], gemini_annotations: Dict[str, dict], config: Config,) -> Dict[str, dict]:
    """Generate relation predictions in standardized format from Gemini."""
    predictions = []

    for doc in data:
        doc_id = doc["id"]
        text = doc["text"]
        needs = doc.get("needs", [])
        persons = doc.get("persons", [])

        annotation_data = gemini_annotations.get(doc_id, {"spans": [], "relations": []})
        id_map = build_id_map(annotation_data["spans"], needs + persons)

        # Translate Gemini relation annotations onto the real gold entity ids.
        doc_relations = []

        for from_id, to_id in annotation_data["relations"]:
            mapped_from = id_map.get(from_id)
            mapped_to = id_map.get(to_id)

            if not mapped_from or not mapped_to:
                continue

            doc_relations.append({
                "from": mapped_from,
                "to": mapped_to,
                "confidence": 1.0
            })

        predictions.append({
            "id": doc_id,
            "text": text,
            "model": doc.get("model", "oracle") + "_" + config.model_name, # Either spanModel_relationModel or oracle_relationModel
            "needs": needs,
            "persons": persons,
            "relations": doc_relations,
            "tenure_ids": doc.get("tenure_ids", []),
            "household_members": doc.get("household_members", []),
        })

    config.logger.info(f"Generated predictions for {len(predictions)} documents")
    return predictions


def main():
    if len(sys.argv) > 1:
        config = Config(sys.argv[1]) # Override val data with other data
    else:
        config = Config()
    config.logger.info("Generating %s relation predictions...", config.model_name)

    # Load data
    data = load_json(config.data_path, config.logger)
    gemini_annotations = load_gemini_annotations(config.predictions_path, config.logger)

    # Generate 'predictions' (map relations to GS)
    predictions = predict_relations(data, gemini_annotations, config)
    model_names = predictions[0]['model']
    filename = config.output_filename.format(model_names=model_names)

    # Save predictions
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / filename
    save_json(output_path, predictions, config.logger)
    config.logger.info(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()

"""
Converts Gemini Model Predictions to Annotation Spans.

Ingests pre-annotated data from Gemini (TSV), finds the outputted labels' character spans,
and formats the result to fit the Label Studio JSON structure.
"""

import ast
import json
import re
import uuid
from typing import Any, Dict, List

import pandas as pd

from common.paths import RAW, PROCESSED
from common.logging import setup_logger
from utils.utils import is_valid_json


class Config:
    LOGGER = setup_logger("preprocessing.handle_gemini_annotations","handle_gemini_annotations.log")
    INPUT_PATH = RAW / "annotations" / "gold_standard_gemini_pre_annotated.tsv"
    OUTPUT_PATH = PROCESSED / "gold_standard_gemini_pre_annotated.json"
    MODEL_VERSION = "gemini_pre_annotated"
    PREDICTION_SCORE = 1.0
    PERSON_LABELS = {"Person_Name", "Person_Role", "Person_Pronoun"}


def nearest_span(from_start: int,candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Finds the span closest to a starting character index."""
    preceding = [c for c in candidates if c["start"] <= from_start]
    if preceding:
        return max(preceding, key=lambda c: c["start"])

    return min(candidates, key=lambda c: c["start"])


def process_row(row) -> Dict[str, Any] | None:
    """Converts a single Gemini prediction row into Label Studio format."""
    # Get note data
    note_data = ast.literal_eval(str(row.data))
    note_id = note_data.get("id")
    note_content = str(note_data.get("note_content"))

    # Convert to JSON & handle incorrect json
    if not is_valid_json(row.gemini_predictions):
        config.LOGGER.warning("Invalid JSON prediction for note %s", note_id)
        return None
    gemini_data = json.loads(str(row.gemini_predictions))

    results = []
    span_lookup = {}  # Tracks found spans so we can link relations to the closest one.


    # --- Step A: Process Text Labels (Spans) ---

    for prediction in gemini_data.get("labels", []):
        text = prediction.get("text", "")
        label_name = prediction.get("label", "")
        if not text:
            continue

        key = text.lower()
        existing_count = len(span_lookup.get(key, []))

        # Find the exact words from the output (can be multiple)
        matches = list(re.finditer(r"\b" + re.escape(text) + r"\b", note_content, re.IGNORECASE))

        # Find all matches and add (left to right order)
        for match in matches[existing_count:]:
            span_id = str(uuid.uuid4())[:8]

            span_lookup.setdefault(key, []).append({
                "id": span_id,
                "start": match.start(),
                "end": match.end()
            })

            is_person = label_name in config.PERSON_LABELS

            results.append({
                "id": span_id,
                "from_name": "entity_labels" if is_person else "need_labels",
                "to_name": "text",
                "type": "labels",
                "value": {
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(),
                    "labels": [label_name]
                }
            })


    # --- Step B: Process Relations/Links ---

    for link in gemini_data.get("links", []):
        from_key = str(link.get("from", "")).lower()
        to_key = str(link.get("to", "")).lower()

        from_candidates = span_lookup.get(from_key, [])
        to_candidates = span_lookup.get(to_key, [])

        # Ignore broken target relations and ignore links to self
        if not from_candidates or not to_candidates or from_key == to_key:
            config.LOGGER.warning(
                "Could not resolve relation '%s' -> '%s' in note %s",
                from_key,
                to_key,
                note_id
            )
            continue

        # Find closest matching spans (link closest spans together - not perfect, but as good as we can get without character indexes.)
        from_span = nearest_span(len(note_content), from_candidates) # Last occurence
        to_span = nearest_span(from_span['start'], to_candidates)

        results.append({
            "from_id": from_span["id"],
            "to_id": to_span["id"],
            "type": "relation",
            "direction": "right"
        })


    # --- Step C: Structure for Label Studio ---
    return {
        "data": {
            "id": note_id,
            "note_content": note_content
        },
        "predictions": [
            {
                "model_version": config.MODEL_VERSION,
                "score": config.PREDICTION_SCORE,
                "result": results
            }
        ]
    }


def main():
    config = Config()
    config.LOGGER.info("Starting Gemini annotation conversion...")
    
    # 1. Load data from file
    config.LOGGER.info("Loading data from %s...", config.INPUT_PATH)
    df = pd.read_csv(config.INPUT_PATH, sep="\t")

    # 2. Run conversions
    predictions = df.apply(process_row, axis=1).dropna()
    config.LOGGER.info("Processed %s annotations", len(predictions))

    # 3. Save as JSON
    config.LOGGER.info("Saving output to %s",config.OUTPUT_PATH)
    predictions.to_json(config.OUTPUT_PATH, index=False, orient='records', indent=4)

    config.LOGGER.info("Finished successfully")


if __name__ == "__main__":
    main()
"""
Converts Gemini Model Predictions to Annotation Spans
Ingests pre-annotated data from Gemini (TSV), finds the outputted labels' character spans,
and formats them to fit the Label Studio JSON structure.
"""

import ast
import json
import re
from typing import Any, Dict, List
import uuid
import pandas as pd

from utils.utils import is_valid_json

# --- CONSTANTS ---
PRE_ANNOTATIONS_PATH = "../data/input/gold_standard_gemini_pre_annotated.tsv"
OUTPUT_PATH = "../data/output/gold_standard_gemini_pre_annotated.json"

def nearest_span(from_start: int, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Finds the span closest to a starting character index.
    """

    preceding = [c for c in candidates if c['start'] <= from_start]
    if preceding:
        return max(preceding, key=lambda c: c['start'])
    return min(candidates, key=lambda c: c['start'])


# 1. Load Dataset
df = pd.read_csv(PRE_ANNOTATIONS_PATH, sep="\t")
predictions_list = []

# 2. Process Each Row
for row in df.itertuples():
    # Get note data from the stringified column
    note_data = ast.literal_eval(str(row.data))
    note_id = note_data.get("id")
    note_content = str(note_data.get("note_content"))

    # Convert the model's text response to JSON
    if not is_valid_json(row.gemini_predictions):
        print(f"Note with ID: {note_id} has an invalid JSON prediction.")
        continue
    gemini_data = json.loads(str(row.gemini_predictions))
    
    identified_labels = []
    span_lookup = {}  # Tracks found spans so we can link relations to the closest one.

    # --- Step A: Process Text Labels (Spans) ---
    for pred in gemini_data.get("labels", []):
        text = pred.get("text", "")
        label_name = pred.get("label", "")
        if not text:
            continue

        key = text.lower()
        already_found = len(span_lookup.get(key, []))
        
        # Find the exact words from the output (can be multiple)
        all_matches = list(re.finditer(r"\b" + re.escape(text) + r"\b", note_content, re.IGNORECASE))

        # Find all matches and add (left to right order)
        for match in all_matches[already_found:]:
            entity_id = str(uuid.uuid4())[:8]
            is_entity = label_name in ("Person_Name", "Person_Role", "Person_Pronoun")

            span_lookup.setdefault(key, []).append({
                "id": entity_id,
                "start": match.start(),
                "end": match.end()
            })

            identified_labels.append({
                "id": entity_id,
                "from_name": "entity_labels" if is_entity else "need_labels",
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

        # Ignore broken target lookups and ignore links to self
        if not from_candidates or not to_candidates or from_key == to_key:
            print(f"Warning: could not resolve link '{link.get('from', "UNDEFINED")}' -> '{link.get('to', "UNDEFINED")}' in note {note_id}")
            continue

        # Find closest matching spans (link closest spans together - not perfect, but as good as we can get without character indexes.)
        from_span = nearest_span(len(note_content), from_candidates) # Last occurence
        to_span = nearest_span(from_span['start'], to_candidates)

        identified_labels.append({
            "from_id": from_span["id"],
            "to_id": to_span["id"],
            "type": "relation",
            "direction": "right"
        })

    # --- Step C: Structure for Label Studio ---
    predictions_list.append({
        "data": {
            "id": note_id,
            "note_content": note_content,
        },
        "predictions": [{
            "model_version": "gemini_pre_annotated",
            "score": 1.0,
            "result": identified_labels
        }]
    })

# 3. Export to Label Studio JSON format
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(predictions_list, f, indent=4)
"""
Process Fake Data
Get the generated fake data from Gemini (TSV), convert XML tags to JSON lists of needs / entities.
Export as JSON (fake gold standard) and CSV (fake note data - to mock Athena call)
"""

import json
import re
import pandas as pd

from src.utils.utils import is_valid_json

# --- CONSTANTS ---
INPUT_TSV_PATH = "../data/input/generated_fake_data.tsv"
OUTPUT_JSON_PATH = "../data/output/generated_fake_data.json"
OUTPUT_CSV_PATH = "../data/output/generated_fake_data.csv"

parsed_annotated_records = []
parsed_raw_records = []

# 1. Load Generated Data
df = pd.read_csv(INPUT_TSV_PATH, sep="\t")

# Regex for XML tags
tag_pattern = r"<(need|entity)\s+label=['\"]([^'\细]+)['\"]\s+id=['\"]([^'\"]+)['\"]>([\s\S]*?)</\1>"

# 2. Parse XML Tags and get character indexes/offsets
for row in df.itertuples():
    # Remove markdown json wraps (gemini did this for some responses)
    raw_json = str(row.generated_data).replace("``` json", "").replace("```", "")

    if not is_valid_json(raw_json):
        print(f"ERROR: Note {row.id} contains invalid JSON in the generated_data column.")
        continue

    gen_data = json.loads(raw_json)
    
    # Same as in prep_gold_standard notebook.
    full_text = f"[Category: {row.category}] {gen_data['title']} {gen_data['note']}"

    needs = []
    entities = []
    offset = 0

    # Find each XML tag and convert to a json object (same as annotations), append to relevant list
    for match in re.finditer(tag_pattern, full_text):
        tag_type, label, tag_id, inner_text = match.groups()
        start = match.start() - offset
        end = start + len(inner_text)
        offset += len(match.group(0)) - len(inner_text) # Update character offset (since removing XML tags will affect the character indexes)

        item = {"id": tag_id, "start": start, "end": end, "label": label}
        needs.append(item) if tag_type == "need" else entities.append(item)

    clean_full_text = re.sub(tag_pattern, r"\4", full_text)

    # Add to fake gold standard
    parsed_annotated_records.append({
        "id": row.id,
        "text": clean_full_text,
        "needs": needs,
        "entities": entities,
        "relations": gen_data.get("relations", []),
    })

    clean_raw_text = re.sub(tag_pattern, r"\4", gen_data['note'])
    clean_raw_title = re.sub(tag_pattern, r"\4", gen_data['title'])

    # Add to fake raw data (keys match athena column names)
    parsed_raw_records.append({
        "note_id": row.id,
        "note_content": clean_raw_title + clean_raw_text,
        "note_category": row.category
    })

# 3. Export to JSON
with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(parsed_annotated_records, f, indent=2)

# 4. Export to CSV without annotations
pd.DataFrame(parsed_raw_records).to_csv(OUTPUT_CSV_PATH)
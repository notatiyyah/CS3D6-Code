"""
Prepare Fake Data
Extract annotations (Additional Needs labels, labels and relation counts) from gold-standard records. 
These will act as instructions (alongside the actual prompy) for Gemini to generate data.
"""

import json
import re
import pandas as pd

# --- CONSTANTS ---
GOLD_STANDARD_PATH = "../data/output/gold_standard.json"
TAXNOMY_PATH = "../data/output/taxonomy_autogen_v3.csv"
OUTPUT_PATH = "../data/output/stripped_annotated_data.csv"

# 1. Load Data
with open(GOLD_STANDARD_PATH, "r", encoding="utf-8") as f:
    records = json.load(f)

taxonomy = pd.read_csv(TAXNOMY_PATH, index_col="cat_label")

# 2. Strip Text and map AN Taxonomy Definitions (filtered for each record)
stripped_records = []
for r in records:
    # Extract out category
    category_match = re.search(r"\[Category:\s*([^\]]+)\]", r["text"])
    category = category_match.group(1) if category_match else "General"

    # Get need and entity labels
    entities = [e["label"] for e in r["entities"]]
    needs = [n["label"] for n in r["needs"]]

    # Filter taxonomy to just include the ones this note includes.
    taxonomy_strings = []
    for label in set(needs):
        if label in taxonomy.index:
            row = taxonomy.loc[label]
            taxonomy_strings.append({
                "label": label,
                "meaning": row["category_description"],
                "examples": row["values_hint"],
            })

    stripped_records.append({
        "id": r["id"], # Reuse ID from real note (makes it easier to compare)
        "category": category,
        "need_labels": needs,
        "entity_labels": entities,
        "relation_count": len(r["relations"]),
        "note_length": len(r["text"].split()),
        "need_taxonomy": taxonomy_strings,
    })

# 3. Export
pd.DataFrame(stripped_records).to_csv(OUTPUT_PATH, index=False)
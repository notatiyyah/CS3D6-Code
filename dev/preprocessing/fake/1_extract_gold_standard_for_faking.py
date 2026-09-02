"""
Prepare Fake Data
Extract annotations (Additional Needs labels, labels and relation counts) from gold-standard records. 
These will act as instructions (alongside the actual prompt) for Gemini to generate data.
"""

import json
import re
from uuid import uuid4
from dataclasses import dataclass
import pandas as pd

from common.paths import PROCESSED
from common.json_helpers import load_json
from common.logging import setup_logger

# --- CONSTANTS ---
@dataclass
class Config:
    logger                  = setup_logger('preprocessing.extract_gs_for_faking', 'preprocessing.extract_gs_for_faking')
    gold_standard_path: str = PROCESSED / "gold_standard_with_households.json"
    taxonomy_path: str      = PROCESSED / "taxonomy_autogen_v3.csv"
    output_path: str        = PROCESSED / "stripped_annotated_data.csv"

def extract_label(label_list):
    return [
        {
            "label": l['label'],
            "length_words": len(l['text'])
        }
        for l in label_list
    ]

def extract_household_members(record, logger):
    if record.get('household_members') is None:
        logger.warning("NoneType household_members in record %s.", record.get('id', 'unknown'))
        return []
    
    # Try parse json
    if isinstance(record['household_members'], str):
        try:
            raw_members = json.loads(record['household_members'])
        except:
            logger.warning("Invalid household_members in record %s", record.get('id', 'unknown'))
            return []
    else:
        raw_members = record['household_members']

    # Parse fields without PII
    household_members = []
    for person in raw_members:
        household_members.append({
            "id": person['id'],
            "type": person['type'],
            "isResponsible": person['isResponsible'],
            "dateOfBirth": person['dateOfBirth'], # Already stripped in golden standard so not PII
            "personTenureType": person['personTenureType'],
        })
    return household_members

def main():
    config = Config()

    # 1. Load Data
    records = load_json(config.gold_standard_path, config.logger)

    config.logger.info("Extracting taxonomy from %s", config.taxonomy_path)
    taxonomy = pd.read_csv(config.taxonomy_path, index_col="cat_label")

    # 2. Strip Text and extract just span lengths and labels
    config.logger.info("Stripping %s records...", len(records))
    stripped_records = []
    for r in records:
        # Extract out category
        category_match = re.search(r"\[Category:\s*([^\]]+)\]", r["text"])
        category = category_match.group(1) if category_match else "General"

        # Get need and person labels
        persons = extract_label(r['persons'])
        needs = extract_label(r['needs'])

        # Give range (10 words)
        note_length_words = len(r["text"].split())
        lowerBound = note_length_words - (note_length_words % 10)
        upperBound = lowerBound + 10

        stripped_records.append({
            "id": uuid4(),
            "date": r["note_date"],
            "category": category,
            "need_labels": needs,
            "entity_labels": persons,
            "relation_count": len(r["relations"]),
            "note_length_range": f"{lowerBound}-{upperBound}",
            "tenure_count": len(r["tenure_ids"]),
            "household_roster": extract_household_members(r, config.logger),
        })

    # 3. Export
    config.logger.info("Saving stripped records to %s...", config.output_path)
    pd.DataFrame(stripped_records).to_csv(config.output_path, index=False)

if __name__ == "__main__":
    main()
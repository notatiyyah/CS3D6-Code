"""
Process Fake Data
Get the generated fake data from Gemini (TSV), convert XML tags to JSON lists of needs / entities.
Export as JSON (fake gold standard) and CSV (fake note data - to mock Athena call)
"""

import json
import re
from dataclasses import dataclass
import pandas as pd

from common.paths import PROCESSED, RAW
from common.json_helpers import load_json, is_valid_json, save_json
from common.logging import setup_logger

# --- CONSTANTS ---
@dataclass
class Config:
    logger             = setup_logger('preprocessing.process_fake_gs', 'preprocessing.process_fake_gs')
    gemini_output: str = RAW / "generated_fake_data.tsv"
    output_path: str   = PROCESSED / "generated_fake_data.json"

def export_spans_from_tagged_text(row, tag_pattern, logger):
    # Remove markdown json wrapping (gemini did this for about 100 responses) & check if valid
    raw_json = str(row.generated_data).replace("``` json", "").replace("```", "")
    if not is_valid_json(raw_json):
        logger.error("Note %s contains invalid JSON in the generated_data column.", row.id)
        return None
    
    gen_data = json.loads(raw_json)

    # Construct full text - same as in prep_gold_standard notebook.
    full_text = f"[Category: {row.category}] {gen_data['title']} {gen_data['note']}"
    
    needs = []
    persons = []
    offset = 0

    # Find each XML tag and convert to a json object (same as annotations), append to relevant list
    for match in re.finditer(tag_pattern, full_text):
        tag_type, label, tag_id, inner_text = match.groups()
        start = match.start() - offset
        end = start + len(inner_text)
        offset += len(match.group(0)) - len(inner_text) # Update character offset (since removing XML tags will affect the character indexes)

        # Add to relevant list
        item = {"id": tag_id, "text": inner_text, "start": start, "end": end, "label": label}
        needs.append(item) if tag_type == "need" else persons.append(item)

    clean_full_text = re.sub(tag_pattern, r"\4", full_text)

    return pd.Series({
        "id": row.id,
        "text": clean_full_text,
        "needs": needs,
        "persons": persons,
        "relations": gen_data['relations'],
    })


def main():
    config = Config()
    # 1. Load Generated Data
    config.logger.info("Importing gemini generated data from %s", config.gemini_output)
    df = pd.read_csv(config.gemini_output, sep="\t")

    # 2. Parse XML Tags and get character indexes/offsets
    # Regex for XML tags
    tag_pattern = r"<(need|entity)\s+label=['\"]([^'\"]+)['\"]\s+id=['\"]([^'\"]+)['\"]>([\s\S]*?)<\\?/\1>"
    parsed_annotated_records = df.apply(lambda x: export_spans_from_tagged_text(
        row=x, tag_pattern=tag_pattern, logger=config.logger), axis=1)
    
    # 3. Export to JSON
    config.logger.info("Saving %s gemini generated records to %s", len(parsed_annotated_records), config.output_path)
    parsed_annotated_records.to_json(config.output_path, index=False, orient='records', indent=4)

if __name__ == "__main__":
    main()
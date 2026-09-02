"""
Generate span predictions using regex patterns baseline.
Outputs standardized predictions for unified evaluation.
"""
import re
from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path
import uuid

import pandas as pd

from common.paths import PROCESSED, PREDICTIONS, VAL_DATA, TEST_DATA
from common.logging import setup_logger
from common.json_helpers import load_json, save_json


@dataclass
class Config:
    model_name: str = "regex"
    val_path: Path = TEST_DATA
    taxonomy_path: Path = PROCESSED / "taxonomy_autogen_v3.csv"
    person_labels: List[str] = field(default_factory=lambda: ["person_role", "person_name"])

    def __post_init__(self):
        self.logger = setup_logger(
            f"predict.{self.model_name}_spans",
            f"predict_{self.model_name}_spans.log",
        )
        PREDICTIONS.mkdir(parents=True, exist_ok=True)
        self.output_path = PREDICTIONS / f"span.{self.model_name}.json"


def compile_regex_patterns(taxonomy: pd.DataFrame):
    """Compile regex patterns from taxonomy."""
    regexes = {
        row["cat_label"]: re.compile(row["regex"], re.IGNORECASE)
        for _, row in taxonomy.iterrows()
        if pd.notna(row.get("regex"))
    }

    # Manually defined person patterns
    person_role_regex = r"(?i)\b(tenant|tenants|leaseholder|leaseholders|resident|residents|child|children|son|daughter|partner|wife|husband|mother|father|caller|applicant|neighbour|neighbor|neighbours|neighbors)\b"
    regexes["person_role"] = re.compile(person_role_regex, re.IGNORECASE)

    person_name_regex = r"((?:[A-Z]\.\s)?[A-Z][a-z]+\s[A-Z][a-z]+)"
    regexes["person_name"] = re.compile(person_name_regex)

    return regexes


def main():
    config = Config()
    config.logger.info("Generating %s span predictions...", config.model_name)

    val_records = load_json(config.val_path, config.logger)
    taxonomy = pd.read_csv(config.taxonomy_path)
    regex_patterns = compile_regex_patterns(taxonomy)

    # Generate predictions in standardized format
    predictions = []

    for record in val_records:
        text = record.get("text", "")

        # Find all regex matches
        doc_spans = []
        for label, pattern in regex_patterns.items():
            for match in pattern.finditer(text):
                doc_spans.append(
                    {
                        "id": str(uuid.uuid4())[:8],
                        "text": match.group(),
                        "start": match.start(), 
                        "end": match.end(),
                        "label": label,
                        "confidence": 1.0,
                    }
                )
        
        # Split into needs/persons & add record
        prediction = {
            "id": record['id'],
            "text": text,
            "date": record.get("note_date"),
            "model": config.model_name,
            "needs": [x for x in doc_spans if x['label'] not in config.person_labels],
            "persons": [x for x in doc_spans if x['label'] in config.person_labels],
            "tenure_ids": record.get("tenure_ids"),
            "household_members": record.get("household_members")
        }
        predictions.append(prediction)

    # Save out
    save_json(path=config.output_path, data=predictions, logger=config.logger)
    config.logger.info(
        "Generated predictions for %d records. Saved to %s",
        len(predictions),
        config.output_path,
    )


if __name__ == "__main__":
    main()
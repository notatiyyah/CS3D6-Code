import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from common.paths import PROCESSED, METRICS
from common.logging import setup_logger
from common.json_helpers import load_json, save_json
from eval.evaluators import SpanEvaluator


@dataclass
class Config:
    val_path: Path = field(default_factory=lambda: PROCESSED / "val_data.json")
    taxonomy_path: Path = field(default_factory=lambda: PROCESSED / "taxonomy_autogen_v3.csv")
    eval_path: Path = METRICS / 'span_regex.json'
    logger = setup_logger("eval.regex_span_baseline", "eval_regex_span_baseline.log")

def compile_regex_patterns(taxonomy: pd.DataFrame):
    regexes = {
        row["cat_label"]: re.compile(row["regex"], re.IGNORECASE)
        for _, row in taxonomy.iterrows() if pd.notna(row.get("regex"))
    }

    # Copied from prep_gold_standard notebook.
    person_role_regex = r"(?i)\b(tenant|tenants|leaseholder|leaseholders|resident|residents|child|children|son|daughter|partner|wife|husband|mother|father|caller|applicant|neighbour|neighbor|neighbours|neighbor)\b" 
    regexes['person_role'] = re.compile(person_role_regex, re.IGNORECASE)
    
    person_name_regex = r"((?:[A-Z]\.\s)?[A-Z][a-z]+\s[A-Z][a-z]+)"
    regexes['person_name'] = re.compile(person_name_regex)
    return regexes


def main():
    config = Config()
    config.logger.info("Loading validation data and regex taxonomy...")

    val_records = load_json(config.val_path, config.logger)
    taxonomy = pd.read_csv(config.taxonomy_path)
    regex_patterns = compile_regex_patterns(taxonomy)

    all_labels = sorted(regex_patterns.keys())
    evaluator = SpanEvaluator(all_labels, config.logger)

    y_true, y_pred = [], []
    for record in val_records:
        text = record.get("text", "")
        y_true.append([(n["start"], n["end"], n["label"]) for n in record.get("needs", []) + record.get("persons", []) if "label" in n])

        doc_preds = []
        for lbl, pat in regex_patterns.items():
            doc_preds.extend([(m.start(), m.end(), lbl) for m in pat.finditer(text)])
        y_pred.append(doc_preds)

    results = evaluator.evaluate(y_true, y_pred)
    evaluator.print_report(results, "REGEX BASELINE SPAN METRICS")

    save_json(path=config.eval_path, data=results, logger=config.logger)
    config.logger.info("Results saved to %s", config.eval_path)


if __name__ == "__main__":
    main()
"""
Compare evaluation results across multiple models and render them as tables.

This script loads JSON evaluation files and builds pandas DataFrames to compare 
overall metrics (e.g., macro F1, micro F1) and per-label F1 scores side-by-side. 

It handles two common JSON result structures natively:
1. Flat blocks: 
   { "overall": {...}, "per_label": {...} }

2. Threshold sweeps: 
   { "0.5": { "overall": {...}, "per_label": {...} }, "0.6": {...}, ... }
   (If a sweep is detected, the threshold with the highest `macro_f1` is 
   automatically selected for the comparison.)

Usage:
    Edit the `Config.result_sources` list below to point to the specific JSON files you want 
    to compare, then run the script directly: `python compare_models.py`
"""

import sys
import json
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from logging import Logger
from tabulate import tabulate

from common.logging import setup_logger
from common.paths import METRICS
from common.json_helpers import load_json

@dataclass
class ResultSource:
    name: str
    file: Path

@dataclass
class Config:
    mode: str = "iou_0.7" # "loose", "strict" or "iou_0.5" etc

    logger = setup_logger("eval.compare_models", "compare_models.log")
    overall_metrics = ["macro_f1", "micro_f1", "macro_p", "macro_r", "micro_p", "micro_r"]

    # --- Edit this for whichever models/files you want to compare ---
    result_sources = [
        ResultSource("bio_roberta", METRICS / "span_bio_classifier.needs-bio-classifier.json"),
        ResultSource("aws_comprehend", METRICS / "span_comprehend_combined.json"),
        ResultSource("deberta_aws", METRICS / "span.needs-span-classifier.json"),
        ResultSource("regex", METRICS / "span_regex.json"),
    ]


def load_results_set(source: ResultSource, logger: Logger, mode: str) -> dict:
    data = load_json(source.file, logger)

    # Helper to extract the right sub-dictionary based on config.mode
    def extract_results(res_block):
        return (
            res_block["overall"][mode],
            {lbl: stats[mode] for lbl, stats in res_block["per_label"].items() if mode in stats}
        )

    # 1. Base Case: File is a direct block
    if "overall" in data and "per_label" in data:
        overall, per_label_raw = extract_results(data)
        return {"overall": overall, "per_label": per_label_raw}

    # 2. Sweep Case: File contains multiple thresholds
    best_threshold, best_f1, best_res = None, -1.0, None

    # Loop through and choose best threshold
    for threshold, block in data.items():
        if not isinstance(block, dict) or "overall" not in block:
            continue
        try:
            overall, _ = extract_results(block)
        except KeyError:
            continue # Mode doesn't exist in this block
        
        f1 = overall.get("macro_f1", -1.0)
        if f1 > best_f1:
            best_f1, best_threshold, best_res = f1, threshold, block

    if best_res is None:
        raise ValueError(f"No valid data found (mode='{mode}').")

    logger.info("'%s': auto-selected threshold=%s (macro_f1=%.4f)", source.name, best_threshold, best_f1)
    
    overall, per_label_raw = extract_results(best_res)
    return {"overall": overall, "per_label": per_label_raw}

def main():
    config = Config()
    data = {}
    for source in config.result_sources:
        try:
            data[source.name] = load_results_set(source, config.logger, config.mode)
        except Exception as e:
            config.logger.warning("Skipping '%s': %s", source.name, e)

    if not data:
        config.logger.warning("No valid result blocks loaded.")
        sys.exit(1)

    # Build Overall DataFrame
    overall_rows = {name: b["overall"] for name, b in data.items()}
    avail_metrics = [m for m in config.overall_metrics if any(m in b for b in overall_rows.values())]
    overall_df = pd.DataFrame(overall_rows).T[avail_metrics]

    # Build Per-Label DataFrame
    per_label_rows = {name: {lbl: sts.get("f1") for lbl, sts in b["per_label"].items()} for name, b in data.items()}
    per_label_df = pd.DataFrame(per_label_rows)

    # Output cleanly to terminal
    config.logger.info(f"== OVERALL METRICS (mode={config.mode})===")
    config.logger.info("\n" + tabulate(overall_df, headers='keys', tablefmt='psql', floatfmt='.4f'))
    
    config.logger.info(f"=== PER-LABEL F1 (mode={config.mode})===")
    config.logger.info("\n" + tabulate(per_label_df, headers='keys', tablefmt='psql', floatfmt='.4f'))

if __name__ == "__main__":
    main()
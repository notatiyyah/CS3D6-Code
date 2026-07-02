import json
from pathlib import Path
from dataclasses import dataclass

from common.json_helpers import load_json, save_json
from common.paths import PROCESSED, METRICS, PREDICTIONS
from common.logging import setup_logger
from eval.evaluators import SpanEvaluator

@dataclass
class Config:
    gt_span_path: Path            = PROCESSED / "val_data.json"
    preds_model_a_path: Path      =  PREDICTIONS / "comprehend_output_a.jsonl"
    preds_model_b_path: Path      = PREDICTIONS / "comprehend_output_b.jsonl"
    preds_model_person_path: Path = PREDICTIONS / "comprehend_output_persons.jsonl"
    model_name: str               = "comprehend_combined"

    def __post_init__(self):
        self.logger = setup_logger(
            f"eval.{self.model_name}_span",
            f"eval_{self.model_name}_span.log",
        )
        self.eval_path = METRICS / f"span_{self.model_name}.json"


def load_comprehend_jsonl_to_dict(filepath: Path) -> dict:
    """Loads a JSONL file and returns a dictionary keyed by the 'Line' number."""
    preds_dict = {}
    with open(filepath, "r") as f:
        for line in f:
            record = json.loads(line)
            preds_dict[record["Line"]] = record
    return preds_dict

def extract_spans(record):
    label 



def main():
    config = Config()
    
    # 1. Load the gold validation records (to get the true labels)
    val_records = load_json(config.gt_span_path)

    # 2. Load BOTH Comprehend predictions as fast-lookup dictionaries
    config.logger.info("Loading Comprehend predictions...")
    preds_a = load_comprehend_jsonl_to_dict(config.preds_model_a_path)
    preds_b = load_comprehend_jsonl_to_dict(config.preds_model_b_path)
    preds_per = load_comprehend_jsonl_to_dict(config.preds_model_person_path)

    y_true = []
    y_pred = []

    for idx, record in enumerate(val_records):
        # Extract Gold Spans 
        gold_spans = [(n["start"], n["end"], n["label"]) for n in record.get("needs", []) + record.get("persons")]
        y_true.append(gold_spans)

        pred_spans = []
        
        # Look up this document's predictions in Model A
        if idx in preds_a and "Entities" in preds_a[idx]:
            for ent in preds_a[idx]["Entities"]:
                pred_spans.append((ent["BeginOffset"], ent["EndOffset"], ent["Type"].lower())) # lowercase to match format of labels
                
        # Look up this document's predictions in Model B
        if idx in preds_b and "Entities" in preds_b[idx]:
            for ent in preds_b[idx]["Entities"]:
                pred_spans.append((ent["BeginOffset"], ent["EndOffset"], ent["Type"].lower())) # lowercase to match format of labels

        # Look up this document's predictions in Model C (persons)
        if idx in preds_per and "Entities" in preds_per[idx]:
            for ent in preds_per[idx]["Entities"]:
                # Map persons to person names and organisations to role (best match for generic categories)
                if ent["Type"] == "PERSON":
                    ent["Type"] = "person_name"
                elif ent["Type"] == "ORGANIZATION":
                    ent["Type"] = "person_role"
                pred_spans.append((ent["BeginOffset"], ent["EndOffset"], ent["Type"]))
                    
        y_pred.append(pred_spans)

    # 3. Run the Evaluator
    config.logger.info("Running evaluation...")
    
    # Get all unique labels from the ground truth to initialize the evaluator
    all_labels = sorted(list(set([label for doc in y_true for _, _, label in doc])))
    
    evaluator = SpanEvaluator(all_labels, config.logger)
    results = evaluator.evaluate(y_true, y_pred)
    evaluator.print_report(results, title="AWS COMPREHEND METRICS (COMBINED A + B)")

    # Save
    save_json(
        path=config.eval_path,
        data=results,
        logger=config.logger,
    )
    config.logger.info("Results saved to %s", config.eval_path)


if __name__ == "__main__":
    main()
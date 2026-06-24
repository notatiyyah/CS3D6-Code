import json
import re
import pandas as pd
from src.eval.metrics import DocLevelEvaluator

VAL_DATA_PATH = "data/output/val_doc_level.json"
TAXONOMY_PATH = "data/output/taxonomy_autogen_v3.csv"

# Load Data
print("Loading validation data and regex taxonomy...")
with open(VAL_DATA_PATH, "r", encoding="utf-8") as f:
    val_records = json.load(f)

with open("data/output/label_mappings.json", 'r') as f:
    id2label = {int(k): v for k, v in json.load(f)["id2label"].items()}

taxonomy = pd.read_csv(TAXONOMY_PATH)
regex_patterns = {
    row["cat_label"]: re.compile(row["regex"], re.IGNORECASE) 
    for _, row in taxonomy.iterrows() if pd.notna(row.get("regex"))
}

# Initialize Evaluator
evaluator = DocLevelEvaluator(all_labels=list(id2label.values()))
y_true_lists, y_pred_lists = [], []

# Extract Predictions
for record in val_records:
    true_labels = [id2label[i] for i, val in enumerate(record.get("labels", [])) if val == 1]
    pred_labels = [lbl for lbl, pat in regex_patterns.items() if pat.search(record.get("text", ""))]
    
    y_true_lists.append(true_labels)
    y_pred_lists.append(pred_labels)

# Evaluate and Print
results = evaluator.evaluate(y_true_lists, y_pred_lists)
evaluator.print_report(results, title="PER-LABEL REGEX DOCUMENT CLASSIFICATION METRICS")

with open("data/output/regex_doc_baseline_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results saved to data/output/regex_doc_baseline_results.json")
import json
from src.eval.metrics import DocLevelEvaluator

GT_DOC_PATH = "data/output/val_doc_level.json"
GT_SPAN_PATH = "data/output/val_data.json"
GEMINI_DATA_PATH = "data/output/gold_standard_gemini_pre_annotated.json"

print("Loading data...")
with open("data/output/label_mappings.json", 'r') as f:
    id2label = {int(k): v for k, v in json.load(f)["id2label"].items()}

with open(GT_DOC_PATH, "r", encoding="utf-8") as f:
    gt_records = json.load(f)

with open(GT_SPAN_PATH, "r", encoding="utf-8") as f:
    gt_span_lookup = {r["id"]: r for r in json.load(f) if "id" in r}

with open(GEMINI_DATA_PATH, "r", encoding="utf-8") as f:
    gemini_lookup = {
        r.get("data", {}).get("id"): r["predictions"][0].get("result", [])
        for r in json.load(f) if r.get("data", {}).get("id") and r.get("predictions")
    }

evaluator = DocLevelEvaluator(all_labels=list(id2label.values()))
y_true_lists, y_pred_lists = [], []

for gt_record in gt_records:
    doc_id = gt_record.get("id")
    if not doc_id or doc_id not in gt_span_lookup: continue
    
    # Parse true labels from span file
    gt_span_record = gt_span_lookup[doc_id]
    true_labels = list({item["label"] for item in gt_span_record.get("needs", []) + gt_span_record.get("persons", []) if "label" in item})
    
    # Parse Gemini predictions
    gemini_results = gemini_lookup.get(doc_id, [])
    pred_labels = list({
        item["value"]["labels"][0] for item in gemini_results 
        if item.get("type") == "labels" and "labels" in item.get("value", {})
    })
    
    y_true_lists.append(true_labels)
    y_pred_lists.append(pred_labels)

results = evaluator.evaluate(y_true_lists, y_pred_lists)
evaluator.print_report(results, title="GEMINI DOCUMENT-LEVEL CLASSIFICATION METRICS")

with open("data/output/gemini_doc_eval_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results saved to data/output/gemini_doc_eval_results.json")
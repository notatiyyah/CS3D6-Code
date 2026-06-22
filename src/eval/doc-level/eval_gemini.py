"""
Gemini Pre-Annotation Evaluation - Document-Level Only

Evaluates Gemini pre-annotations against ground truth for document-level
classification (does the document contain each label?).

Now using the full 38-label taxonomy.
"""

import json
import numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score

# --- Paths ---
GT_DOC_PATH = "data/output/val_doc_level.json"
GT_SPAN_PATH = "data/output/val_data.json"
GEMINI_DATA_PATH = "data/output/gold_standard_gemini_pre_annotated.json"
LABEL_MAPPINGS_PATH = "data/output/label_mappings.json"

# --- Load data ---
print("Loading data...")

# Load label mappings
with open(LABEL_MAPPINGS_PATH, 'r') as f:
    mappings = json.load(f)
id2label = {int(k): v for k, v in mappings["id2label"].items()}
all_labels = sorted(id2label.values())
print(f"Total labels: {len(all_labels)}")

# Load ground truth (document-level)
with open(GT_DOC_PATH, "r", encoding="utf-8") as f:
    gt_records = json.load(f)
print(f"Loaded {len(gt_records)} ground truth documents")

# Load ground truth (span-level, for extracting document labels from spans)
with open(GT_SPAN_PATH, "r", encoding="utf-8") as f:
    gt_span_records = json.load(f)

# Build span-level ground truth lookup by document ID
gt_span_lookup = {}
for record in gt_span_records:
    doc_id = record.get("id")
    if doc_id:
        gt_span_lookup[doc_id] = record

# Load Gemini predictions
with open(GEMINI_DATA_PATH, "r", encoding="utf-8") as f:
    gemini_records = json.load(f)

# Build Gemini lookup by document ID
gemini_lookup = {}
for record in gemini_records:
    doc_id = record.get("data", {}).get("id")
    if doc_id and "predictions" in record and len(record["predictions"]) > 0:
        gemini_lookup[doc_id] = record["predictions"][0].get("result", [])

print(f"Loaded {len(gemini_lookup)} Gemini-annotated documents")

# --- Document-level metric counters ---
doc_tp = defaultdict(int)
doc_fp = defaultdict(int)
doc_fn = defaultdict(int)

# --- Helper functions ---
def compute_metrics(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1

# --- Process each ground truth document ---
print("\nProcessing documents...")

for gt_record in gt_records:
    doc_id = gt_record.get("id")
    if not doc_id:
        continue
    
    # Get ground truth span data
    gt_span_record = gt_span_lookup.get(doc_id)
    if not gt_span_record:
        continue
    
    # Get Gemini predictions for this document (if they exist)
    gemini_results = gemini_lookup.get(doc_id, [])
    
    # --- Parse ground truth labels from spans ---
    gt_labels = set()
    for item in gt_span_record.get("needs", []) + gt_span_record.get("persons", []):
        if "label" in item:
            gt_labels.add(item["label"])
    
    # --- Parse Gemini predictions (Label Studio format) ---
    pred_labels = set()
    for item in gemini_results:
        if item.get("type") == "labels":
            val = item.get("value", {})
            if "labels" in val and len(val["labels"]) > 0:
                pred_labels.add(val["labels"][0])
    
    # --- Document-level evaluation ---
    for label in gt_labels:
        if label in pred_labels:
            doc_tp[label] += 1
        else:
            doc_fn[label] += 1
    
    for label in pred_labels:
        if label not in gt_labels:
            doc_fp[label] += 1

# ================================================================
# DOCUMENT-LEVEL RESULTS
# ================================================================
print("\n" + "=" * 95)
print("GEMINI DOCUMENT-LEVEL CLASSIFICATION METRICS")
print("=" * 95)
print(f"{'Label':<45} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'TP':<5} | {'FP':<5} | {'FN':<5}")
print("-" * 95)

doc_f1s = []
for label in sorted(all_labels):
    tp = doc_tp[label]
    fp = doc_fp[label]
    fn = doc_fn[label]
    p, r, f1 = compute_metrics(tp, fp, fn)
    doc_f1s.append(f1)
    if tp + fp + fn > 0:
        print(f"{label:<45} | {p:<10.3f} | {r:<10.3f} | {f1:<10.3f} | {tp:<5} | {fp:<5} | {fn:<5}")

# Overall document metrics
total_tp = sum(doc_tp.values())
total_fp = sum(doc_fp.values())
total_fn = sum(doc_fn.values())
doc_macro_f1 = np.mean(doc_f1s) if doc_f1s else 0.0
doc_micro_p, doc_micro_r, doc_micro_f1 = compute_metrics(total_tp, total_fp, total_fn)

print("\n" + "-" * 95)
print(f"{'OVERALL':<45} | {'':<10} | {'':<10} | {'':<10} | {total_tp:<5} | {total_fp:<5} | {total_fn:<5}")
print(f"{'Macro F1':<45} | {'':<10} | {'':<10} | {doc_macro_f1:<10.4f} | {'':<5} | {'':<5} | {'':<5}")
print(f"{'Micro F1':<45} | {'':<10} | {'':<10} | {doc_micro_f1:<10.4f} | {'':<5} | {'':<5} | {'':<5}")
print("=" * 95)

# ================================================================
# COMPARISON TABLE
# ================================================================
print("\n" + "=" * 95)
print("FULL COMPARISON: Document-Level Models")
print("=" * 95)
print(f"{'Model':<35} | {'Macro F1':<12}")
print("-" * 95)
print(f"{'Regex (document-level)':<35} | {0.6228:<12.4f}")
print(f"{'Flat 38-class (DistilBERT)':<35} | {0.2600:<12.4f}")
print(f"{'Hierarchical (parent)':<35} | {0.5977:<12.4f}")
print(f"{'Hierarchical (child)':<35} | {0.4910:<12.4f}")
print(f"{'Gemini (document-level)':<35} | {doc_macro_f1:<12.4f}")
print("=" * 95)

# ================================================================
# SAVE RESULTS
# ================================================================
results = {
    "document": {
        "macro_f1": float(doc_macro_f1),
        "micro_f1": float(doc_micro_f1),
        "micro_precision": float(doc_micro_p),
        "micro_recall": float(doc_micro_r),
        "per_label": {
            label: {
                "precision": float(compute_metrics(doc_tp[label], doc_fp[label], doc_fn[label])[0]),
                "recall": float(compute_metrics(doc_tp[label], doc_fp[label], doc_fn[label])[1]),
                "f1": float(compute_metrics(doc_tp[label], doc_fp[label], doc_fn[label])[2]),
                "tp": int(doc_tp[label]),
                "fp": int(doc_fp[label]),
                "fn": int(doc_fn[label])
            }
            for label in all_labels
        }
    }
}

with open("data/output/gemini_doc_eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to data/output/gemini_doc_eval_results.json")
print("=" * 95)
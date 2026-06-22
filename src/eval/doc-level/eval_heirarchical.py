"""
Hierarchical Document Classifier Evaluation

Evaluates the parent + child classifier pipeline on validation data.
For each document:
1. Parent classifier predicts high-level categories
2. For each predicted parent, child classifier predicts specific needs
3. Metrics computed at parent level, child level, and overall
"""

import json
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

# --- Paths ---
VAL_PATH = "data/output/val_doc_level.json"
PARENT_MODEL_PATH = "data/output/models/parent-classifier/final_model"
CHILD_MODELS_BASE = "data/output/models"
PARENT_THRESHOLD = 0.2  # Lowered from default 0.5
CHILD_THRESHOLD = 0.5

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Device: {DEVICE}")

# --- Parent mapping (from your taxonomy) ---
taxonomy = pd.read_csv('data/output/taxonomy_autogen_v3.csv')
# cat_label -> parent
parent_map = pd.Series(taxonomy.high_level_category.values, index=taxonomy.cat_label).to_dict()

# Reverse mapping
child_to_parent = {child: parent for child, parent in parent_map.items()}

# Parent -> children mapping
parent_children = defaultdict(list)
for child, parent in parent_map.items():
    parent_children[parent].append(child)
for parent in parent_children:
    parent_children[parent] = sorted(parent_children[parent])

# --- Load validation data ---
print(f"Loading validation data from {VAL_PATH}...")
with open(VAL_PATH, "r", encoding="utf-8") as f:
    val_data = json.load(f)

print(f"Validation examples: {len(val_data)}")

# --- Load parent classifier ---
print(f"Loading parent classifier from {PARENT_MODEL_PATH}...")
parent_tokenizer = AutoTokenizer.from_pretrained(PARENT_MODEL_PATH)
parent_model = AutoModelForSequenceClassification.from_pretrained(PARENT_MODEL_PATH).to(DEVICE)
parent_model.eval()

# Get parent label mappings
parent_id2label = parent_model.config.id2label
parent_label2id = parent_model.config.label2id
parent_labels = sorted(parent_label2id.keys())
print(f"Parent labels: {parent_labels}")

# --- Load child classifiers ---
print("Loading child classifiers...")
child_tokenizers = {}
child_models = {}
child_label_mappings = {}

for parent in parent_children:
    if len(parent_children[parent]) < 2:
        print(f"  Skipping {parent}: only {len(parent_children[parent])} child")
        continue
    
    model_path = f"{CHILD_MODELS_BASE}/child-{parent.replace(' ', '_').replace('&', 'and')}/final_model"
    try:
        child_tokenizers[parent] = AutoTokenizer.from_pretrained(model_path)
        child_models[parent] = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE)
        child_models[parent].eval()
        
        # Store label mappings
        child_id2label = child_models[parent].config.id2label
        child_label2id = child_models[parent].config.label2id
        child_label_mappings[parent] = {
            "id2label": child_id2label,
            "label2id": child_label2id,
            "labels": sorted(child_label2id.keys())
        }
        print(f"  Loaded {parent}: {len(child_label_mappings[parent]['labels'])} children")
    except Exception as e:
        print(f"  WARNING: Could not load {parent}: {e}")
        child_models[parent] = None


# --- Evaluation function ---
def predict_parent(text, model, tokenizer, device):
    """Run parent classifier on text"""
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    
    return probs

def predict_child(text, parent, model, tokenizer, device):
    """Run child classifier for a specific parent on text"""
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    
    return probs

def hierarchical_predict(text, parent_model, parent_tokenizer, child_models, child_tokenizers, 
                         child_label_mappings, parent_labels, parent_children, parent_threshold=0.3, child_threshold=0.5):
    """Full hierarchical prediction pipeline"""
    # Step 1: Predict parents
    parent_probs = predict_parent(text, parent_model, parent_tokenizer, DEVICE)
    predicted_parents = []
    for i, prob in enumerate(parent_probs):
        if prob >= parent_threshold:
            parent_name = parent_labels[i]
            predicted_parents.append(parent_name)
    
    # Step 2: For each predicted parent, predict children
    predictions = {}
    for parent in predicted_parents:
        if parent in child_models and child_models[parent] is not None:
            child_probs = predict_child(text, parent, child_models[parent], child_tokenizers[parent], DEVICE)
            child_labels = child_label_mappings[parent]["labels"]
            predicted_children = []
            for i, prob in enumerate(child_probs):
                if prob >= child_threshold:
                    predicted_children.append(child_labels[i])
            predictions[parent] = predicted_children
        else:
            # Single child or no model: parent = child
            if parent in parent_children and len(parent_children[parent]) == 1:
                predictions[parent] = parent_children[parent]
            else:
                predictions[parent] = []
    
    return predictions, parent_probs


# --- Run evaluation ---
print("\nEvaluating hierarchical classifier...")

# Metrics storage
y_true_parents = []
y_pred_parents = []
y_true_children = []
y_pred_children = []

parent_results = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
child_results = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

# For per-example tracking
example_results = []

for idx, item in enumerate(val_data):
    text = item["text"]
    true_labels = item["labels"]  # 38-dim vector
    
    # Get true parent labels
    true_parents = set()
    for child_idx, val in enumerate(true_labels):
        if val == 1:
            child_name = parent_id2label.get(child_idx, None)  # Wait, this is wrong
            # Need to use the original id2label from the data
            pass
    
    # Let's rebuild the true labels properly
    # The val_data has labels as a 38-dim vector using the original id2label
    
    # For now, let's use a simpler approach: track true parents from the label vector
    # We need the original id2label from the data
    
    # Load the original label mappings
    with open("data/output/label_mappings.json", 'r') as f:
        mappings = json.load(f)
    id2label = {int(k): v for k, v in mappings["id2label"].items()}
    
    true_parents = set()
    true_children = []
    for child_idx, val in enumerate(true_labels):
        if val == 1:
            child_name = id2label[child_idx]
            true_children.append(child_name)
            if child_name in child_to_parent:
                true_parents.add(child_to_parent[child_name])
    
    # Get predictions
    pred_parents_dict, parent_probs = hierarchical_predict(
        text, parent_model, parent_tokenizer, child_models, child_tokenizers,
        child_label_mappings, parent_labels, parent_children, PARENT_THRESHOLD, CHILD_THRESHOLD
    )
    pred_parents = set(pred_parents_dict.keys())
    pred_children = []
    for parent, children in pred_parents_dict.items():
        pred_children.extend(children)
    
    # Store for metrics
    y_true_parents.append([1 if p in true_parents else 0 for p in parent_labels])
    y_pred_parents.append([1 if p in pred_parents else 0 for p in parent_labels])
    
    # Child metrics: only for examples with at least one true child
    if true_children:
        # Get all possible child labels
        all_child_labels = sorted(child_to_parent.keys())
        y_true_children.append([1 if c in true_children else 0 for c in all_child_labels])
        y_pred_children.append([1 if c in pred_children else 0 for c in all_child_labels])
    
    # Per-label tracking for parent
    for parent in true_parents:
        if parent in pred_parents:
            parent_results[parent]["tp"] += 1
        else:
            parent_results[parent]["fn"] += 1
    for parent in pred_parents:
        if parent not in true_parents:
            parent_results[parent]["fp"] += 1
    
    # Per-label tracking for child
    for child in true_children:
        if child in pred_children:
            child_results[child]["tp"] += 1
        else:
            child_results[child]["fn"] += 1
    for child in pred_children:
        if child not in true_children:
            child_results[child]["fp"] += 1
    
    example_results.append({
        "text": text[:100] + "...",
        "true_parents": list(true_parents),
        "pred_parents": list(pred_parents),
        "true_children": true_children,
        "pred_children": pred_children
    })

# --- Compute metrics ---
def compute_metrics_from_dict(results):
    """Compute per-label metrics from tp/fp/fn dict"""
    metrics = {}
    for label, counts in results.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        metrics[label] = {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    return metrics

# Parent metrics
parent_metrics = compute_metrics_from_dict(parent_results)
print("\n" + "="*80)
print("PARENT CLASSIFIER METRICS (per-label)")
print("="*80)
for parent in sorted(parent_metrics.keys()):
    m = parent_metrics[parent]
    print(f"{parent:25s} | P: {m['precision']:.3f} | R: {m['recall']:.3f} | F1: {m['f1']:.3f} | TP: {m['tp']:3d} | FP: {m['fp']:3d} | FN: {m['fn']:3d}")

# Overall parent metrics
y_true_parents_arr = np.array(y_true_parents)
y_pred_parents_arr = np.array(y_pred_parents)
parent_f1_macro = f1_score(y_true_parents_arr, y_pred_parents_arr, average="macro", zero_division=0)
parent_f1_micro = f1_score(y_true_parents_arr, y_pred_parents_arr, average="micro", zero_division=0)
parent_precision_macro = precision_score(y_true_parents_arr, y_pred_parents_arr, average="macro", zero_division=0)
parent_recall_macro = recall_score(y_true_parents_arr, y_pred_parents_arr, average="macro", zero_division=0)

print("\n" + "="*80)
print("PARENT CLASSIFIER OVERALL")
print("="*80)
print(f"Macro F1: {parent_f1_macro:.4f}")
print(f"Micro F1: {parent_f1_micro:.4f}")
print(f"Macro Precision: {parent_precision_macro:.4f}")
print(f"Macro Recall: {parent_recall_macro:.4f}")

# Child metrics
child_metrics = compute_metrics_from_dict(child_results)
print("\n" + "="*80)
print("CHILD CLASSIFIER METRICS (per-label)")
print("="*80)
for child in sorted(child_metrics.keys()):
    m = child_metrics[child]
    if m["tp"] + m["fp"] + m["fn"] > 0:
        print(f"{child:45s} | P: {m['precision']:.3f} | R: {m['recall']:.3f} | F1: {m['f1']:.3f} | TP: {m['tp']:3d}")

# Overall child metrics
if y_true_children:
    y_true_children_arr = np.array(y_true_children)
    y_pred_children_arr = np.array(y_pred_children)
    child_f1_macro = f1_score(y_true_children_arr, y_pred_children_arr, average="macro", zero_division=0)
    child_f1_micro = f1_score(y_true_children_arr, y_pred_children_arr, average="micro", zero_division=0)
    child_precision_macro = precision_score(y_true_children_arr, y_pred_children_arr, average="macro", zero_division=0)
    child_recall_macro = recall_score(y_true_children_arr, y_pred_children_arr, average="macro", zero_division=0)
    
    print("\n" + "="*80)
    print("CHILD CLASSIFIER OVERALL")
    print("="*80)
    print(f"Macro F1: {child_f1_macro:.4f}")
    print(f"Micro F1: {child_f1_micro:.4f}")
    print(f"Macro Precision: {child_precision_macro:.4f}")
    print(f"Macro Recall: {child_recall_macro:.4f}")

# --- Compare with flat classifier ---
print("\n" + "="*80)
print("COMPARISON: Hierarchical vs Flat Classifier")
print("="*80)
print(f"{'Metric':30s} | {'Hierarchical':20s} | {'Flat (from earlier)':20s}")
print("-"*80)
print(f"{'Parent Macro F1':30s} | {parent_f1_macro:.4f}          | N/A")
print(f"{'Child Macro F1':30s} | {child_f1_macro:.4f}          | {0.1090:.4f}")
print(f"{'Improvement':30s} | {child_f1_macro - 0.1090:.4f}          | N/A")

# Save results
print("\nSaving results to data/output/hierarchical_eval_results.json...")
results = {
    "parent": {
        "macro_f1": parent_f1_macro,
        "micro_f1": parent_f1_micro,
        "threshold": PARENT_THRESHOLD,
        "per_label": {k: {"precision": v["precision"], "recall": v["recall"], "f1": v["f1"]} 
                     for k, v in parent_metrics.items()}
    },
    "child": {
        "macro_f1": child_f1_macro if y_true_children else None,
        "micro_f1": child_f1_micro if y_true_children else None,
        "threshold": CHILD_THRESHOLD,
        "per_label": {k: {"precision": v["precision"], "recall": v["recall"], "f1": v["f1"]} 
                     for k, v in child_metrics.items()}
    },
    "flat_baseline": 0.1090,
}

with open("data/output/hierarchical_eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Also save some example predictions for qualitative analysis
print("\nSaving example predictions to data/output/hierarchical_examples.json...")
with open("data/output/hierarchical_examples.json", "w") as f:
    json.dump(example_results[:50], f, indent=2)

print("\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)
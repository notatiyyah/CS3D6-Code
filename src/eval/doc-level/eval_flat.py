"""
Evaluate Flat 38-Class Document Classifier (All Children)

Loads the trained DistilBERT model for 38-class multi-label document classification.
Evaluates at multiple thresholds to find optimal cutoff, matching hierarchical
evaluation format for direct comparison.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, EvalPrediction
)
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import defaultdict

# --- Paths ---
MODEL_PATH = "data/output/models/qwen-doc-classifier/final_model"
VAL_PATH = "data/output/val_doc_level.json"
LABEL_MAPPINGS_PATH = "data/output/label_mappings.json"

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Device: {DEVICE}")

# --- Load data ---
print("Loading data...")
with open(VAL_PATH, 'r', encoding='utf-8') as f:
    val_data = json.load(f)

with open(LABEL_MAPPINGS_PATH, 'r') as f:
    mappings = json.load(f)
id2label = {int(k): v for k, v in mappings["id2label"].items()}
label2id = mappings["label2id"]
NUM_LABELS = len(id2label)
all_labels = sorted(id2label.values())  # This is a list of strings

print(f"Validation examples: {len(val_data)}")
print(f"Number of labels: {NUM_LABELS}")
print(f"First few labels: {all_labels[:5]}")

# --- Dataset class ---
class MultiLabelDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(item["labels"], dtype=torch.float)
        }

# --- Load model ---
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

# --- Custom evaluation at multiple thresholds ---
def evaluate_threshold(model, dataset, device, threshold=0.5):
    """Evaluate model at a specific threshold"""
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    all_probs = []
    all_labels_batch = []  # Renamed to avoid conflict
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_probs.extend(probs)
            all_labels_batch.extend(labels.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels_batch = np.array(all_labels_batch)
    
    # Apply threshold
    preds = (all_probs >= threshold).astype(int)
    
    # Compute metrics
    f1_macro = f1_score(all_labels_batch, preds, average="macro", zero_division=0)
    f1_micro = f1_score(all_labels_batch, preds, average="micro", zero_division=0)
    precision_macro = precision_score(all_labels_batch, preds, average="macro", zero_division=0)
    recall_macro = recall_score(all_labels_batch, preds, average="macro", zero_division=0)
    precision_micro = precision_score(all_labels_batch, preds, average="micro", zero_division=0)
    recall_micro = recall_score(all_labels_batch, preds, average="micro", zero_division=0)
    
    # Per-label metrics - using the global all_labels list
    per_label = {}
    for i, label_name in enumerate(all_labels):  # all_labels is the list of strings
        tp = np.sum((all_labels_batch[:, i] == 1) & (preds[:, i] == 1))
        fp = np.sum((all_labels_batch[:, i] == 0) & (preds[:, i] == 1))
        fn = np.sum((all_labels_batch[:, i] == 1) & (preds[:, i] == 0))
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        
        per_label[label_name] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn)
        }
    
    return {
        "threshold": threshold,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "per_label": per_label
    }

# --- Create dataset ---
print("Creating dataset...")
val_dataset = MultiLabelDataset(val_data, tokenizer)

# --- Evaluate at multiple thresholds ---
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
results_by_threshold = {}

print("\n" + "="*95)
print("FLAT 38-CLASS CLASSIFIER EVALUATION AT MULTIPLE THRESHOLDS")
print("="*95)
print(f"{'Threshold':<12} | {'Macro F1':<12} | {'Micro F1':<12} | {'Macro P':<12} | {'Macro R':<12}")
print("-"*95)

for thresh in thresholds:
    result = evaluate_threshold(model, val_dataset, DEVICE, thresh)
    results_by_threshold[thresh] = result
    print(f"{thresh:<12.1f} | {result['f1_macro']:<12.4f} | {result['f1_micro']:<12.4f} | {result['precision_macro']:<12.4f} | {result['recall_macro']:<12.4f}")

# --- Per-label details at best threshold ---
# Find best threshold (by macro F1)
best_thresh = max(thresholds, key=lambda t: results_by_threshold[t]['f1_macro'])
best_results = results_by_threshold[best_thresh]

print("\n" + "="*95)
print(f"PER-LABEL METRICS AT BEST THRESHOLD (threshold={best_thresh})")
print("="*95)
print(f"{'Label':<45} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'TP':<5} | {'FP':<5} | {'FN':<5}")
print("-"*95)

# Sort by F1 descending
sorted_labels = sorted(best_results['per_label'].items(), key=lambda x: x[1]['f1'], reverse=True)

for label, metrics in sorted_labels:
    tp = metrics['tp']
    fp = metrics['fp']
    fn = metrics['fn']
    if tp + fp + fn > 0:
        print(f"{label:<45} | {metrics['precision']:<10.3f} | {metrics['recall']:<10.3f} | {metrics['f1']:<10.3f} | {tp:<5} | {fp:<5} | {fn:<5}")

# --- Summary comparison ---
print("\n" + "="*95)
print("SUMMARY: FLAT 38-CLASS CLASSIFIER")
print("="*95)
print(f"{'Metric':<30} | {'Value'}")
print("-"*95)
print(f"{'Best Threshold':<30} | {best_thresh:.1f}")
print(f"{'Macro F1':<30} | {best_results['f1_macro']:.4f}")
print(f"{'Micro F1':<30} | {best_results['f1_micro']:.4f}")
print(f"{'Macro Precision':<30} | {best_results['precision_macro']:.4f}")
print(f"{'Macro Recall':<30} | {best_results['recall_macro']:.4f}")

# --- Compare with hierarchical ---
print("\n" + "="*95)
print("COMPARISON: Flat vs Hierarchical Classifier")
print("="*95)
print(f"{'Metric':<30} | {'Flat (38-class)':<18} | {'Hierarchical':<18} | {'Δ'}")
print("-"*95)

# Try to load hierarchical results
try:
    with open("data/output/hierarchical_eval_results.json", 'r') as f:
        hierarchical = json.load(f)
    child_macro_f1 = hierarchical.get("child", {}).get("macro_f1", 0)
    parent_macro_f1 = hierarchical.get("parent", {}).get("macro_f1", 0)
    
    print(f"{'Parent Macro F1':<30} | {'N/A':<18} | {parent_macro_f1:<18.4f} | {'N/A'}")
    print(f"{'Child Macro F1':<30} | {best_results['f1_macro']:<18.4f} | {child_macro_f1:<18.4f} | {child_macro_f1 - best_results['f1_macro']:.4f}")
    print(f"{'Improvement':<30} | {'N/A':<18} | {'N/A':<18} | {child_macro_f1 - best_results['f1_macro']:.4f}")
except FileNotFoundError:
    print("Hierarchical results not found. Run hierarchical evaluation first.")
except Exception as e:
    print(f"Error loading hierarchical results: {e}")

# --- Save results ---
output = {
    "model": "flat_38_class_distilbert",
    "thresholds": {
        str(t): {
            "f1_macro": float(r["f1_macro"]),
            "f1_micro": float(r["f1_micro"]),
            "precision_macro": float(r["precision_macro"]),
            "recall_macro": float(r["recall_macro"])
        } for t, r in results_by_threshold.items()
    },
    "best_threshold": best_thresh,
    "best_results": {
        "f1_macro": float(best_results["f1_macro"]),
        "f1_micro": float(best_results["f1_micro"]),
        "precision_macro": float(best_results["precision_macro"]),
        "recall_macro": float(best_results["recall_macro"]),
        "per_label": best_results["per_label"]
    },
    "num_labels": NUM_LABELS,
    "val_examples": len(val_data)
}

with open("data/output/flat_classifier_eval_results.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nResults saved to data/output/flat_classifier_eval_results.json")

# --- Also compare with regex if available ---
try:
    with open("data/output/regex_doc_baseline_results.json", 'r') as f:
        regex_results = json.load(f)
    
    print("\n" + "="*95)
    print("FULL COMPARISON: Regex vs Flat vs Hierarchical")
    print("="*95)
    print(f"{'Model':<25} | {'Macro F1':<12} | {'Threshold'}")
    print("-"*95)
    print(f"{'Regex (document-level)':<25} | {regex_results['overall']['macro_f1']:<12.4f} | {'N/A'}")
    print(f"{'Flat 38-class':<25} | {best_results['f1_macro']:<12.4f} | {best_thresh:.1f}")
    if 'child_macro_f1' in locals():
        print(f"{'Hierarchical (child)':<25} | {child_macro_f1:<12.4f} | {'0.3'}")
        print(f"{'Hierarchical (parent)':<25} | {parent_macro_f1:<12.4f} | {'0.3'}")
except FileNotFoundError:
    print("\nRegex results not found. Run regex document baseline first.")
except Exception as e:
    print(f"\nError loading regex results: {e}")

print("\n" + "="*95)
print("EVALUATION COMPLETE")
print("="*95)
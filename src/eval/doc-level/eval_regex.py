"""
Regex Baseline Evaluation — DOCUMENT LEVEL

Evaluates regex patterns as document-level classifiers.
For each document, regex finds all matches; if any match for a category,
that category is predicted as present in the document.

Metrics: precision, recall, F1 at document level (multi-label).
"""

import json
import re
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

# --- Paths ---
VAL_DATA_PATH = "data/output/val_doc_level.json"
TAXONOMY_PATH = "data/output/taxonomy_autogen_v3.csv"

# --- Load data ---
print("Loading validation data...")
with open(VAL_DATA_PATH, "r", encoding="utf-8") as f:
    val_records = json.load(f)

print(f"Validation examples: {len(val_records)}")

print("Loading taxonomy with regexes...")
taxonomy = pd.read_csv(TAXONOMY_PATH)

# --- Load label mappings ---
with open("data/output/label_mappings.json", 'r') as f:
    mappings = json.load(f)
id2label = {int(k): v for k, v in mappings["id2label"].items()}
NUM_LABELS = len(id2label)

# --- Compile regexes ---
regex_patterns = {}
for _, row in taxonomy.iterrows():
    label = row["cat_label"]
    regex_str = row.get("regex", "")
    if pd.isna(regex_str) or regex_str.strip() == "":
        continue
    try:
        regex_patterns[label] = re.compile(regex_str, re.IGNORECASE)
    except re.error as e:
        print(f"WARNING: Skipping invalid regex for '{label}': {e}")

print(f"Loaded {len(regex_patterns)} valid regex patterns.\n")

# --- Get all labels in validation data ---
all_labels = sorted(id2label.values())
print(f"Total possible labels: {len(all_labels)}")
print(f"Labels with regex patterns: {len(regex_patterns)}")
print(f"Labels without regex patterns: {len(set(all_labels) - set(regex_patterns.keys()))}")

# --- Evaluate ---
y_true = []
y_pred = []

# Per-label metrics storage
label_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

print("Running regex matching...")
for record in val_records:
    text = record.get("text", "")
    true_labels = record.get("labels", [])  # Binary vector
    
    # Get true labels as list of category names
    true_label_names = []
    for idx, val in enumerate(true_labels):
        if val == 1:
            true_label_names.append(id2label[idx])
    
    # Get predicted labels from regex
    pred_label_names = set()
    for label, pattern in regex_patterns.items():
        if pattern.search(text):
            pred_label_names.add(label)
    
    # Store for sklearn metrics
    true_vector = [1 if label in true_label_names else 0 for label in all_labels]
    pred_vector = [1 if label in pred_label_names else 0 for label in all_labels]
    y_true.append(true_vector)
    y_pred.append(pred_vector)
    
    # Per-label counts
    for label in set(true_label_names) | pred_label_names:
        if label in true_label_names and label in pred_label_names:
            label_metrics[label]["tp"] += 1
        elif label in true_label_names and label not in pred_label_names:
            label_metrics[label]["fn"] += 1
        elif label not in true_label_names and label in pred_label_names:
            label_metrics[label]["fp"] += 1

# --- Compute metrics ---
y_true_arr = np.array(y_true)
y_pred_arr = np.array(y_pred)

# Overall metrics
f1_macro = f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
f1_micro = f1_score(y_true_arr, y_pred_arr, average="micro", zero_division=0)
precision_macro = precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
recall_macro = recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
precision_micro = precision_score(y_true_arr, y_pred_arr, average="micro", zero_division=0)
recall_micro = recall_score(y_true_arr, y_pred_arr, average="micro", zero_division=0)

# Per-label metrics
print("\n" + "="*95)
print("PER-LABEL REGEX DOCUMENT CLASSIFICATION METRICS")
print("="*95)
print(f"{'Label':<45} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'TP':<5} | {'FP':<5} | {'FN':<5}")
print("-"*95)

per_label_f1 = {}
for label in sorted(all_labels):
    if label in label_metrics:
        m = label_metrics[label]
        tp = m["tp"]
        fp = m["fp"]
        fn = m["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_label_f1[label] = f1
        if tp + fp + fn > 0:
            print(f"{label:<45} | {p:<10.3f} | {r:<10.3f} | {f1:<10.3f} | {tp:<5} | {fp:<5} | {fn:<5}")
    else:
        # No predictions for this label
        per_label_f1[label] = 0.0
        print(f"{label:<45} | {'0.000':<10} | {'0.000':<10} | {'0.000':<10} | {'0':<5} | {'0':<5} | {'?':<5}")

print("\n" + "="*95)
print("OVERALL REGEX DOCUMENT CLASSIFICATION METRICS")
print("="*95)
print(f"{'Metric':<25} | {'Value'}")
print("-"*95)
print(f"{'Macro F1':<25} | {f1_macro:.4f}")
print(f"{'Micro F1':<25} | {f1_micro:.4f}")
print(f"{'Macro Precision':<25} | {precision_macro:.4f}")
print(f"{'Macro Recall':<25} | {recall_macro:.4f}")
print(f"{'Micro Precision':<25} | {precision_micro:.4f}")
print(f"{'Micro Recall':<25} | {recall_micro:.4f}")

# --- Compare with hierarchical classifier ---
print("\n" + "="*95)
print("COMPARISON: Regex vs Hierarchical Classifier")
print("="*95)
print(f"{'Metric':<30} | {'Regex':<15} | {'Hierarchical':<15} | {'Δ'}")
print("-"*95)

# Load hierarchical results if available
hierarchical_results = None
try:
    with open("data/output/hierarchical_eval_results.json", 'r') as f:
        hierarchical_results = json.load(f)
except:
    print("Hierarchical results not found. Run hierarchical evaluation first.")

if hierarchical_results:
    child_f1 = hierarchical_results.get("child", {}).get("macro_f1", 0)
    print(f"{'Macro F1':<30} | {f1_macro:<15.4f} | {child_f1:<15.4f} | {child_f1 - f1_macro:.4f}")
    print(f"{'Micro F1':<30} | {f1_micro:<15.4f} | {hierarchical_results.get('child', {}).get('micro_f1', 0):<15.4f} | {hierarchical_results.get('child', {}).get('micro_f1', 0) - f1_micro:.4f}")

# --- Save results ---
results = {
    "overall": {
        "macro_f1": f1_macro,
        "micro_f1": f1_micro,
        "macro_precision": precision_macro,
        "macro_recall": recall_macro,
        "micro_precision": precision_micro,
        "micro_recall": recall_micro
    },
    "per_label": per_label_f1,
    "num_labels": len(all_labels),
    "num_regex_patterns": len(regex_patterns)
}

with open("data/output/regex_doc_baseline_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to data/output/regex_doc_baseline_results.json")

# --- Also compute by parent category for easier comparison ---
print("\n" + "="*95)
print("REGEX BY PARENT CATEGORY")
print("="*95)

# Load parent mapping
parent_map = {
    "care_care_setting": "Care",
    "care_has_caring_responsibility": "Care",
    "care_social_care_involvement": "Care",
    "cautions_asbo_or_injunction_obtained": "Cautions",
    "cautions_physical_abuse_or_threat_of": "Cautions",
    "cautions_unclean_unsafe_living_environment": "Cautions",
    "cautions_verbal_abuse_or_threat_of": "Cautions",
    "reasonable_adjustments_communication_needs": "Reasonable Adjustments",
    "communication_digital_exclusion": "Communication",
    "communication_fluency_in_english": "Communication",
    "disability_requires_adapted_property": "Disability",
    "disability_sensory": "Disability",
    "health_substance_misuse": "Health",
    "health_breathing_respiratory_problems": "Health",
    "health_care_setting": "Health",
    "health_cognitive_impairment": "Health",
    "health_neurodiversity_learning_disability": "Health",
    "health_medical_condition": "Health",
    "health_mental_health": "Health",
    "health_terminally_ill": "Health",
    "housing_conditions_utilities": "Housing Conditions",
    "housing_conditions_hoarding": "Housing Conditions",
    "life_events_life_events": "Life Events",
    "life_events_temporary": "Life Events",
    "mobility_mobility_physical": "Mobility",
    "property_level_property_adapted": "Property level",
    "property_level_disrepair_damp_mould": "Property level",
    "property_level_infestation": "Property level",
    "safety_risk_antisocial_behaviour": "Safety & Risk",
    "safety_risk_domestic_abuse": "Safety & Risk",
    "safety_risk_firerelated_risks": "Safety & Risk",
    "safety_risk_gas_capped": "Safety & Risk",
    "safety_risk_risk_of_exploitation": "Safety & Risk",
}

parent_children = defaultdict(list)
for child, parent in parent_map.items():
    parent_children[parent].append(child)
for parent in parent_children:
    parent_children[parent] = sorted(parent_children[parent])

# Compute parent-level regex F1 (average of children)
print(f"{'Parent':<25} | {'Avg Child F1':<15} | {'Children with Regex'}")
print("-"*95)

for parent, children in parent_children.items():
    child_f1s = [per_label_f1.get(child, 0) for child in children if child in regex_patterns]
    if child_f1s:
        avg_f1 = np.mean(child_f1s)
        print(f"{parent:<25} | {avg_f1:<15.4f} | {len(child_f1s)}/{len(children)}")
    else:
        print(f"{parent:<25} | {'N/A':<15} | {0}/{len(children)}")

print("="*95)
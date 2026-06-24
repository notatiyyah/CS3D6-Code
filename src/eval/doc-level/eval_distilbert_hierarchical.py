import json
import torch
import pandas as pd
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.eval.metrics import DocLevelEvaluator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_DATA_PATH = "data/output/val_doc_level.json"
TAXONOMY_PATH = "data/output/taxonomy_autogen_v3.csv"
PARENT_MODEL_DIR = "data/models/parent-classifier/final_model"
CHILD_MODELS_BASE_DIR = "data/models"

print(f"Loading data and hierarchical models on {DEVICE}...")
taxonomy_df = pd.read_csv(TAXONOMY_PATH)
child_to_parent_map = pd.Series(taxonomy_df.high_level_category.values, index=taxonomy_df.cat_label).to_dict()
parent_to_children_map = defaultdict(list)
for child, parent in child_to_parent_map.items(): parent_to_children_map[parent].append(child)

with open("data/output/label_mappings.json", "r", encoding="utf-8") as f:
    id2label = {int(k): v for k, v in json.load(f)["id2label"].items()}
with open(VAL_DATA_PATH, "r", encoding="utf-8") as f:
    validation_records = json.load(f)

def extract_threshold(threshold_data, label, default=0.5):
    entry = threshold_data.get(label, default)
    return entry.get("threshold", default) if isinstance(entry, dict) else entry

# Load Parent Model
parent_tokenizer = AutoTokenizer.from_pretrained(PARENT_MODEL_DIR)
parent_model = AutoModelForSequenceClassification.from_pretrained(PARENT_MODEL_DIR).to(DEVICE).eval()
parent_labels_list = sorted(parent_model.config.label2id.keys())
with open(f"{PARENT_MODEL_DIR}/best_thresholds.json", "r") as f:
    parent_thresholds_map = {lbl: extract_threshold(json.load(f), lbl) for lbl in parent_labels_list}

# Load Child Models
child_tokenizers, child_models, child_thresholds_map = {}, {}, {}
for parent_node in parent_to_children_map:
    if len(parent_to_children_map[parent_node]) < 2: continue
    sanitized = parent_node.replace(' ', '_').replace('&', 'and')
    path = f"{CHILD_MODELS_BASE_DIR}/child-{sanitized}/final_model"
    try:
        child_tokenizers[parent_node] = AutoTokenizer.from_pretrained(path)
        child_models[parent_node] = AutoModelForSequenceClassification.from_pretrained(path).to(DEVICE).eval()
        with open(f"{path}/best_thresholds.json", "r") as f:
            raw_t = json.load(f)
            child_thresholds_map[parent_node] = {lbl: extract_threshold(raw_t, lbl) for lbl in child_models[parent_node].config.label2id.keys()}
    except: pass

def extract_probabilities(text, model, tokenizer):
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        return torch.sigmoid(model(**inputs).logits).cpu().numpy()[0]

def run_hierarchical_inference(text):
    p_probs = extract_probabilities(text, parent_model, parent_tokenizer)
    pred_parents = [parent_labels_list[i] for i, prob in enumerate(p_probs) if prob >= parent_thresholds_map.get(parent_labels_list[i], 0.5)]
    
    pred_children = []
    for parent in pred_parents:
        if parent in child_models and child_models[parent]:
            c_probs = extract_probabilities(text, child_models[parent], child_tokenizers[parent])
            local_labels = sorted(child_models[parent].config.label2id.keys())
            pred_children.extend([local_labels[i] for i, p in enumerate(c_probs) if p >= child_thresholds_map[parent].get(local_labels[i], 0.5)])
        else:
            pred_children.extend(parent_to_children_map.get(parent, []))
    return pred_parents, pred_children

# Evaluate
parent_evaluator = DocLevelEvaluator(parent_labels_list)
child_evaluator = DocLevelEvaluator(child_to_parent_map.keys())

y_true_parents, y_pred_parents = [], []
y_true_children, y_pred_children = [], []

print("Running hierarchical inference over validation set...")
for entry in validation_records:
    true_children = [id2label[i] for i, val in enumerate(entry["labels"]) if val == 1]
    true_parents = list({child_to_parent_map[c] for c in true_children if c in child_to_parent_map})
    
    pred_parents, pred_children = run_hierarchical_inference(entry["text"])
    
    y_true_parents.append(true_parents)
    y_pred_parents.append(pred_parents)
    y_true_children.append(true_children)
    y_pred_children.append(pred_children)

# Print and Save
p_results = parent_evaluator.evaluate(y_true_parents, y_pred_parents)
c_results = child_evaluator.evaluate(y_true_children, y_pred_children)

parent_evaluator.print_report(p_results, title="PARENT ROUTING ANALYSIS MATRIX")
child_evaluator.print_report(c_results, title="FINE-GRAINED LEAF CHILD CLASS METRICS")

with open("data/output/hierarchical_eval_results.json", "w") as f:
    json.dump({"parent": p_results, "child": c_results}, f, indent=2)
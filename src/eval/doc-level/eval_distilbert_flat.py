import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.eval.metrics import DocLevelEvaluator

MODEL_PATH = "data/output/models/qwen-doc-classifier/final_model"
VAL_PATH = "data/output/val_doc_level.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class MultiLabelDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(item["text"], truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        return {"input_ids": encoding["input_ids"].squeeze(), "attention_mask": encoding["attention_mask"].squeeze(), "labels": torch.tensor(item["labels"], dtype=torch.float)}

print(f"Loading data and model on {DEVICE}...")
with open(VAL_PATH, 'r', encoding='utf-8') as f:
    val_data = json.load(f)

with open("data/output/label_mappings.json", 'r') as f:
    id2label = {int(k): v for k, v in json.load(f)["id2label"].items()}
all_labels = [id2label[i] for i in range(len(id2label))]

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE).eval()
dataset = MultiLabelDataset(val_data, tokenizer)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# 1. Generate Raw Probabilities ONCE
all_probs, all_true_labels = [], []
with torch.no_grad():
    for batch in dataloader:
        probs = torch.sigmoid(model(input_ids=batch["input_ids"].to(DEVICE), attention_mask=batch["attention_mask"].to(DEVICE)).logits)
        all_probs.extend(probs.cpu().numpy())
        all_true_labels.extend(batch["labels"].cpu().numpy())

all_probs = np.array(all_probs)
all_true_labels = np.array(all_true_labels)

# Convert binary true matrix to list of label strings for the evaluator
y_true_lists = [[all_labels[i] for i, val in enumerate(row) if val == 1] for row in all_true_labels]

evaluator = DocLevelEvaluator(all_labels)
results_by_threshold = {}
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

# 2. Sweep thresholds
for thresh in thresholds:
    # Convert probability matrix to binary, then to list of label strings
    binary_preds = (all_probs >= thresh).astype(int)
    y_pred_lists = [[all_labels[i] for i, val in enumerate(row) if val == 1] for row in binary_preds]
    
    results = evaluator.evaluate(y_true_lists, y_pred_lists)
    results_by_threshold[thresh] = results
    print(f"Threshold {thresh:.1f} | Macro F1: {results['overall']['macro_f1']:.4f}")

# Find best and print report
best_thresh = max(thresholds, key=lambda t: results_by_threshold[t]['overall']['macro_f1'])
evaluator.print_report(results_by_threshold[best_thresh], title=f"FLAT 38-CLASS CLASSIFIER (Best Thresh: {best_thresh})")

with open("data/output/flat_classifier_eval_results.json", "w") as f:
    json.dump({"best_threshold": best_thresh, "results": results_by_threshold[best_thresh]}, f, indent=2)
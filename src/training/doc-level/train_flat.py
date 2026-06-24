import json
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction
from sklearn.metrics import f1_score, precision_score, recall_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "data/output/models/qwen-doc-classifier"

print(f"Loading data and initializing Flat Classifier on {DEVICE}...")

with open("data/output/label_mappings.json", 'r', encoding='utf-8') as f:
    mappings = json.load(f)
label2id = mappings["label2id"]
id2label = {int(k): v for k, v in mappings["id2label"].items()}

with open("data/output/train_doc_level.json", 'r', encoding='utf-8') as f:
    train_data = json.load(f)
with open("data/output/val_doc_level.json", 'r', encoding='utf-8') as f:
    val_data = json.load(f)

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

def compute_metrics(eval_pred: EvalPrediction):
    probs = 1 / (1 + np.exp(-eval_pred.predictions))
    preds = (probs > 0.5).astype(int)
    labels = eval_pred.label_ids
    return {
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "precision_micro": precision_score(labels, preds, average="micro", zero_division=0),
        "recall_micro": recall_score(labels, preds, average="micro", zero_division=0),
    }

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(label2id), problem_type="multi_label_classification", id2label=id2label, label2id=label2id
).to(DEVICE)

args = TrainingArguments(
    output_dir=OUTPUT_DIR, eval_strategy="epoch", save_strategy="epoch",
    learning_rate=2e-5, per_device_train_batch_size=16, per_device_eval_batch_size=32,
    num_train_epochs=10, weight_decay=0.01, load_best_model_at_end=True, metric_for_best_model="f1_macro",
    logging_steps=10, save_total_limit=2, fp16=torch.cuda.is_available()
)

trainer = Trainer(
    model=model, args=args, train_dataset=MultiLabelDataset(train_data, tokenizer),
    eval_dataset=MultiLabelDataset(val_data, tokenizer), compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model(f"{OUTPUT_DIR}/final_model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")
print(f"Model saved to {OUTPUT_DIR}/final_model")
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EvalPrediction
)
from sklearn.metrics import f1_score, precision_score, recall_score
import sys

if __name__ == "__main__":
    # Load your existing data
    with open("data/output/label_mappings.json", 'r', encoding='utf-8') as f:
        mappings = json.load(f)
    
    with open("data/output/train_doc_level.json", 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open("data/output/val_doc_level.json", 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    # Define parent mapping from your table
    taxonomy = pd.read_csv('data/output/taxonomy_autogen_v3.csv')
    # # cat_label -> parent
    parent_map = pd.Series(taxonomy.high_level_category.values, index=taxonomy.cat_label).to_dict()
    
    # Get unique parents, create mappings
    parents = sorted(set(parent_map.values()))
    parent2id = {p: i for i, p in enumerate(parents)}
    id2parent = {i: p for p, i in parent2id.items()}
    NUM_PARENTS = len(parents)
    
    print(f"Parents: {parents}")
    print(f"Number of parents: {NUM_PARENTS}")
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")
    
    # Convert child labels to parent labels
    def convert_to_parents(child_labels, parent_map, parent2id):
        parent_labels = set()
        for child_idx, val in enumerate(child_labels):
            if val == 1:
                # Find the cat_label for this index
                # You need the reverse mapping: id2label from your mappings
                pass  # We'll handle this below
    
    # Actually, let's do this properly
    
    # Load id2label from your mappings
    id2label = {int(k): v for k, v in mappings["id2label"].items()}
    
    def convert_to_parents(child_labels, id2label, parent_map, parent2id):
        """Convert child label vector to parent label vector"""
        parent_labels = [0] * len(parent2id)
        for child_idx, val in enumerate(child_labels):
            if val == 1:
                child_name = id2label[child_idx]
                if child_name in parent_map:
                    parent_name = parent_map[child_name]
                    parent_labels[parent2id[parent_name]] = 1
        return parent_labels
    
    # Convert training data
    parent_train = []
    for item in train_data:
        parent_labels = convert_to_parents(item["labels"], id2label, parent_map, parent2id)
        parent_train.append({
            "text": item["text"],
            "labels": parent_labels
        })
    
    parent_val = []
    for item in val_data:
        parent_labels = convert_to_parents(item["labels"], id2label, parent_map, parent2id)
        parent_val.append({
            "text": item["text"],
            "labels": parent_labels
        })
    
    print(f"Parent train: {len(parent_train)} | Parent val: {len(parent_val)}")
    
    # Check class distribution
    print("\nParent class distribution (train):")
    for parent, idx in parent2id.items():
        count = sum(1 for item in parent_train if item["labels"][idx] == 1)
        print(f"  {parent}: {count}")
    
    # Dataset class
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
    
    def compute_metrics(eval_pred: EvalPrediction):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
        
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs > 0.5).astype(int)
        
        return {
            "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "precision_micro": precision_score(labels, preds, average="micro", zero_division=0),
            "recall_micro": recall_score(labels, preds, average="micro", zero_division=0),
        }
    
    MODEL_NAME = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_dataset = MultiLabelDataset(parent_train, tokenizer)
    val_dataset = MultiLabelDataset(parent_val, tokenizer)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_PARENTS,
        problem_type="multi_label_classification",
        id2label=id2parent,
        label2id=parent2id
    )
    
    training_args = TrainingArguments(
        output_dir="data/output/models/parent-classifier",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=10,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("\nStarting parent-level training...")
    trainer.train()
    
    print("\nFinal Evaluation:")
    results = trainer.evaluate()
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    
    trainer.save_model("data/output/models/parent-classifier/final_model")
    tokenizer.save_pretrained("data/output/models/parent-classifier/final_model")
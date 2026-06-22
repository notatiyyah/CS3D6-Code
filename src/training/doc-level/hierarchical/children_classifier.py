import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EvalPrediction
)
from sklearn.metrics import f1_score, precision_score, recall_score
import os
from collections import defaultdict

if __name__ == "__main__":
    # Load data
    with open("data/output/label_mappings.json", 'r', encoding='utf-8') as f:
        mappings = json.load(f)
    id2label = {int(k): v for k, v in mappings["id2label"].items()}
    
    with open("data/output/train_doc_level.json", 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open("data/output/val_doc_level.json", 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    # Parent mapping from your taxonomy
    taxonomy = pd.read_csv('data/output/taxonomy_autogen_v3.csv')
    # # cat_label -> parent
    parent_map = pd.Series(taxonomy.high_level_category.values, index=taxonomy.cat_label).to_dict()
    
    # Reverse mapping: child_name -> parent
    child_to_parent = {}
    for child_name, parent in parent_map.items():
        child_to_parent[child_name] = parent
    
    # Build parent -> children mapping
    parent_children = defaultdict(list)
    for child_name, parent in parent_map.items():
        parent_children[parent].append(child_name)
    
    # Sort for consistency
    for parent in parent_children:
        parent_children[parent] = sorted(parent_children[parent])
    
    print("Parent -> Children:")
    for parent, children in parent_children.items():
        print(f"  {parent}: {len(children)} children")
    
    # Get child index for each label
    child_to_idx = {v: i for i, v in id2label.items()}
    
    def get_parent_labels(child_labels, id2label, child_to_parent):
        """Get parent labels from child labels"""
        parents = set()
        for idx, val in enumerate(child_labels):
            if val == 1:
                child_name = id2label[idx]
                if child_name in child_to_parent:
                    parents.add(child_to_parent[child_name])
        return parents
    
    def filter_by_parent(data, parent_name, id2label, child_to_parent):
        """Filter data to only examples that have this parent label"""
        filtered = []
        for item in data:
            parents = get_parent_labels(item["labels"], id2label, child_to_parent)
            if parent_name in parents:
                filtered.append(item)
        return filtered
    
    def convert_to_child_labels(item, parent_name, parent_children, child_to_idx):
        """Convert full label vector to child-only vector for this parent"""
        # Get indices of children for this parent
        child_indices = [child_to_idx[child] for child in parent_children[parent_name]]
        # Create child-only label vector
        child_labels = [0] * len(parent_children[parent_name])
        for i, child_idx in enumerate(child_indices):
            if item["labels"][child_idx] == 1:
                child_labels[i] = 1
        return child_labels
    
    # Dataset class for child training
    class ChildDataset(Dataset):
        def __init__(self, data, tokenizer, parent_name, parent_children, child_to_idx, max_length=128):
            self.data = data
            self.tokenizer = tokenizer
            self.parent_name = parent_name
            self.parent_children = parent_children
            self.child_to_idx = child_to_idx
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
            labels = convert_to_child_labels(
                item, self.parent_name, self.parent_children, self.child_to_idx
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": torch.tensor(labels, dtype=torch.float)
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
    
    # Train child classifiers for parents with >= 2 children
    results = {}
    
    for parent_name, children in parent_children.items():
        if len(children) < 2:
            print(f"\nSkipping {parent_name}: only {len(children)} child (parent = child)")
            results[parent_name] = {"skipped": True, "reason": "single_child"}
            continue
        
        print(f"\n{'='*60}")
        print(f"Training child classifier for: {parent_name}")
        print(f"  Children: {children}")
        
        # Filter training and validation data for this parent
        parent_train = filter_by_parent(train_data, parent_name, id2label, child_to_parent)
        parent_val = filter_by_parent(val_data, parent_name, id2label, child_to_parent)
        
        if len(parent_train) == 0:
            print(f"  WARNING: No training examples for {parent_name}, skipping")
            results[parent_name] = {"skipped": True, "reason": "no_training_data"}
            continue
        
        print(f"  Training examples: {len(parent_train)}")
        print(f"  Validation examples: {len(parent_val)}")
        
        # Check class distribution
        child_names = children
        print(f"  Class distribution (train):")
        for i, child in enumerate(child_names):
            count = sum(1 for item in parent_train if item["labels"][child_to_idx[child]] == 1)
            print(f"    {child}: {count}")
        
        # Create datasets
        train_dataset = ChildDataset(
            parent_train, tokenizer, parent_name, parent_children, child_to_idx
        )
        val_dataset = ChildDataset(
            parent_val, tokenizer, parent_name, parent_children, child_to_idx
        )
        
        # Create label mappings for this child model
        child_id2label = {i: child for i, child in enumerate(child_names)}
        child_label2id = {child: i for i, child in enumerate(child_names)}
        
        # Initialize model
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(child_names),
            problem_type="multi_label_classification",
            id2label=child_id2label,
            label2id=child_label2id
        )
        
        # Training args
        output_dir = f"data/output/models/child-{parent_name.replace(' ', '_').replace('&', 'and')}"
        training_args = TrainingArguments(
            output_dir=output_dir,
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
        
        # Train
        print(f"  Training children of {parent_name}...")
        trainer.train()
        
        # Evaluate
        eval_results = trainer.evaluate()
        print(f"  Results for {parent_name}:")
        for k, v in eval_results.items():
            if k.startswith("eval_"):
                print(f"    {k}: {v:.4f}")
        
        # Save
        trainer.save_model(f"{output_dir}/final_model")
        tokenizer.save_pretrained(f"{output_dir}/final_model")
        
        results[parent_name] = {
            "children": children,
            "train_examples": len(parent_train),
            "val_examples": len(parent_val),
            "f1_macro": eval_results.get("eval_f1_macro", 0),
            "f1_micro": eval_results.get("eval_f1_micro", 0),
        }
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Child Classifier Results")
    print("="*60)
    for parent, result in results.items():
        if result.get("skipped"):
            print(f"{parent}: SKIPPED ({result['reason']})")
        else:
            print(f"{parent}: F1-Macro={result['f1_macro']:.4f} (n={result['train_examples']})")
    
    # Save results
    with open("data/output/models/child_classifier_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to data/output/models/child_classifier_results.json")
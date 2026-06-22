import sys
print("Script started", file=sys.stderr)
sys.stderr.flush()

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EvalPrediction
)
from sklearn.metrics import f1_score, precision_score, recall_score

if __name__ == "__main__":
    try:
        print("Loading training data...")
        # Load label mappings
        with open("data/output/label_mappings.json", 'r', encoding='utf-8') as f:
            mappings = json.load(f)
        label2id = mappings["label2id"]
        id2label = {int(k): v for k, v in mappings["id2label"].items()}
        NUM_LABELS = len(label2id)

        # Load converted data
        with open("data/output/train_doc_level.json", 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        with open("data/output/val_doc_level.json", 'r', encoding='utf-8') as f:
            val_data = json.load(f)

        print(f"Train: {len(train_data)} | Val: {len(val_data)} | Labels: {NUM_LABELS}")

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

        # Metrics
        def compute_metrics(eval_pred: EvalPrediction):
            logits = eval_pred.predictions
            labels = eval_pred.label_ids
            
            # Convert logits to probabilities, then to binary predictions
            probs = 1 / (1 + np.exp(-logits))  # sigmoid
            preds = (probs > 0.5).astype(int)
            
            # Micro: overall performance (weights by frequency)
            # Macro: treats all labels equally (important for rare labels)
            return {
                "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
                "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
                "precision_micro": precision_score(labels, preds, average="micro", zero_division=0),
                "recall_micro": recall_score(labels, preds, average="micro", zero_division=0),
            }

        # Setup
        MODEL_NAME = "distilbert-base-uncased"  # Fast, works well for short text
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        train_dataset = MultiLabelDataset(train_data, tokenizer)
        val_dataset = MultiLabelDataset(val_data, tokenizer)

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            problem_type="multi_label_classification",
            id2label=id2label,
            label2id=label2id
        )

        # Training args
        training_args = TrainingArguments(
            output_dir="data/output/models/qwen-doc-classifier",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            num_train_epochs=10,  # More epochs for small data
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            logging_steps=10,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        # Train
        print("\nStarting training...")
        trainer.train()

        # Final evaluation
        print("\nFinal Evaluation:")
        results = trainer.evaluate()
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")

        # Save
        trainer.save_model("data/output/models/qwen-doc-classifier/final_model")
        tokenizer.save_pretrained("data/output/models/qwen-doc-classifier/final_model")
        print("\nModel saved to data/output/models/qwen-doc-classifier/final_model")
    except Exception as e:
        import traceback
        print(f"ERROR: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
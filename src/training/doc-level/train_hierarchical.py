import os, json, torch, random
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

MODEL_BASE_NAME = "roberta-base"
TAXONOMY_PATH = "data/output/taxonomy_autogen_v3.csv"
PARENT_MODEL_DIR = "data/models/parent-classifier"

print(f"Initializing Hierarchical Training Pipeline on {DEVICE}...")

class MultiLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts, self.labels, self.tokenizer, self.max_length = texts, labels, tokenizer, max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0), "labels": torch.tensor(self.labels[idx], dtype=torch.float)}

def compute_sample_weights(labels):
    labels = np.array(labels)
    label_freqs = np.where(labels.sum(axis=0) == 0, 1, labels.sum(axis=0))
    return [max((1.0 / label_freqs[i] for i in np.where(row == 1)[0]), default=1.0 / label_freqs.max()) for row in labels]

class WeightedTrainer(Trainer):
    def __init__(self, *args, sample_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_weights = sample_weights
    def get_train_dataloader(self):
        if self.sample_weights is None: return super().get_train_dataloader()
        gen = torch.Generator().manual_seed(RANDOM_SEED)
        sampler = WeightedRandomSampler(self.sample_weights, num_samples=len(self.sample_weights), replacement=True, generator=gen)
        return DataLoader(self.train_dataset, batch_size=self._train_batch_size, sampler=sampler, collate_fn=self.data_collator)

def optimize_thresholds(y_true, y_prob, label_names):
    y_true, y_prob = np.array(y_true), np.array(y_prob)
    results = {}
    for i, name in enumerate(label_names):
        c_true, c_prob = y_true[:, i], y_prob[:, i]
        best_f1, best_t = max(((f1_score(c_true, (c_prob > t).astype(int), zero_division=0), t) for t in np.arange(0.05, 0.95, 0.05)), key=lambda x: x[0])
        results[name] = {"threshold": float(best_t), "optimized_f1": float(best_f1)}
    return results

def build_compute_metrics_fn(label_names):
    def compute_metrics(eval_pred: EvalPrediction):
        preds = (1 / (1 + np.exp(-eval_pred.predictions)) > 0.5).astype(int)
        return {"f1_macro": f1_score(eval_pred.label_ids, preds, average="macro", zero_division=0)}
    return compute_metrics

# --- 1. Load Data & Taxonomy ---
with open("data/output/train_doc_level.json", 'r') as f: train_data = json.load(f)
with open("data/output/val_doc_level.json", 'r') as f: val_data = json.load(f)

taxonomy_df = pd.read_csv(TAXONOMY_PATH)
parent_map = pd.Series(taxonomy_df.high_level_category.values, index=taxonomy_df.cat_label).to_dict()
unique_parents = sorted(list(set(parent_map.values())))
parent2id = {p: i for i, p in enumerate(unique_parents)}

def map_to_parents(dataset):
    transformed = []
    for item in dataset:
        vec = [0] * len(parent2id)
        for name in item["label_names"]:
            if name in parent_map: vec[parent2id[parent_map[name]]] = 1
        transformed.append({"text": item["text"], "labels": vec})
    return transformed

parent_train = map_to_parents(train_data)
parent_val = map_to_parents(val_data)
tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE_NAME)

# --- 2. Train Parent Classifier ---
print("\n--- Training Parent Classifier ---")
p_model = AutoModelForSequenceClassification.from_pretrained(MODEL_BASE_NAME, num_labels=len(unique_parents), problem_type="multi_label_classification", id2label={i:p for p,i in parent2id.items()}, label2id=parent2id).to(DEVICE)
p_args = TrainingArguments(output_dir=PARENT_MODEL_DIR, eval_strategy="epoch", save_strategy="epoch", learning_rate=2e-5, per_device_train_batch_size=16, num_train_epochs=10, load_best_model_at_end=True, metric_for_best_model="f1_macro")

p_trainer = WeightedTrainer(
    model=p_model, args=p_args, 
    train_dataset=MultiLabelDataset([x["text"] for x in parent_train], [x["labels"] for x in parent_train], tokenizer),
    eval_dataset=MultiLabelDataset([x["text"] for x in parent_val], [x["labels"] for x in parent_val], tokenizer),
    compute_metrics=build_compute_metrics_fn(unique_parents),
    sample_weights=compute_sample_weights([x["labels"] for x in parent_train])
)
p_trainer.train()

# Optimize Parent Thresholds
p_model.eval()
val_texts, val_lbls = [x["text"] for x in parent_val], [x["labels"] for x in parent_val]
with torch.no_grad():
    probs = np.vstack([1 / (1 + np.exp(-p_model(**tokenizer(val_texts[i:i+32], truncation=True, padding="max_length", max_length=128, return_tensors="pt").to(DEVICE)).logits.cpu().numpy())) for i in range(0, len(val_texts), 32)])

os.makedirs(f"{PARENT_MODEL_DIR}/final_model", exist_ok=True)
with open(f"{PARENT_MODEL_DIR}/final_model/best_thresholds.json", "w") as f: json.dump(optimize_thresholds(val_lbls, probs, unique_parents), f, indent=2)
p_trainer.save_model(f"{PARENT_MODEL_DIR}/final_model")
tokenizer.save_pretrained(f"{PARENT_MODEL_DIR}/final_model")

# --- 3. Train Child Classifiers ---
parent_to_children = defaultdict(list)
for child, prnt in parent_map.items(): parent_to_children[prnt].append(child)

for parent_name, children in parent_to_children.items():
    children = sorted(children)
    if len(children) < 2: continue
    
    print(f"\n--- Training Child Classifier: {parent_name} ---")
    sub_train = [x for x, p in zip(train_data, parent_train) if p["labels"][parent2id[parent_name]] == 1]
    sub_val = [x for x, p in zip(val_data, parent_val) if p["labels"][parent2id[parent_name]] == 1]
    if not sub_train: continue

    child2id = {c: i for i, c in enumerate(children)}
    t_labels = [[1 if n in child2id else 0 for n in children] for item in sub_train for n in item["label_names"]] # flattened logic
    t_labels = [[1 if n in item["label_names"] else 0 for n in children] for item in sub_train]
    v_labels = [[1 if n in item["label_names"] else 0 for n in children] for item in sub_val]

    c_model = AutoModelForSequenceClassification.from_pretrained(MODEL_BASE_NAME, num_labels=len(children), problem_type="multi_label_classification", id2label={i:c for c,i in child2id.items()}, label2id=child2id).to(DEVICE)
    c_out = f"data/models/child-{parent_name.replace(' ', '_').replace('&', 'and')}"
    
    c_trainer = WeightedTrainer(
        model=c_model, args=TrainingArguments(output_dir=c_out, eval_strategy="epoch", save_strategy="epoch", learning_rate=2e-5, per_device_train_batch_size=16, num_train_epochs=10, load_best_model_at_end=True, metric_for_best_model="f1_macro"),
        train_dataset=MultiLabelDataset([x["text"] for x in sub_train], t_labels, tokenizer),
        eval_dataset=MultiLabelDataset([x["text"] for x in sub_val], v_labels, tokenizer),
        compute_metrics=build_compute_metrics_fn(children),
        sample_weights=compute_sample_weights(t_labels)
    )
    c_trainer.train()

    # Optimize Child Thresholds
    c_model.eval()
    c_texts = [x["text"] for x in sub_val]
    with torch.no_grad():
        c_probs = np.vstack([1 / (1 + np.exp(-c_model(**tokenizer(c_texts[i:i+32], truncation=True, padding="max_length", max_length=128, return_tensors="pt").to(DEVICE)).logits.cpu().numpy())) for i in range(0, len(c_texts), 32)])

    os.makedirs(f"{c_out}/final_model", exist_ok=True)
    with open(f"{c_out}/final_model/best_thresholds.json", "w") as f: json.dump(optimize_thresholds(v_labels, c_probs, children), f, indent=2)
    c_trainer.save_model(f"{c_out}/final_model")
    tokenizer.save_pretrained(f"{c_out}/final_model")

print("\nPipeline execution sequence complete.")
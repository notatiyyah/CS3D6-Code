import sys
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from common.paths import PROCESSED, METRICS
from common.logging import setup_logger
from common.json_helpers import load_json, save_json
from eval.metrics import DocLevelEvaluator


@dataclass
class Config:
    run_dir: Path              # .../hierarchical/{run_name}
    val_path: Path           = PROCESSED / "val_data_doc_level.json"
    label_mapping_path: Path = PROCESSED / "label_mapping.json"
    taxonomy_path: Path      = PROCESSED / "taxonomy_autogen_v3.csv"
    max_length: int          = 128
    default_threshold: float = 0.5
    device                   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        self.run_dir    = Path(self.run_dir) # From args
        self.parent_dir = self.run_dir / "parent-classifier" / "final_model"
        self.run_name   = self.run_dir.name
        self.eval_path  = METRICS / f"doc_hierarchical.{self.run_name}.json"
        self.logger     = setup_logger(f"eval.hierarchical.{self.run_name}", f"eval_hierarchical_{self.run_name}.log")

    def child_model_dir(self, parent_name: str) -> Path:
        safe_name = parent_name.replace(" ", "_").replace("&", "and")
        return self.run_dir / f"child-{safe_name}" / "final_model"


def extract_threshold(threshold_data, label, default=0.5):
    '''Get best threshold for a given label.'''
    entry = threshold_data.get(label, default)
    return entry.get("threshold", default) if isinstance(entry, dict) else entry

def load_parent_model(config: Config):
    '''Load parent model from config.parent_dir and get thresholds.'''
    tokenizer = AutoTokenizer.from_pretrained(config.parent_dir)
    model = AutoModelForSequenceClassification.from_pretrained(config.parent_dir).to(config.device).eval()
    labels_list = sorted(model.config.label2id.keys())

    raw_thresholds = load_json(config.parent_dir / "best_thresholds.json", config.logger)
    thresholds = {lbl: extract_threshold(raw_thresholds, lbl, config.default_threshold) for lbl in labels_list}

    return tokenizer, model, labels_list, thresholds


def load_child_models(config: Config, parent_to_children_map):
    '''Load children models from config.run_dir and get thresholds for each.'''
    tokenizers, models, thresholds_map = {}, {}, {}
    for parent_node, children in parent_to_children_map.items():
        if len(children) < 2:
            continue # If only one leaf, no child classifier
        path = config.child_model_dir(parent_node)
        if not path.exists():
            config.logger.info("No child model found for '%s' at %s, skipping", parent_node, path)
            continue
        try:
            tokenizers[parent_node] = AutoTokenizer.from_pretrained(path)
            models[parent_node] = AutoModelForSequenceClassification.from_pretrained(path).to(config.device).eval()
            raw_thresholds = load_json(path / "best_thresholds.json", config.logger)
            thresholds_map[parent_node] = {
                lbl: extract_threshold(raw_thresholds, lbl, config.default_threshold)
                for lbl in models[parent_node].config.label2id.keys()
            }
        except Exception as e:
            config.logger.info("Failed to load child model for '%s': %s", parent_node, e)
    return tokenizers, models, thresholds_map


def extract_probabilities(text, model, tokenizer, device, max_length):
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt").to(device)
    with torch.no_grad():
        return torch.sigmoid(model(**inputs).logits).cpu().numpy()[0]


def predict_children_for_parents(
    text, config, active_parents, child_models, child_tokenizers, child_thresholds_map, parent_to_children_map,
):
    """Given a set of 'active' parent categories, run child inference within each active branch and return predicted leaf labels."""
    pred_children = []
    for parent in active_parents:
        if parent in child_models:
            c_probs = extract_probabilities(
                text, child_models[parent], child_tokenizers[parent], config.device, config.max_length
            )
            local_labels = sorted(child_models[parent].config.label2id.keys())
            pred_children.extend([
                local_labels[i] for i, p in enumerate(c_probs)
                if p >= child_thresholds_map[parent].get(local_labels[i], config.default_threshold)
            ])
        else:
            # No trained child model for this parent (e.g. <2 children) — fall back
            # to predicting all of its children whenever the parent is active.
            pred_children.extend(parent_to_children_map.get(parent, []))
    return pred_children


def run_hierarchical_inference(
    text, config, parent_model, parent_tokenizer, parent_labels_list, parent_thresholds_map,
    child_models, child_tokenizers, child_thresholds_map, parent_to_children_map,
):
    '''Predict cascading. Run the text through the parent model and then through the child(ren) model(s).'''
    p_probs = extract_probabilities(text, parent_model, parent_tokenizer, config.device, config.max_length)
    pred_parents = [
        parent_labels_list[i] for i, prob in enumerate(p_probs)
        if prob >= parent_thresholds_map.get(parent_labels_list[i], config.default_threshold)
    ]
    pred_children = predict_children_for_parents(
        text, config, pred_parents, child_models, child_tokenizers, child_thresholds_map, parent_to_children_map,
    )
    return pred_parents, pred_children


def main():
    if len(sys.argv) < 2:
        print("Usage: python eval_hierarchical.py <dir/for/heirarchical/{name}>")
        sys.exit(1)

    config = Config(run_dir=Path(sys.argv[1]))
    config.logger.info("Evaluating hierarchical classifier at %s on %s", config.run_dir, config.device)

    # Load taxonomy and map parents to children and vice versa
    config.logger.info("Loading taxonomy from %s", config.taxonomy_path)
    taxonomy_df = pd.read_csv(config.taxonomy_path)
    child_to_parent_map = pd.Series(taxonomy_df.high_level_category.values, index=taxonomy_df.cat_label).to_dict()
    parent_to_children_map = defaultdict(list)
    for child, parent in child_to_parent_map.items():
        parent_to_children_map[parent].append(child)

    # Map labels
    label2id = load_json(config.label_mapping_path, config.logger)
    id2label = {int(v): k for k, v in label2id.items()}

    # Load validation data
    validation_records = load_json(config.val_path, config.logger)

    # Load parent and children models
    parent_tokenizer, parent_model, parent_labels_list, parent_thresholds_map = load_parent_model(config)
    child_tokenizers, child_models, child_thresholds_map = load_child_models(config, parent_to_children_map)

    # Create evaluation classes
    overall_evaluator = DocLevelEvaluator(list(child_to_parent_map.keys()), config.logger)
    parent_evaluator = DocLevelEvaluator(parent_labels_list, config.logger)
    child_isolated_evaluator = DocLevelEvaluator(list(child_to_parent_map.keys()), config.logger)

    # Evaluate each record sequentially
    y_true_parents, y_pred_parents = [], []
    y_true_children, y_pred_children_chained, y_pred_children_isolated = [], [], []

    config.logger.info("Running hierarchical inference over validation set (%s records)", len(validation_records))
    for entry in validation_records:
        true_children = [id2label[i] for i, val in enumerate(entry["labels"]) if val == 1]
        true_parents = list({child_to_parent_map[c] for c in true_children if c in child_to_parent_map})

        # Evaluate chained -> parent predicts, then goes down to children
        pred_parents, pred_children_chained = run_hierarchical_inference(
            entry["text"], config, parent_model, parent_tokenizer, parent_labels_list, parent_thresholds_map,
            child_models, child_tokenizers, child_thresholds_map, parent_to_children_map,
        )
        
        # Evaluate isolated -> imagine parent was perfect, how do children perform in isolation?
        pred_children_isolated = predict_children_for_parents(
            entry["text"], config, true_parents, child_models, child_tokenizers, child_thresholds_map,
            parent_to_children_map,
        )

        y_true_parents.append(true_parents)
        y_pred_parents.append(pred_parents)
        y_true_children.append(true_children)
        y_pred_children_chained.append(pred_children_chained)
        y_pred_children_isolated.append(pred_children_isolated)

    # Run evaluations. Overall = whole pipeline. p_results and c_results are independent.
    overall_results = overall_evaluator.evaluate(y_true_children, y_pred_children_chained)
    p_results = parent_evaluator.evaluate(y_true_parents, y_pred_parents)
    c_results = child_isolated_evaluator.evaluate(y_true_children, y_pred_children_isolated)

    overall_evaluator.print_report(overall_results, title="OVERALL METRICS (FULL CHAINED PIPELINE)")
    parent_evaluator.print_report(p_results, title="PARENT METRICS")
    child_isolated_evaluator.print_report(c_results, title="CHILD MODEL METRICS (INDEPENDENT)")

    save_json(
        path=config.eval_path,
        data={"overall_chained": overall_results, "parent": p_results, "child_isolated": c_results},
        logger=config.logger,
    )
    config.logger.info("Eval results saved to %s", config.eval_path)


if __name__ == "__main__":
    main()
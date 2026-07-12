import os, sys
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from common.logging import setup_logger
from common.json_helpers import load_json, save_json
from common.paths import PREDICTIONS
from shared.relation_model import insert_markers

class Config:
    output_dir: str = PREDICTIONS
    batch_size: int = 16 # Memory saving for running on a laptop

    def __init__(self, model_dir, input_file):
        self.model_dir = Path(model_dir)
        self.input_file = Path(input_file)

        self.run_dir = self.model_dir.parent 
        self.run_name = self.run_dir.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.logger = setup_logger(f"extract.relations.{self.run_name}", f"extract_relations_{self.run_name}.log")
        self._set_model_params()
    
    def _set_model_params(self):
        config_path = self.run_dir / 'config.json'
        run_config = load_json(config_path, self.logger)

        self.base_model = run_config['base_model']
        self.max_length = run_config.get('max_length', 256)

def predict_relations(doc, model, tokenizer, device, max_length, batch_size=8):
    text = doc.get("text", "")
    needs = doc.get("needs", [])
    persons = doc.get("persons", [])

    if not needs or not persons:
        return []

    pairs, marked_texts = [], []
    for need in needs:
        if need['label'].startswith('property_level'):
            continue # Ignore property level labels. Will never need relations.
        for person in persons:
            pairs.append((need["id"], person["id"]))
            marked_texts.append(insert_markers(text, need, person))

    if not marked_texts:
        return []

    all_labels = []
    with torch.inference_mode():
        # Process in bite-sized chunks to save memory
        for i in range(0, len(marked_texts), batch_size):
            chunk = marked_texts[i : i + batch_size]
            batch = tokenizer(
                chunk,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            logits = model(**batch).logits
            labels = torch.argmax(logits, dim=1).cpu().tolist()
            all_labels.extend(labels)

    return [pair for pair, label in zip(pairs, all_labels) if label == 1]

def main():
    if len(sys.argv) < 3:
        print("Usage: python link_relations.py <relation_model_dir> <span_predictions_file>")
        sys.exit(1)

    cfg = Config(Path(sys.argv[1]), sys.argv[2])
    
    # Set up models
    cfg.logger.info(
        "Extracting relations at %s on %s (base_model=%s)",
        cfg.model_dir, cfg.device, cfg.base_model
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_dir).to(cfg.device)
    model.eval()

    # Get data
    records = load_json(cfg.input_file)

    # Loop through each record
    for doc in tqdm(records):
        doc["relations"] = predict_relations(doc, model, tokenizer, cfg.device, cfg.max_length, cfg.batch_size)

    # Save out (safely)
    out_name = cfg.input_file.name.replace("spans_", f"e2e_predictions_{cfg.run_name}_")
    if not out_name.startswith("e2e_predictions_"):
        out_name = f"e2e_predictions_{cfg.run_name}_{cfg.input_file.name}"
         
    save_json(cfg.output_dir / out_name, records)

if __name__ == "__main__":
    main()
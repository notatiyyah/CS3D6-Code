import json
from collections import defaultdict
import pandas as pd

# All labels from your taxonomy (update if needed)
ALL_LABELS = pd.read_csv('data/output/taxonomy_autogen_v3.csv')['cat_label']


label2id = {label: i for i, label in enumerate(ALL_LABELS)}
id2label = {i: label for i, label in enumerate(ALL_LABELS)}

def convert_to_doc_level(input_file, output_file):
    """Convert Gemini span annotations to document-level multi-label format."""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    converted = []
    label_counts = defaultdict(int)
    
    for record in data:
        text = record["text"]
        
        # Extract all unique labels from the "needs" spans
        labels_present = set()
        for need in record.get("needs", []):
            label = need["label"]
            if label in label2id:
                labels_present.add(label)
                label_counts[label] += 1
        
        # Create multi-hot vector
        multi_hot = [0] * len(ALL_LABELS)
        for label in labels_present:
            multi_hot[label2id[label]] = 1
        
        converted.append({
            "id": record['id'],
            "text": text,
            "labels": multi_hot,
            "label_names": list(labels_present)  # Keep for debugging
        })
    
    # Save converted data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)
    
    # Print stats
    print(f"\n{'='*60}")
    print(f"Converted: {input_file} -> {output_file}")
    print(f"Total records: {len(converted)}")
    print(f"Records with at least 1 label: {sum(1 for r in converted if any(r['labels']))}")
    print(f"Records with NO labels (negative samples): {sum(1 for r in converted if not any(r['labels']))}")
    print(f"\nLabel distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")
    print(f"{'='*60}\n")
    
    return converted

if __name__ == "__main__":
    # Convert both splits
    train_data = convert_to_doc_level("data/output/train_data.json", "data/output/train_doc_level.json")
    val_data = convert_to_doc_level("data/output/val_data.json", "data/output/val_doc_level.json")
    
    # Save label mappings for the trainer
    with open("data/output/label_mappings.json", 'w') as f:
        json.dump({"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}, f, indent=2)
    
    print("✅ Conversion complete. Ready for training.")
import json
from typing import Any, Dict, List
import pandas as pd

def make_binary_label_matrix(raw_data: List[Dict[str, Any]], taxonomy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts raw data (array of records) into a binary multi-label matrix (1 if class exists in note, 0 if does not).
    Based on the provided Additional Needs taxonomy.
    """
    cat_labels = taxonomy_df['cat_label'].tolist()
    rows = []
    
    for record in raw_data:
        # Get unique AN labels in the current row
        unique_labels = set(need.get('label') for need in record.get('needs', []) if 'label' in need)
        
        row = {cat: (1 if cat in unique_labels else 0) for cat in cat_labels}
        rows.append(row)
        
    return pd.DataFrame(rows)

def is_valid_json(text: Any) -> bool:
    """Safely check if a given value is valid JSON syntax."""
    if not isinstance(text, str):
        return False
    try:
        json.loads(text)
        return True
    except (ValueError, TypeError):
        return False
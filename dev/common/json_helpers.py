from typing import Any, Dict
import json
from pathlib import Path
import logging

def is_valid_json(text: Any) -> bool:
    """Safely check if a given value is valid JSON syntax."""
    if not isinstance(text, str):
        return False
    try:
        json.loads(text)
        return True
    except (ValueError, TypeError):
        return False

def save_json(path: Path, data, logger: logging.Logger | None = None,) -> None:
    """Save data to json & log."""
    if logger:
        logger.info("Saving data to %s...", path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=str, ensure_ascii=False)

def load_json(path: Path, logger: logging.Logger | None = None) -> Any:
    """Load data to json & log."""
    if logger:
        logger.info("Loading data from %s...", path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
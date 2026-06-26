from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"

RAW = DATA / "raw"
ANNOTATIONS = RAW / "annotations"

PROCESSED = DATA / "processed"
MODELS = DATA / "models"
RESULTS = DATA / "results"
LOGS = DATA / "logs"
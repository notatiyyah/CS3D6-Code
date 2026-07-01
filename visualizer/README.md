# Visualizer

Side-by-side comparison of ground truth annotations and model predictions. Displays spans, labels, confidence scores, and relations with hover highlighting.

## Setup

```bash
npm install
npm run dev
```

## Data

Copy your data files into `public/data/` before starting:

```
public/data/
├── val_data.json                        # ground truth (val or test)
└── e2e.<span_run>_<relation_run>.json   # predictions from inference/full_e2e.py
```

The prediction filename is generated automatically by `full_e2e.py` from the run names of the two models. These files are not committed — `public/data/` is gitignored.

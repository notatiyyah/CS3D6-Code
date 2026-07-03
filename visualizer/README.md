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
├── val_data.json                                  # ground truth (val or test)
└── e2e.<span_run>_<relation_run>.json             # predictions from inference
└── entity linking.<span_run>_<relation_run>.csv   # household attributions (post-processing)

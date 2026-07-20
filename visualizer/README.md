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
├── test_data.json                          # ground truth (val or test)
└── <relation_model>_<span_model>_.json     # predictions from inference
└── e2e_<relation_model>_<span_model>.csv   # household attributions (post-processing)

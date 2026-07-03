# CS3D6 — Additional Needs NER & Relation Extraction Pipeline

This repository contains the complete data preparation, model training, and evaluation pipeline for a dissertation project investigating automatic extraction of **Additional Needs** (vulnerabilities and risks) from housing case notes provided by the London Borough of Hackney.

The pipeline identifies both the additional need described in the text and the individual to whom it refers, and maps predictions back to real database IDs.

---

## Overview

The pipeline consists of four stages:

### 1. Additional Need & Person Reference Extraction

The repository contains two approaches for span extraction:

| Model | Purpose |
| ----- | ------- |
| **Span Candidate Classifier** | Primary model. Detects both Additional Needs and person references (`person_name`, `person_role`). Because spans are predicted independently, it naturally supports overlapping annotations. |
| **BIO Token Classifier** | Baseline sequence-labelling model. Detects Additional Needs only. BIO tagging cannot represent overlapping entities, so supporting person references would require training a separate model. |

The candidate-based approach was adopted for the main pipeline because it achieved stronger performance while simultaneously extracting the person references required for relation extraction.

> **Why use the candidate-based model?**
>
> Unlike BIO tagging, the candidate-based classifier predicts each span independently, allowing overlapping annotations to be represented naturally. This enables a single model to detect both Additional Needs and person references. In contrast, a BIO-based pipeline would require an additional person-reference model, since overlapping entities cannot be represented within a single BIO tag sequence.

### 2. Relation Extraction

Once person references and Additional Needs have been identified, a marker-based binary classifier links each need to the appropriate individual.

This consistently outperformed simple heuristic approaches such as linking each need to the nearest preceding person.

### 3. Entity Linking

After relation extraction, predicted person references are resolved to real household member IDs from the database using a confidence-score ranker (`src/inference/match_needs_to_tenants.py`).

The ranker operates in three tiers:

| Tier | Signal | Confidence |
| ---- | ------ | ---------- |
| 1 | Exact name match against household roster | 1.00 |
| 2 | Fuzzy name match (Levenshtein-based) | 0.85 |
| 3 | Primary role word (tenant, client, applicant…) | 0.75–0.90 |
| 4 | Family/relational role, narrowed by age and pronoun gender | 0.40 |

Any need that cannot be confidently assigned to a person falls back to the tenure ID(s) for the record. The output is a flat CSV suitable for direct database import.

> **Comparison with AWS Comprehend**
>
> AWS Comprehend Medical-style NER follows a token classification approach and similarly does not support overlapping entity annotations. As a result, person references are excluded from the Comprehend comparison, and only Additional Need extraction is evaluated.

### 4. Output

The final output is a flat CSV with one row per (need, target) assignment:

```
target_id, target_type, need_label, note_id
```

Where `target_type` is either `person` (resolved to a household member ID) or `tenure` (fallback to tenure ID).

---

## Performance Context

A duplicate annotation study was conducted on **88** records to estimate human consistency.

| Metric | Macro F1 |
| ------ | -------- |
| Strict | ~0.62 |
| Loose | ~0.78 |

These values provide an approximate upper bound for expected model performance, as exact span boundaries often contain genuine ambiguity even for human annotators.

---

## Repository Structure

The project follows a standard `src/` layout. All commands should be run from the repository root.

```text
.
├── pyproject.toml
├── requirements.txt
├── data/
│   ├── raw/
│   ├── processed/
│   ├── logs/
│   └── results/
│       ├── metrics/
│       └── predictions/
│
└── src/
    ├── common/
    ├── preprocessing/
    ├── training/
    ├── eval/
    ├── shared/
    └── inference/
```

| Directory | Purpose |
| --------- | ------- |
| `preprocessing/` | Dataset construction and annotation processing |
| `training/` | Span extraction and relation extraction training scripts |
| `eval/` | Evaluation scripts and metrics |
| `common/` | Shared utilities (paths, logging, JSON helpers) |
| `shared/` | Model architecture and inference helpers shared across training, eval, and inference |
| `inference/` | End-to-end inference pipeline and entity linking |

---

## Installation

Create a virtual environment and install the project in editable mode.

```bash
python -m venv venv
source venv/bin/activate

pip install -e .
```

For GPU training on Linux or the Warwick cluster, install the CUDA-enabled version of PyTorch before installing the remaining dependencies:

```bash
pip install torch==2.4.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

pip install transformers==4.46.3 accelerate==0.34.2 \
    scikit-learn==1.5.2 sentencepiece protobuf
```

---

## Data Preparation

The preprocessing pipeline should be executed sequentially.

| Step | Script |
| ---- | ------ |
| 1 | `1_construct_taxonomy.py` |
| 2 | `2_prep_gold_standard.ipynb` |
| 3 | `3_generate_label_studio_ui.py` *(optional)* |
| 4 | `4_handle_gemini_annotations.py` *(optional)* |
| 5 | `5_post_process_annotated_data.ipynb` |
| 6 | `6_train_test_split.py` |
| 7 | `7_prep_data_for_comprehend.py` |

These scripts construct the taxonomy, prepare annotation datasets, process completed annotations, create train/validation/test splits, and generate comparison datasets.

---

## Model Training

### Span Extraction

The primary span extraction model is a candidate-based classifier rather than a traditional BIO tagging model.

Recommended backbone:

```
microsoft/deberta-v3-base
```

### Relation Extraction

The model injects special entity markers around candidate entity spans before classifying whether an additional need belongs to that individual.

---

## Inference

### End-to-end span + relation prediction

```bash
python src/inference/full_e2e.py <span_model_dir> <relation_model_dir>
```

Output is written to `data/results/predictions/e2e.<span_run>_<relation_run>.json`.

### Entity linking

Resolves predicted person references to household member IDs and produces a flat CSV for database import.

```bash
python src/inference/match_needs_to_tenants.py data/processed/val_data.json
```

Output is written to `data/results/predictions/entity_linking_<input_stem>.csv`.

---

## Evaluation

Span evaluation is implemented in `src/eval/evaluators.py`.

Three matching strategies are reported:

| Metric | Description |
| ------ | ----------- |
| Loose | Any overlap between predicted and gold spans |
| Strict | Exact character boundary match |
| IoU | Intersection-over-Union at thresholds 0.3, 0.5, 0.7, and 0.9 |

Relation extraction is evaluated with pair-level exact match via `RelationEvaluator` in the same file.

End-to-end evaluation (`src/eval/eval_e2e.py`) additionally reports:
- Span recall — what proportion of gold spans the span classifier found
- Relation recall given spans found — of gold relations where both spans were detected, how many were linked correctly

This separates span classifier errors from relation model errors.

---

## Visualizer

A lightweight Vite/React app for side-by-side comparison of ground truth and model predictions. (Requires `Node 24+`)

```bash
cd visualizer
npm install
npm run dev
```

Before starting, copy the data files into `visualizer/public/data/` and set the filenames in `visualizer/src/config.js`:

```js
// visualizer/src/config.js
const config = {
  groundTruthFile: "val_data.json",                              // ground truth or test data
  predictionsFile: "e2e.span_model_relation_model.json",         // output of inference/full_e2e.py
  entityLinkingFile: "entity_linking_val_data_with_households.csv", // output of inference/match_needs_to_tenants.py
};
```

The app displays spans, labels, relations, and entity linking assignments side by side with hover highlighting.

---

## Running on the Warwick Batch Compute Cluster

The experiments were designed to run on the University of Warwick Department of Computer Science Batch Compute Cluster.

Full instructions for submitting and monitoring jobs are available in the official documentation:

[https://warwick.ac.uk/fac/sci/dcs/intranet/user_guide/batch_compute/](https://warwick.ac.uk/fac/sci/dcs/intranet/user_guide/batch_compute/)

### Recommended GPUs

The training configuration assumes access to a GPU with **24 GB of VRAM**.

Recommended partitions:

- `falcon`
- `gecko`

These provide NVIDIA A10/A5000 GPUs and support the configured batch sizes. Smaller GPUs (such as `eagle`) may encounter out-of-memory errors during training.

---

## Future Work

This project was developed as a dissertation prototype and is not intended for production deployment. The following notes outline a credible production path on Hackney's existing data platform.

**Data platform (Medallion architecture)**

Bronze → Silver reshaping is already implemented as an Airflow DAG with Athena SQL. The ML pipeline would occupy the Silver → Gold layer, split into two decoupled batch jobs:

- **Batch inference** (SageMaker spot instances): reads from silver, runs span extraction and relation extraction, writes raw predictions and confidence scores to a gold predictions table.
- **Entity linking** (AWS Glue job): reads gold predictions alongside the household roster from silver, runs the confidence ranker, and writes final assignments to a separate gold assignments table. Decoupling these two jobs means entity linking can be re-run independently — for example when household data changes — without repeating expensive GPU inference.

Note: the current inference script processes records sequentially. This would need to be refactored to batch across records before the SageMaker job is viable at scale.

**Human review**

A writable extension of the visualiser would serve as a review dashboard, reading from the gold assignments table and routing predictions to human reviewers based on confidence score. High-confidence assignments (exact name match, sole responsible tenant) would skip review; low-confidence ones (family role fallback, tenure fallback) would always require it.

**Feedback loop**

Reviewer corrections should write to a dedicated feedback table, preserving the original prediction alongside the human judgement rather than overwriting in place. This table is the retraining signal for periodic fine-tuning. Correction rate per label category over time is the key metric for detecting model drift.

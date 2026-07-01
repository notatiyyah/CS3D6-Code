# CS3D6 — Additional Needs NER & Relation Extraction Pipeline

This repository contains the complete data preparation, model training, and evaluation pipeline for a dissertation project investigating automatic extraction of **Additional Needs** (vulnerabilities and risks) from housing case notes provided by the London Borough of Hackney.

The pipeline identifies both the additional need described in the text and the individual to whom it refers.

---

## Overview

The pipeline consists of three tasks:

### 1. Additional Need & Person Reference Extraction

The repository contains two approaches for span extraction:

| Model  | Purpose                |
| ------ | ---------------------- |
| **Span Candidate Classifier** | Primary model. Detects both Additional Needs and person references. Because spans are predicted independently, it naturally supports overlapping annotations.                                 |
| **BIO Token Classifier**      | Baseline sequence-labelling model. Detects Additional Needs only. BIO tagging cannot represent overlapping entities, so supporting person references would require training a separate model. |

The candidate-based approach was ultimately adopted for the main pipeline because it achieved stronger performance while simultaneously extracting person references required for relation extraction.

### 2. Relation Extraction

Once person references and Additional Needs have been identified, a marker-based binary classifier links each need to the appropriate individual.

---

Then later, under the span classifier section, I'd add a short note:

> **Why use the candidate-based model?**
>
> Unlike BIO tagging, the candidate-based classifier predicts each span independently, allowing overlapping annotations to be represented naturally. This enables a single model to detect both Additional Needs and person references. In contrast, a BIO-based pipeline would require an additional person-reference model, since overlapping entities cannot be represented within a single BIO tag sequence.

And in the evaluation/baselines section, I'd briefly mention AWS Comprehend:

> **Comparison with AWS Comprehend**
>
> AWS Comprehend Medical-style NER follows a token classification approach and similarly does not support overlapping entity annotations. As a result, person references are excluded from the Comprehend comparison, and only Additional Need extraction is evaluated.

---

# Performance Context

A duplicate annotation study was conducted on **88** records to estimate human consistency.

| Metric | Macro F1 |
| ------ | -------- |
| Strict | ~0.62    |
| Loose  | ~0.78    |

These values provide an approximate upper bound for expected model performance, as exact span boundaries often contain genuine ambiguity even for human annotators.

---

# Repository Structure

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
│
└── src/
    ├── common/
    ├── preprocessing/
    ├── training/
    ├── eval/
    └── utils/
```

The major directories are:

| Directory        | Purpose                                                  |
| ---------------- | -------------------------------------------------------- |
| `preprocessing/` | Dataset construction and annotation processing           |
| `training/`      | Span extraction and relation extraction training scripts |
| `eval/`          | Evaluation scripts and metrics                           |
| `common/`        | Shared models and utilities                              |
| `shared/`        | Model architecture and inference helpers shared across training, eval, and inference |
| `inference/`     | End-to-end inference pipeline (need to run training first)   |
| `visualizer/`    | Vite app for side-by-side comparison of ground truth and model predictions |

---

# Installation

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

# Data Preparation

The preprocessing pipeline should be executed sequentially.

| Step | Script                                        |
| ---- | --------------------------------------------- |
| 1    | `1_construct_taxonomy.py`                     |
| 2    | `2_prep_gold_standard.ipynb`                  |
| 3    | `3_generate_label_studio_ui.py` *(optional)*  |
| 4    | `4_handle_gemini_annotations.py` *(optional)* |
| 5    | `5_post_process_annotated_data.ipynb`         |
| 6    | `6_train_test_split.py`                       |
| 7    | `7_prep_data_for_comprehend.py`               |

These scripts construct the taxonomy, prepare annotation datasets, process completed annotations, create train/validation/test splits, and generate comparison datasets.

---

# Model Training

## Span Extraction

The primary span extraction model is a candidate-based classifier rather than a traditional BIO tagging model.

Recommended backbone:

```
microsoft/deberta-v3-base
```

## Relation Extraction

Relation extraction is performed after entity detection.

The model injects special entity markers around candidate entity spans before classifying whether an additional need belongs to that individual.

This consistently outperformed simple heuristic approaches such as linking each need to the nearest preceding person.

---

# Evaluation

Span evaluation is implemented in `src/eval/evaluators.py`.

Three matching strategies are reported:

| Metric | Description                                                                           |
| ------ | ------------------------------------------------------------------------------------- |
| Loose  | Any overlap between predicted and gold spans                                          |
| Strict | Exact character boundary match                                                        |
| IoU    | Intersection-over-Union evaluated across multiple thresholds (0.3, 0.5, 0.7 and 0.9) |

Relation extraction is evaluated with pair-level exact match via `RelationEvaluator` in the same file.

End-to-end evaluation (`src/eval/eval_e2e.py`) additionally reports:
- Span recall — what proportion of gold spans the span classifier found
- Relation recall given spans found — of gold relations where both spans were detected, how many were linked correctly

This separates span classifier errors from relation model errors.
This allows both boundary accuracy and approximate span localisation to be assessed.

---

# Running on the Warwick Batch Compute Cluster

The experiments were designed to run on the University of Warwick Department of Computer Science Batch Compute Cluster.

Full instructions for submitting and monitoring jobs are available in the official documentation:

[https://warwick.ac.uk/fac/sci/dcs/intranet/user_guide/batch_compute/](https://warwick.ac.uk/fac/sci/dcs/intranet/user_guide/batch_compute/)

## Recommended GPUs

The training configuration assumes access to a GPU with **24 GB of VRAM**.

Recommended partitions:

* `falcon`
* `gecko`

These provide NVIDIA A10/A5000 GPUs and support the configured batch sizes. Smaller GPUs (such as `eagle`) may encounter out-of-memory errors during training.
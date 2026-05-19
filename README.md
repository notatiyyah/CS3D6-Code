# CS3D6 — Additional Needs NER Pipeline

This repo contains the data preparation, annotation, and model training pipeline for a NLP project that automatically identifies **Additional Needs** (vulnerabilities and risks) in housing case notes from the London Borough of Hackney.

---

## What it does

The system performs **Named Entity Recognition (NER)** on housing case notes to:

1. Identify **vulnerability/risk spans** — exact text substrings labelled with an Additional Needs category (e.g. `health_mental_health`, `safety_risk_domestic_abuse`)
2. Identify **entity spans** — people referenced in the note (by name, role, or pronoun)
3. Link vulnerability spans to the entity they belong to

Annotations are produced in [Label Studio](https://labelstud.io/) and used to fine-tune a DistilBERT-based span extraction model.

---

## Repo structure

```
.
├── notebooks/
│   ├── prep-gold-standard.ipynb        # Pulls data from Athena, applies weak labels, samples gold standard
│   ├── generate_label_studio_ui.ipynb  # Generates the Label Studio XML config from the taxonomy CSV
│   └── handle-gemini-annotations.ipynb # Converts Gemini pre-annotations into Label Studio format
│
├── annotation/
│   ├── annotation_codebook.md          # Annotation guidelines for human annotators
│   ├── gemini-prompt.txt               # Prompt used to generate Gemini pre-annotations
│   └── label-studio-ui-template.xml   # Template for the Label Studio annotation interface
│
├── reference/                          # Copied from the infrastructure repo — not run here
│   ├── additional-needs-reshape.py     # Airflow DAG script: reshapes raw notes and writes to S3/Glue
│   └── sql/
│       └── notes-reshape.sql           # SQL query used by the reshape script
│
└── data/                               # Gitignored — not committed
    ├── input/                          # Raw taxonomy CSVs and annotation exports
    ├── output/                         # Generated taxonomy, Label Studio config, gold standard
    └── models/                         # Fine-tuned DistilBERT model checkpoints
```

---

## Pipeline overview

### 0. Data reshaping (`reference/additional-needs-reshape.py` + `reference/sql/notes-reshape.sql`)
- Runs as a scheduled **Airflow DAG** in a separate infrastructure repo — included here for reference
- Executes `notes-reshape.sql` via AWS Athena against the raw MTFH notes and tenure tables
- Reshapes notes (which can be targeted at a person, tenure, or asset) so each note is linked to the relevant tenure based on timestamps
- Appends data quality flags (e.g. inactive tenancies, organisation targets, date parse failures)
- Writes the output to S3 as Snappy-compressed Parquet and registers it in the AWS Glue Data Catalog

### 1. Data preparation (`notebooks/prep-gold-standard.ipynb`)
- Connects to AWS Athena to pull housing case notes from the reshaped table
- Normalises common abbreviations (e.g. `tnt` → `tenant`, `asb` → `anti-social behaviour`)
- Applies regex-based **weak labels** to each note for each AN category
- Performs stratified sampling to produce a balanced gold standard dataset for annotation ($n=2000$)

### 2. Label Studio UI generation (`notebooks/generate_label_studio_ui.ipynb`)
- Reads the taxonomy CSV (`data/output/taxonomy_v2_autogen.csv`)
- Generates a Label Studio XML config with colour-coded labels and value hints

### 3. Gemini pre-annotation (`annotation/gemini-prompt.txt` + `notebooks/handle-gemini-annotations.ipynb`)
- Notes are pre-annotated using the prompt in `annotation/gemini-prompt.txt` via a Gemini / Google Sheets integration (not in this repo).
- The resulting CSV is converted into Label Studio import format by this notebook.

### 4. Human annotation
- Annotators review and correct pre-annotations in Label Studio using `annotation/annotation_codebook.md` as a guide

### 5. Model training
- Fine-tuned DistilBERT models are stored under `data/models`

---

## Annotation taxonomy

The taxonomy covers **11 high-level categories** of Additional Needs:

- Care 
- Caution
- Reasonable Adjustments
- Communication
- Disability 
- Health
- Housing Conditions
- Life Events
- Mobility
- Property Level
- Safety & Risk

See [`annotation/annotation_codebook.md`](annotation/annotation_codebook.md) for full label definitions, value hints, and annotation rules.

---

## AWS setup

Data is pulled from AWS Athena. You will need:
- An AWS profile named `data-platform-housing-prod`
- Access to the `housing-refined-zone` database, `additional_needs_notes_reshaped` table
- `awswrangler` and `boto3` installed

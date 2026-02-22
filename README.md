# UK Tax Categorisation — DPL Classification

Automated classification of UK financial transaction descriptions into
**HMRC Detailed Profit & Loss (DPL) tags** (DPL001–DPL076).

Given a free-text accounting line description such as
`"INV-78451 – Deloitte LLP – Audit services FY2025"`, the models predict the
correct DPL tag (e.g. `DPL003 – Audit and accountancy, tax services`).

---

## Problem Overview

| Dimension | Detail |
|---|---|
| Task | Multi-class text classification |
| Input | Free-text transaction / accounting line description |
| Output | One of 76 active DPL tags (DPL001–DPL076) |
| Classes | 76 |
| Dataset | Synthetic — generated with `generate_dpl_data.py` |
| Language | English |

The tag taxonomy covers the full HMRC DPL chart of accounts: staff costs,
professional services, finance & treasury, gains/losses, operational costs,
revenue, and more. See [`DPLTags.md`](DPLTags.md) for the full list.

---

## Repository Layout

```
UKTaxCategorisationDPL/
│
├── datasets/                    # Train / val / test CSVs
│   ├── dpl_full.csv
│   ├── dpl_train.csv
│   ├── dpl_val.csv
│   └── dpl_test.csv
│
├── models/                      # Saved model artefacts (gitignored)
│   ├── svc_base/                # TF-IDF + LinearSVC
│   ├── lgbm/                    # TF-IDF + LightGBM
│   ├── xgboost/                 # TF-IDF + SVD + XGBoost
│   ├── tfidf_svc/               # Pipeline: TF-IDF → LinearSVC (CPU notebook)
│   ├── tfidf_lr/                # Pipeline: TF-IDF → Logistic Regression
│   ├── cal_svc/                 # CalibratedClassifierCV wrapper
│   ├── lr/                      # Standalone Logistic Regression
│   ├── distilbert/              # Fine-tuned DistilBERT
│   ├── distilbert_conf/         # DistilBERT with confidence analysis
│   ├── hierarchical/            # Two-stage hierarchical classifier
│   └── hierarchical_conf/       # Two-stage hierarchical + confidence scores
│
├── docs/                        # Design notes and research
│   ├── 1-Explore-Options.md
│   ├── 2-Data-analysis-and-strategy.md
│   ├── 3-Hierarchical-grouping-and-confusion-pairs.md
│   └── 4-data-generation.md
│
├── evaluation/                  # Evaluation-only notebooks (load saved model)
│   ├── dpl_finetune_eval.ipynb
│   ├── dpl_boosting_comparison_eval.ipynb
│   ├── dpl_finetune_cpu_eval.ipynb
│   ├── dpl_hierarchical_confidence_eval.ipynb
│   └── dpl_finetune_hierarchical_eval.ipynb
│
├── dpl_finetune.ipynb           # Training: DeBERTa-v3-base (GPU)
├── dpl_boosting_comparison.ipynb# Training: LinearSVC vs LightGBM vs XGBoost
├── dpl_finetune_cpu.ipynb       # Training: TF-IDF models + DistilBERT (CPU)
├── dpl_cpu_confidence.ipynb     # Training: CPU models with confidence analysis
├── dpl_hierarchical_confidence.ipynb # Training: two-stage hierarchical + conf.
├── dpl_finetune_hierarchical.ipynb   # Training: two-stage hierarchical (flat baseline)
│
├── generate_dpl_data.py         # Synthetic dataset generator
├── analyse_dataset.py           # Dataset statistics and EDA script
├── DPLTags.md                   # Full DPL tag reference table
├── requirements.txt             # Python dependencies
└── README.md
```

---

## Training Notebooks

Each notebook is self-contained: it loads the dataset, trains the model(s),
evaluates on the test set, and saves all artefacts to `models/`.

### `dpl_finetune.ipynb` — DeBERTa-v3-base (GPU)

Fine-tunes `microsoft/deberta-v3-base` for sequence classification.

- Tokenises descriptions with the DeBERTa SentencePiece tokeniser
- Trains with HuggingFace `Trainer` (mixed precision on GPU)
- Evaluates with accuracy, weighted F1, per-class F1 bar chart, and confusion
  matrix heatmap
- Saves model + tokeniser to `models/deberta/model/`

**Best for**: highest raw accuracy, production-grade classification.

---

### `dpl_boosting_comparison.ipynb` — LinearSVC vs LightGBM vs XGBoost

Compares three gradient-boosted / linear approaches on the same TF-IDF features.

| Model | Accuracy | Weighted F1 | Train time |
|---|---|---|---|
| LinearSVC | 99.96 % | 99.96 % | ~5 s |
| LightGBM | 99.74 % | 99.74 % | ~139 s |
| XGBoost | 99.47 % | 99.47 % | ~137 s |

- TF-IDF vectoriser + SVD dimensionality reduction (for tree models)
- `CalibratedClassifierCV` wrapper on LinearSVC for probability estimates
- Per-model: accuracy/F1 comparison table, per-class F1 chart, confidence
  distribution, LightGBM feature importance
- Saves each model + its TF-IDF/SVD/label-encoder to `models/{svc_base,lgbm,xgboost}/`

**Best for**: very fast inference, no GPU required, near-perfect accuracy.

---

### `dpl_finetune_cpu.ipynb` — TF-IDF pipelines + DistilBERT (CPU)

CPU-friendly comparison of sklearn pipelines and a lightweight transformer.

| Model | Accuracy | Weighted F1 |
|---|---|---|
| TF-IDF + LinearSVC | 99.82 % | 99.82 % |
| TF-IDF + LogReg | 99.82 % | 99.82 % |
| DistilBERT | 100.00 % | 100.00 % |

- Full `sklearn.pipeline.Pipeline` objects (vectoriser + classifier in one file)
- DistilBERT fine-tuned with HuggingFace `Trainer` (CPU mode, fp32)
- Bar chart comparison, per-class F1 analysis, inference examples
- Saves to `models/{tfidf_svc,tfidf_lr,distilbert}/`

**Best for**: CPU-only environments with a balance of speed and accuracy.

---

### `dpl_cpu_confidence.ipynb` — CPU Models with Confidence Analysis

Extends the CPU comparison with calibrated probability scores.

| Model | Accuracy | Mean Confidence |
|---|---|---|
| Calibrated LinearSVC | 99.96 % | 97.8 % |
| Logistic Regression | 99.96 % | 96.0 % |
| DistilBERT | 99.91 % | — |

- `CalibratedClassifierCV` wrapping `LinearSVC` for well-calibrated probabilities
- Confidence histogram, reliability diagram, low-confidence sample analysis
- Saves to `models/{cal_svc,lr,distilbert_conf}/`

**Best for**: production use-cases where a confidence threshold must gate predictions.

---

### `dpl_hierarchical_confidence.ipynb` — Two-Stage Hierarchical Classifier

Implements a custom `HierarchicalDPLClassifier` that first predicts a broad
**group** (e.g. *Finance & Treasury*) and then narrows to a specific **DPL tag**
within that group.

| Metric | Value |
|---|---|
| Overall accuracy | 99.96 % |
| Weighted F1 | 99.96 % |
| L1 group accuracy | 100.00 % |
| Mean group confidence | 99.6 % |
| Mean joint confidence | 97.5 % |

- 10 top-level groups derived from the DPL taxonomy
- Separate L2 classifier per group (one-vs-rest within the group)
- Joint confidence = L1 confidence × L2 confidence
- Confidence threshold analysis and group-level heatmaps
- Saves full hierarchy to `models/hierarchical_conf/`

**Best for**: explainability — surfaces which semantic group was chosen before the
final tag, and provides two independent confidence signals.

---

### `dpl_finetune_hierarchical.ipynb` — Hierarchical vs Flat Baseline

Benchmarks the hierarchical classifier against a flat LinearSVC baseline,
providing confusion analysis across the group structure.

| Approach | Accuracy | Weighted F1 |
|---|---|---|
| Flat LinearSVC | 99.96 % | 99.96 % |
| Hierarchical (L1+L2) | 99.96 % | 99.96 % |
| L1 group only | 100.00 % | 100.00 % |

- L1 confusion matrix (group-level errors)
- L2 per-group accuracy breakdown
- High-risk confusion pair analysis
- Saves to `models/hierarchical/`

**Best for**: understanding where and why the classifier confuses specific tags.

---

## Evaluation Notebooks (`evaluation/`)

Each `*_eval.ipynb` notebook is a **run-only** companion to its training
counterpart. It loads the pre-trained artefacts and re-runs every evaluation
section — no re-training required.

| Eval notebook | Loads from | Companion training notebook |
|---|---|---|
| `dpl_finetune_eval.ipynb` | `models/deberta/` | `dpl_finetune.ipynb` |
| `dpl_boosting_comparison_eval.ipynb` | `models/{svc_base,lgbm,xgboost}/` | `dpl_boosting_comparison.ipynb` |
| `dpl_finetune_cpu_eval.ipynb` | `models/{tfidf_svc,tfidf_lr,distilbert}/` | `dpl_finetune_cpu.ipynb` |
| `dpl_hierarchical_confidence_eval.ipynb` | `models/hierarchical_conf/` | `dpl_hierarchical_confidence.ipynb` |
| `dpl_finetune_hierarchical_eval.ipynb` | `models/hierarchical/` | `dpl_finetune_hierarchical.ipynb` |

Each eval notebook follows this structure:
1. **Imports** — identical to the training notebook
2. **Load test data** — only the test split; no training data needed
3. **Load saved model(s)** — raises `FileNotFoundError` if artefacts are missing
4. **Evaluation** — classification report, per-class F1, confusion matrix, confidence analysis
5. **Inference helper** — ready-to-use prediction functions with sample descriptions

> Paths inside evaluation notebooks are relative (`../datasets/`, `../models/`)
> because the notebooks live one level below the project root.

---

## Dataset

Synthetic data generated by `generate_dpl_data.py`.

| Split | Rows |
|---|---|
| Full | ~57,000 |
| Train | ~45,600 |
| Validation | ~5,700 |
| Test | ~5,700 |

Each row: `dpl_tag` (e.g. `DPL003`), `description` (free-text accounting line).
Descriptions include realistic noise: invoice numbers, vendor names, dates,
department codes, and varied formatting.

Run `python analyse_dataset.py` for class distribution and descriptive statistics.

---

## Setup

**Requirements: Python 3.11**

```bash
# Create and activate virtual environment
python -m venv .venv311
.venv311\Scripts\activate        # Windows
# source .venv311/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

For GPU training (`dpl_finetune.ipynb`) install the CUDA-enabled PyTorch build
before running:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Launch Jupyter:

```bash
jupyter lab
```

---

## Quick Start

### Train a model

Open any training notebook and run all cells top-to-bottom. Models are saved
automatically to `models/`.

### Evaluate a saved model

Open the corresponding notebook in `evaluation/` and run all cells. No GPU or
long training run required.

### Run inference from Python

```python
import joblib

# Example: TF-IDF + LinearSVC (fastest)
pipe = joblib.load("models/tfidf_svc/model.joblib")

descriptions = [
    "INV-78451 Deloitte LLP Audit services FY2025",
    "Monthly AWS infrastructure invoice",
    "Staff away-day travel and hotel costs",
]

predictions = pipe.predict(descriptions)
print(predictions)
# ['DPL003', 'DPL037', 'DPL065']
```

---

## DPL Tag Groups

The 76 tags are organised into 10 semantic groups used by the hierarchical models:

| Group | Example tags |
|---|---|
| Finance & Treasury | DPL005, DPL006, DPL035, DPL057, DPL058, DPL075, DPL076 |
| Staff & Employment | DPL048, DPL051, DPL052, DPL062, DPL068, DPL070, DPL071, DPL073 |
| Professional & External Services | DPL003, DPL009, DPL038, DPL049, DPL060 |
| Operational & Administrative | DPL008, DPL025, DPL033, DPL047, DPL054, DPL055, DPL061, DPL063, DPL064, DPL069 |
| Gains, Losses & Adjustments | DPL013–DPL023, DPL026–DPL029, DPL039, DPL043, DPL044 |
| Revenue & Income | DPL006, DPL024, DPL030, DPL045, DPL058, DPL066, DPL072 |
| Tax & Compliance | DPL003, DPL015, DPL031, DPL036, DPL053 |
| Depreciation & Amortisation | DPL002, DPL010, DPL067 |
| IT, Marketing & Communications | DPL001, DPL037, DPL056, DPL064 |
| Other | DPL046, DPL074 |

See [`docs/3-Hierarchical-grouping-and-confusion-pairs.md`](docs/3-Hierarchical-grouping-and-confusion-pairs.md)
for the full grouping rationale.

---

## Dependencies

| Package | Purpose |
|---|---|
| `scikit-learn` | TF-IDF, LinearSVC, Logistic Regression, calibration |
| `lightgbm` | Gradient-boosted trees |
| `xgboost` | Gradient-boosted trees |
| `transformers` | DeBERTa, DistilBERT fine-tuning and inference |
| `datasets` | HuggingFace dataset wrapper for tokenised inputs |
| `accelerate` | HuggingFace distributed / mixed-precision training |
| `pandas` | Data loading and manipulation |
| `matplotlib` / `seaborn` | Evaluation visualisations |
| `joblib` | Model serialisation |
| `jupyter` / `ipykernel` | Notebook environment |

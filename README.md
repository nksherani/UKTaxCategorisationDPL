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
├── datasets/                         # Train / val / test CSVs
│   ├── dpl_full.csv
│   ├── dpl_train.csv
│   ├── dpl_val.csv
│   └── dpl_test.csv
│
├── models/                           # Saved model artefacts (gitignored)
│   ├── tfidf_svc/                    # Pipeline: TF-IDF → LinearSVC
│   ├── tfidf_lr/                     # Pipeline: TF-IDF → Logistic Regression
│   ├── distilbert/                   # Fine-tuned DistilBERT (CPU)
│   ├── cal_svc/                      # CalibratedClassifierCV wrapper (LinearSVC)
│   ├── lr/                           # Standalone Logistic Regression
│   ├── distilbert_conf/              # DistilBERT with confidence analysis
│   ├── svc_base/                     # TF-IDF + LinearSVC (boosting comparison)
│   ├── lgbm/                         # TF-IDF + LightGBM
│   ├── xgboost/                      # TF-IDF + SVD + XGBoost
│   ├── hierarchical/                 # Two-stage hierarchical classifier
│   ├── hierarchical_conf/            # Two-stage hierarchical + confidence scores
│   ├── cpu_model_comparison.json     # Metrics: dpl_main.ipynb
│   ├── cpu_confidence_metrics.json   # Metrics: dpl_main_confidence.ipynb
│   └── boosting_comparison.json      # Metrics: dpl_boosting_comparison.ipynb
│
├── docs/                             # Design notes and research
│   ├── 1-Explore-Options.md
│   ├── 2-Data-analysis-and-strategy.md
│   ├── 3-Hierarchical-grouping-and-confusion-pairs.md
│   └── 4-data-generation.md
│
├── evaluation/                       # Evaluation-only notebooks (load saved model)
│   ├── dpl_main_eval.ipynb
│   ├── dpl_main_confidence_eval.ipynb
│   ├── dpl_boosting_comparison_eval.ipynb
│   ├── dpl_hierarchical_eval.ipynb
│   └── dpl_hierarchical_confidence_eval.ipynb
│
├── dpl_main.ipynb                    # Training: TF-IDF + LinearSVC / LogReg / DistilBERT (CPU)
├── dpl_main_confidence.ipynb         # Training: CPU models with confidence scores
├── dpl_boosting_comparison.ipynb     # Training: LinearSVC vs LightGBM vs XGBoost
├── dpl_hierarchical.ipynb            # Training: two-stage hierarchical vs flat baseline
├── dpl_hierarchical_confidence.ipynb # Training: two-stage hierarchical + confidence scores
│
├── dpl_main.html                     # HTML export of dpl_main.ipynb
├── dpl_main_confidence.html          # HTML export of dpl_main_confidence.ipynb
│
├── generate_dpl_data.py              # Synthetic dataset generator
├── analyse_dataset.py                # Dataset statistics and EDA script
├── DPLTags.md                        # Full DPL tag reference table
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## Training Notebooks

Each notebook is self-contained: it loads the dataset, trains the model(s),
evaluates on the test set, and saves all artefacts to `models/`.

### `dpl_main.ipynb` — CPU-Friendly Baseline (TF-IDF + DistilBERT)

Trains three models optimised for CPU speed and compares them head-to-head.

| Model | Accuracy | Weighted F1 |
|---|---|---|
| TF-IDF + LinearSVC | 99.82 % | 99.82 % |
| TF-IDF + Logistic Regression | 99.82 % | 99.82 % |
| DistilBERT (fine-tuned, CPU) | 100.00 % | 100.00 % |

- Full `sklearn.pipeline.Pipeline` objects (vectoriser + classifier in one file)
- DistilBERT fine-tuned with HuggingFace `Trainer` (CPU mode, fp32)
- Bar chart comparison, per-class F1 analysis, inference examples
- Saves to `models/{tfidf_svc, tfidf_lr, distilbert}/`

**Best for**: CPU-only environments with a balance of speed and accuracy.

---

### `dpl_main_confidence.ipynb` — CPU Models with Confidence Scores

Extends the CPU comparison with calibrated probability scores for every prediction.

| Model | Accuracy | Weighted F1 | Mean Confidence |
|---|---|---|---|
| TF-IDF + Calibrated SVC | 99.96 % | 99.96 % | 97.8 % |
| TF-IDF + Logistic Regression | 99.96 % | 99.96 % | 96.0 % |
| DistilBERT | 99.91 % | 99.91 % | 90.1 % |

| Model | How confidence is produced |
|---|---|
| Calibrated SVC | `CalibratedClassifierCV` wraps LinearSVC with Platt sigmoid calibration |
| Logistic Regression | Native `predict_proba()` |
| DistilBERT | Softmax over logits |

- Confidence histogram, reliability diagrams, low-confidence sample analysis
- Confidence threshold analysis (coverage vs accuracy trade-off)
- Top-K inference helper with ranked predictions
- Saves to `models/{cal_svc, lr, distilbert_conf}/`

**Best for**: production use-cases where a confidence threshold must gate predictions.

---

### `dpl_boosting_comparison.ipynb` — LinearSVC vs LightGBM vs XGBoost

Compares three gradient-boosted / linear approaches on the same TF-IDF features.

| Model | Accuracy | Weighted F1 | Train time | Mean Confidence |
|---|---|---|---|---|
| LinearSVC | 99.96 % | 99.96 % | ~5 s | 97.8 % |
| LightGBM | 99.74 % | 99.74 % | ~139 s | 99.4 % |
| XGBoost | 99.47 % | 99.47 % | ~137 s | 96.7 % |

- TF-IDF vectoriser + SVD dimensionality reduction (for tree models)
- `CalibratedClassifierCV` wrapper on LinearSVC for probability estimates
- Per-model: accuracy/F1 comparison table, per-class F1 chart, confidence distribution, LightGBM feature importance
- Saves to `models/{svc_base, lgbm, xgboost}/`

**Best for**: very fast inference, no GPU required, near-perfect accuracy.

---

### `dpl_hierarchical.ipynb` — Hierarchical 2-Step Classifier vs Flat Baseline

Implements a custom `HierarchicalDPLClassifier` that first predicts a broad
**group** (e.g. *Finance & Treasury*) and then narrows to a specific **DPL tag**
within that group. Benchmarks the approach against a flat LinearSVC baseline.

```
Description
    │
    ▼  Level 1 — 9-class group classifier
  Group  (e.g. "Finance & Treasury")
    │
    ▼  Level 2 — group-specific classifier (~5–13 classes)
  DPL Tag  (e.g. "DPL035")
```

| Approach | Accuracy | Weighted F1 |
|---|---|---|
| Flat LinearSVC | 99.96 % | 99.96 % |
| Hierarchical (L1 + L2) | 99.96 % | 99.96 % |
| L1 group only | 100.00 % | 100.00 % |

- All models use TF-IDF + LinearSVC — trains in < 30 seconds total on CPU
- L1 confusion matrix (group-level errors), L2 per-group accuracy breakdown
- High-risk confusion pair analysis
- Saves to `models/hierarchical/`

**Best for**: understanding where and why the classifier confuses specific tags.

---

### `dpl_hierarchical_confidence.ipynb` — Hierarchical Classifier + Confidence Scores

Extends the hierarchical model with calibrated probability scores at each stage.

| Metric | Value |
|---|---|
| Overall accuracy | 99.96 % |
| Weighted F1 | 99.96 % |
| L1 group accuracy | 100.00 % |
| Mean group confidence | 99.6 % |
| Mean joint confidence | 97.5 % |

- Joint confidence = L1 confidence × L2 confidence (most conservative overall measure)
- Confidence threshold analysis and group-level confidence heatmaps
- Saves full hierarchy to `models/hierarchical_conf/`

**Best for**: explainability — surfaces which semantic group was chosen before the
final tag, and provides two independent confidence signals.

---

## Evaluation Notebooks (`evaluation/`)

Each `*_eval.ipynb` notebook is a **run-only** companion to its training
counterpart. It loads the pre-trained artefacts and re-runs every evaluation
section — no re-training required.

| Eval notebook | Loads from | Companion training notebook |
|---|---|---|
| `dpl_main_eval.ipynb` | `models/{tfidf_svc, tfidf_lr, distilbert}/` | `dpl_main.ipynb` |
| `dpl_main_confidence_eval.ipynb` | `models/{cal_svc, lr, distilbert_conf}/` | `dpl_main_confidence.ipynb` |
| `dpl_boosting_comparison_eval.ipynb` | `models/{svc_base, lgbm, xgboost}/` | `dpl_boosting_comparison.ipynb` |
| `dpl_hierarchical_eval.ipynb` | `models/hierarchical/` | `dpl_hierarchical.ipynb` |
| `dpl_hierarchical_confidence_eval.ipynb` | `models/hierarchical_conf/` | `dpl_hierarchical_confidence.ipynb` |

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
| Full | ~15,123 |
| Train | ~10,584 |
| Validation | ~2,267 |
| Test | ~2,272 |

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

Open the corresponding notebook in `evaluation/` and run all cells. No long
training run required.

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

```python
import joblib

# Example: Calibrated SVC — with confidence scores
data = joblib.load("models/cal_svc/model.joblib")
pipe, le = data["pipeline"], data["label_encoder"]

proba = pipe.predict_proba(descriptions)
predicted_indices = proba.argmax(axis=1)
tags = le.inverse_transform(predicted_indices)
confidences = proba.max(axis=1)

for desc, tag, conf in zip(descriptions, tags, confidences):
    print(f"{tag}  ({conf:.1%})  {desc[:60]}")
```

---

## DPL Tag Groups

The 76 tags are organised into 9 semantic groups used by the hierarchical models:

| Group | Example tags |
|---|---|
| Finance & Treasury | DPL005, DPL006, DPL035, DPL057, DPL058, DPL075, DPL076 |
| Staff & Employment | DPL048, DPL051, DPL052, DPL062, DPL068, DPL070, DPL071, DPL073 |
| Professional & External Services | DPL003, DPL009, DPL038, DPL049, DPL060 |
| Operational & Administrative | DPL008, DPL025, DPL033, DPL047, DPL054, DPL055, DPL061, DPL063, DPL064, DPL069 |
| Gains, Losses & Adjustments | DPL013–DPL023, DPL026–DPL029, DPL039, DPL043, DPL044 |
| Revenue & Income | DPL024, DPL030, DPL045, DPL066, DPL072 |
| Tax & Compliance | DPL015, DPL031, DPL036, DPL053 |
| Depreciation & Amortisation | DPL002, DPL010, DPL067 |
| IT, Marketing & Communications | DPL001, DPL037, DPL056, DPL064 |

See [`docs/3-Hierarchical-grouping-and-confusion-pairs.md`](docs/3-Hierarchical-grouping-and-confusion-pairs.md)
for the full grouping rationale.

---

## Dependencies

| Package | Purpose |
|---|---|
| `scikit-learn` | TF-IDF, LinearSVC, Logistic Regression, calibration |
| `lightgbm` | Gradient-boosted trees |
| `xgboost` | Gradient-boosted trees |
| `transformers` | DistilBERT fine-tuning and inference |
| `datasets` | HuggingFace dataset wrapper for tokenised inputs |
| `accelerate` | HuggingFace mixed-precision training utilities |
| `sentencepiece` / `protobuf` | Tokeniser support |
| `pandas` | Data loading and manipulation |
| `matplotlib` / `seaborn` | Evaluation visualisations |
| `joblib` | Model serialisation |
| `jupyter` / `ipykernel` | Notebook environment |

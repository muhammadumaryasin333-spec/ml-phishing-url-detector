# Explainable Phishing URL and Website Detection

Explainable phishing URL and website detection using machine learning, transformer-based features, and SHAP explanations.

MSc Cybersecurity project. EDA, preprocessing, baseline modelling, leakage auditing, and Week 6 XGBoost/LightGBM comparison are complete; transformer and explainability work remain pending.

## Description

This project aims to detect phishing URLs and websites using machine learning. Beyond raw classification accuracy, the focus is on **explainability**: combining transformer-based feature representations with SHAP explanations so that model predictions can be interpreted and trusted in a security context.

## Repository Structure

```
.
├── README.md            # Project overview and progress tracking
├── requirements.txt     # Starter Python dependencies
├── .gitignore           # Python, venv, datasets, reports, models
├── data/
│   ├── raw/             # Original, immutable datasets
│   └── processed/       # Cleaned / feature-engineered datasets
├── notebooks/           # Reserved folder; current workflow uses Python scripts
├── src/                 # Source code (Python package)
│   └── __init__.py
├── models/              # Trained model artifacts
├── reports/             # Generated figures, metrics, results
├── app/                 # Demo application
└── tests/               # Unit / integration tests
```

## Setup Instructions

```bash
# 1. Clone and enter the project
cd ml-phishing-url-detector

# 2. Create a virtual environment
python3 -m venv venv

# 3. Activate it
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows (PowerShell/CMD)

# 4. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

```

To deactivate the environment when done:

```bash
deactivate
```

## Week 2 Progress

- Dataset source identified: **PhiUSIIL Phishing URL Dataset** from the UCI Machine Learning Repository.
- Dataset download location prepared: `data/raw/`
- Basic data exploration script added: `src/data_exploration.py`
- Dataset summary and class distribution output planned in `reports/`
- No ML model training yet.

### How to run dataset exploration

First download the PhiUSIIL Phishing URL Dataset from the UCI ML Repository and place the CSV at `data/raw/PhiUSIIL_Phishing_URL_Dataset.csv`. Then:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python src/data_exploration.py
```

Outputs:

- `reports/dataset_summary.txt`
- `reports/class_distribution.png`

## Pre-IPR EDA and Preprocessing Progress

- Dataset downloaded and added at `data/raw/PhiUSIIL_Phishing_URL_Dataset.csv`
- Initial exploration completed
- Complete EDA script added: `src/eda_analysis.py`
- Class distribution charts generated
- Correlation heatmap generated
- Missing values and duplicate checks completed
- Outlier summary prepared
- Feature grouping prepared
- Cleaned dataset generated
- Train/validation/test split prepared
- No model training yet

Run the pre-IPR workflow:

```bash
source .venv/bin/activate
pip install -r requirements.txt

python src/eda_analysis.py
python src/preprocess_data.py
```

Output locations:

```text
reports/eda/
reports/eda/figures/
reports/eda/tables/
data/processed/
```

## Week 5 Progress: Baseline Model Training

- Started baseline model training after EDA and preprocessing.
- Trained Logistic Regression, Decision Tree, and Random Forest.
- Evaluated models using accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC.
- Saved baseline results in `reports/model_results/`.
- Saved baseline model files in `models/baseline/`.
- Completed feature ablation with and without URLSimilarityIndex; saved the audit in `reports/model_results/feature_ablation/`.
- Uses the Python training script only; no Jupyter notebook is required.
- Next step: train stronger models and compare them with the baseline results.
- Advanced models, transformers, SHAP, and the demo app are not included in this week.

Run baseline training:

```bash
source .venv/bin/activate
pip install -r requirements.txt

python src/train_baseline_models.py
```

Expected outputs:

```text
reports/model_results/baseline_results.csv
reports/model_results/baseline_training_report.md
reports/model_results/confusion_matrices/
reports/model_results/roc_curves/
models/baseline/
```

## Progress Tracker

- [x] Initial repo setup
- [x] Dataset download
- [x] Data exploration
- [x] Preprocessing preparation
- [x] Baseline ML models
- [x] Evaluation metrics
- [x] Wider leakage and target-proxy audit
- [x] Domain-group-disjoint evaluation protocol
- [x] XGBoost and LightGBM comparison
- [ ] SHAP explainability
- [ ] Demo app

## Week 6 Progress: Leakage Audit and Advanced Models

- Replaced the legacy row-random evaluation protocol for new experiments with
  normalized-host-group-disjoint 70/15/15 splits.
- Verified zero normalized-host and exact-URL overlap across audited splits.
- Quarantined `URLSimilarityIndex`, `CharContinuationRate`,
  `TLDLegitimateProb`, and `URLCharProb` from primary Week 6 modelling.
- Retained raw `TLD` as categorical data with train-fitted one-hot encoding.
- Retrained Logistic Regression, Decision Tree, and Random Forest under the
  audited protocol.
- Completed bounded validation-only tuning for XGBoost and LightGBM.
- Evaluated each locked model once on the frozen audited test split.
- Best audited test macro F1: LightGBM, 0.999971. This is dataset-specific
  evidence, not deployment or cross-dataset generalisation evidence.

Run the Week 6 workflow:

```bash
source .venv/bin/activate
pip install -r requirements.txt

python src/run_leakage_audit.py
python src/train_advanced_models.py
```

Key outputs:

```text
data/processed/audited/
reports/model_results/leakage_audit/
reports/model_results/advanced_tuning_results.csv
reports/model_results/advanced_results.csv
reports/model_results/model_comparison.csv
reports/model_results/advanced_training_report.md
models/advanced/
models/audited_baseline/
```

## Next Steps

1. Run the Week 7 transformer or lightweight URL-sequence feature experiment using raw URL strings and the audited split membership.
2. Compare URL-text representations against the audited tabular models without using the frozen test set for model selection.
3. Add SHAP explanations to the best audited model.
4. Wrap the final audited model in a demo app under `app/`.

> XGBoost and LightGBM are pinned for Week 6. Transformer, SHAP, and demo dependencies will be added only when their stages begin.

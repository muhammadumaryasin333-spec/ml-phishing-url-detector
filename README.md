# Explainable Phishing URL and Website Detection

Explainable phishing URL and website detection using machine learning, transformer-based features, and SHAP explanations.

MSc Cybersecurity project. This repository contains completed EDA, preprocessing, and Week 5 baseline model training; advanced models and explainability work remain pending.

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
- Uses the Python training script only; no Jupyter notebook is required.
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
- [ ] SHAP explainability
- [ ] Demo app

## Next Steps

1. Audit highly target-separative baseline features before making generalisation claims.
2. Replace placeholder high-cardinality label encoding with a fitted preprocessing pipeline or URL/text feature extraction.
3. Compare advanced models against the Week 5 baseline in a later stage.
4. Add transformer-based features and SHAP explanations.
5. Wrap the best model in a demo app under `app/`.

> Heavier dependencies (transformers, torch, xgboost, lightgbm, shap, streamlit) will be added to `requirements.txt` as each stage requires them.

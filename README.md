# Explainable Phishing URL and Website Detection

Explainable phishing URL and website detection using machine learning, transformer-based features, and SHAP explanations.

MSc Cybersecurity project. This repository currently contains the initial project scaffold only — the ML pipeline is not implemented yet.

## Description

This project aims to detect phishing URLs and websites using machine learning. Beyond raw classification accuracy, the focus is on **explainability**: combining transformer-based feature representations with SHAP explanations so that model predictions can be interpreted and trusted in a security context.

## Repository Structure

```
.
├── README.md            # Project overview and progress tracking
├── requirements.txt     # Starter Python dependencies
├── .gitignore           # Python, venv, datasets, notebooks, models
├── data/
│   ├── raw/             # Original, immutable datasets
│   └── processed/       # Cleaned / feature-engineered datasets
├── notebooks/           # Jupyter notebooks for exploration
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

# 5. (Optional) register a Jupyter kernel
python -m ipykernel install --user --name phishing-detector
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

## Progress Tracker

- [ ] Initial repo setup
- [ ] Dataset download
- [ ] Data exploration
- [ ] Preprocessing
- [ ] Baseline ML models
- [ ] Evaluation metrics
- [ ] SHAP explainability
- [ ] Demo app

## Next Steps

1. Select and download a phishing/legitimate URL dataset into `data/raw/`.
2. Explore the data in `notebooks/` and document initial findings.
3. Build a preprocessing and feature-extraction pipeline in `src/`.
4. Train baseline ML models and record evaluation metrics in `reports/`.
5. Add transformer-based features and SHAP explanations.
6. Wrap the best model in a demo app under `app/`.

> Heavier dependencies (transformers, torch, xgboost, lightgbm, shap, streamlit) will be added to `requirements.txt` as each stage requires them.

# Explainable Phishing URL Detector

Project for phishing detection using leakage-audited machine learning, URL-text experiments, SHAP explanations, and a local FastAPI demo.

The project uses the [UCI PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusil-phishing-url-dataset): 235,795 records, with label `0` for phishing and `1` for legitimate. Results are strong on this dataset but are not evidence of real-world, temporal, or cross-dataset performance.

## What is included

- Exploratory data analysis and reproducible preprocessing.
- Wider leakage and target-proxy audit.
- Normalized-host-group-disjoint train/validation/test protocol.
- Audited Logistic Regression, Decision Tree, Random Forest, XGBoost, and LightGBM models.
- Raw URL character TF-IDF and pinned MiniLM embedding experiments.
- Global and local SHAP explanations for the audited LightGBM model.
- FastAPI plus vanilla HTML/CSS/JavaScript demo with model switching, guarded live-page extraction, and uncertainty reporting.
- Reproducibility manifests, saved metrics, plots, model hashes, and final report.

## Evaluation design

The current protocol creates deterministic 70/15/15 splits grouped by normalized hostname. It asserts zero normalized-host and exact-URL overlap across splits. Model and preprocessing choices use train and validation only; each locked configuration is evaluated once on the frozen test set.

Four unresolved derived features are quarantined from primary tabular modelling: `URLSimilarityIndex`, `CharContinuationRate`, `TLDLegitimateProb`, and `URLCharProb`. Raw identifiers and text are excluded from tabular models. Host grouping does not merge sibling subdomains under one registered domain, which remains a limitation.

## Main results

All rows below use the same 35,371-row audited test split. Input scopes differ, so tabular and URL-only scores are descriptive rather than interchangeable.

| Model | Input | Macro F1 | ROC-AUC | Phishing recall |
| --- | --- | ---: | ---: | ---: |
| LightGBM | URL and webpage-derived tabular features | 0.999971 | 1.000000 | 0.999934 |
| XGBoost | URL and webpage-derived tabular features | 0.999913 | 1.000000 | 0.999934 |
| Character TF-IDF + Logistic Regression | Raw URL only | 0.997747 | 0.999446 | 0.994849 |
| MiniLM embeddings + Logistic Regression | Raw URL only | 0.980200 | 0.996591 | 0.972395 |

LightGBM is the strongest dataset experiment. The demo can approximate its 47 required inputs through a guarded HTML request, but the dataset authors did not publish an executable extraction contract. Live page-model scores are therefore useful comparisons, not equivalent reproductions of dataset evaluation.

SHAP analysis identifies `LineOfCode`, `IsHTTPS`, `NoOfSelfRef`, `NoOfImage`, and `NoOfOtherSpecialCharsInURL` as the highest mean-absolute attribution groups in the sampled audited test cohort. These are model attributions, not causal explanations. Several webpage-volume features are highly separable in this corpus and may reflect collection bias.

## Setup

Python 3.11 is the recorded environment.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Place `PhiUSIIL_Phishing_URL_Dataset.csv` in `data/raw/` if it is not already available. Raw data, processed data, model bundles, and embedding caches are intentionally ignored by Git.

## Run the demo

The demo expects saved artifacts under `models/advanced/`, `models/audited_baseline/`, and `models/url_text/`.

```bash
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The dropdown contains seven clearly named models: TF-IDF, MiniLM, Logistic Regression, Decision Tree, Random Forest, XGBoost, and LightGBM. Automatic and Compare All are convenience modes, not additional models. Only host-separated, leakage-audited tabular models are exposed.

Live analysis performs one bounded server-side HTML fetch. It blocks private, loopback, link-local, reserved, and mixed public/private DNS answers; pins the validated IP; revalidates redirects; accepts only standard HTTP/S ports and HTML; limits redirects and compressed/decompressed body size; and never executes JavaScript. This reduces SSRF risk but does not make the demo suitable for public internet deployment.

API example:

```bash
curl -X POST http://127.0.0.1:8000/api/analyze \
  -H 'Content-Type: application/json' \
  --data '{"url":"https://www.google.com/","model":"automatic","deep_scan":true}'
```

Returned scores are dataset model outputs, not calibrated real-world risk percentages. A prediction is a signal, not proof that a site is safe or malicious.

## Reproduce the pipeline

These commands are intentionally separate because later stages are expensive and refuse to overwrite artifacts unless their explicit replacement option is supplied.

```bash
python src/data_exploration.py
python src/eda_analysis.py
python src/preprocess_data.py
python src/train_baseline_models.py
python src/run_leakage_audit.py
python src/train_advanced_models.py
python src/train_url_text_models.py
python src/generate_explanations.py
```

`train_url_text_models.py` downloads the pinned `sentence-transformers/all-MiniLM-L6-v2` revision on its first full run. The URL-text manifests record revision, versions, hashes, timing, and cache alignment evidence.

## Verification

```bash
python -m compileall -q app src
node --check app/static/app.js
python -m pip check
```

## Repository map

```text
app/                                  FastAPI service and responsive web UI
data/processed/audited/               Group-disjoint split data and manifest
models/advanced/                      XGBoost and LightGBM bundles
models/audited_baseline/               Group-audited classical model bundles
models/url_text/                      URL-only model bundles and manifests
reports/eda/                          EDA evidence
reports/model_results/                Leakage audit and model comparisons
reports/transformer_experiments/      URL-text tuning and locked test results
reports/explainability/               SHAP global/local evidence
reports/final_report.pdf              Submission-ready report
src/                                  Reproducible data, training, and explanation scripts
```

## Limitations and responsible use

- Single static dataset; no temporal or independent external evaluation.
- Normalized-host grouping is weaker than registered-domain grouping.
- Near-perfect tabular scores may partly reflect dataset construction and webpage-feature proxies.
- URL-text models can produce severe false positives: TF-IDF scored GitHub as phishing in a live check.
- The live extractor approximates seven semantic fields because no executable reference implementation was published.
- Automatic mode abstains as `uncertain` when scores are near the threshold, models disagree, or a requested live scan fails. It is not score calibration.
- Live checks on 16 August 2026 labelled Google and Python.org legitimate, while GitHub remained uncertain because the tree models disagreed sharply.
- Demo does not inspect sender context, domain reputation, JavaScript-rendered content, domain age, or newly registered-domain feeds.
- Do not use this educational prototype as a browser security control or sole basis for blocking, reporting, or entering credentials.

Dataset citation: Prasad and Chandra, “PhiUSIIL: A diverse security profile empowered phishing URL detection framework based on similarity index and incremental learning,” *Computers & Security*, 2024. Dataset licence: CC BY 4.0.

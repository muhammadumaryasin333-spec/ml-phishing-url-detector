# Week 5 Baseline Model Training Report

## Purpose

Create a reproducible, untuned machine-learning benchmark for phishing URL detection before evaluating advanced ensemble and transformer-based methods.

## Dataset files used

- `data/processed/train.csv` — 165,056 rows
- `data/processed/validation.csv` — 35,369 rows
- `data/processed/test.csv` — 35,370 rows
- Detected target column: `label`

## Models trained

- Logistic Regression (`max_iter=1000`)
- Decision Tree Classifier (default baseline settings)
- Random Forest Classifier (`n_estimators=100`)
- `random_state=42` used for all applicable models; class weighting: balanced.

## Feature handling

- Trained with 51 feature columns.
- Numeric imputation is fitted on training data; numeric scaling is applied only to Logistic Regression. Categorical fields, if present, are imputed and one-hot encoded inside each training-fitted pipeline.
- Excluded train-only placeholder encodings: `FILENAME`, `URL`, `Domain`, `Title`.
- Each excluded column was high-cardinality in training and encoded as `-1` for at least 50% of both held-out splits. The raw category values cannot be recovered safely from the processed CSV files.

## Evaluation metrics

Accuracy, macro precision, macro recall, macro F1-score, ROC-AUC, classification reports, test-set confusion matrices, and test-set ROC curves.
Macro averages prevent either class from being favoured in the summary metrics. The ROC curve treats `0` as the positive phishing class when available.

## Test set results

| model_name | accuracy | precision | recall | f1_score | roc_auc |
| --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.9999 | 0.9999 | 0.9998 | 0.9999 | 1.0000 |
| decision_tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| random_forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Best baseline model

Selection rule: highest test macro F1-score, with ROC-AUC used as the tie-breaker.
- Best by this rule: **decision_tree** (F1=1.0000, ROC-AUC=1.0000).
- Best ROC-AUC: **decision_tree** (ROC-AUC=1.0000, F1=1.0000).

## Observations

- Preprocessing is fitted inside each model pipeline using training data only, which avoids validation/test leakage from imputation or categorical encoding. Numeric scaling is applied only to Logistic Regression because tree models do not require it.
- High-cardinality train-only placeholder encodings are excluded before fitting so models cannot learn unavailable category mappings.
- Validation metrics provide a development comparison; test metrics are retained for the final Week 5 baseline comparison.
- No hyperparameter search, calibration, or explainability method was used in this baseline stage.

## Limitations of baseline models

- Results are based on one fixed train/validation/test split and default model settings.
- Existing high-cardinality categorical values may have been encoded during the preprocessing stage, so their original URL/text semantics are unavailable here.
- The near-perfect scores and use of `URLSimilarityIndex` as the first Decision Tree split require a target-proxy/leakage audit before making generalisation claims.
- These tabular baselines do not model URL or website text context as transformers can.
- Model probabilities are not calibrated and results should not be interpreted as deployment readiness.

## Next step

Compare advanced models such as XGBoost/LightGBM in a later project stage, then evaluate transformer-based representations and explainability methods. These are not included in Week 5 baseline training.

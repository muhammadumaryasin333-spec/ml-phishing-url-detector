# Week 5 Feature Ablation Audit

## Purpose

Test whether the unusually strong Week 5 baseline results depend on URLSimilarityIndex. The same processed train, validation, and test splits were used in both conditions.

## Conditions

- with_url_similarity_index: current valid baseline feature set.
- without_url_similarity_index: same feature set with URLSimilarityIndex removed.
- Models: Logistic Regression, Decision Tree, and Random Forest.
- No hyperparameter tuning, new data split, advanced model, SHAP, or LIME was used.

## Shared Feature Safeguard

- Both conditions retain the existing exclusion of train-only placeholder encodings: `FILENAME`, `URL`, `Domain`, `Title`.
- This isolates the effect of URLSimilarityIndex rather than reintroducing the previously identified placeholder-encoding issue.

## All Recorded Results

| ablation_condition | model_name | dataset_split | accuracy | precision | recall | f1_score | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| with_url_similarity_index | logistic_regression | validation | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 1.0000 |
| with_url_similarity_index | logistic_regression | test | 0.9999 | 0.9999 | 0.9998 | 0.9999 | 1.0000 |
| with_url_similarity_index | decision_tree | validation | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| with_url_similarity_index | decision_tree | test | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| with_url_similarity_index | random_forest | validation | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| with_url_similarity_index | random_forest | test | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| without_url_similarity_index | logistic_regression | validation | 0.9993 | 0.9992 | 0.9993 | 0.9992 | 1.0000 |
| without_url_similarity_index | logistic_regression | test | 0.9993 | 0.9994 | 0.9993 | 0.9993 | 1.0000 |
| without_url_similarity_index | decision_tree | validation | 0.9990 | 0.9990 | 0.9991 | 0.9990 | 0.9991 |
| without_url_similarity_index | decision_tree | test | 0.9988 | 0.9988 | 0.9988 | 0.9988 | 0.9988 |
| without_url_similarity_index | random_forest | validation | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 1.0000 |
| without_url_similarity_index | random_forest | test | 0.9998 | 0.9998 | 0.9998 | 0.9998 | 1.0000 |

## Test-Set Impact of Removing URLSimilarityIndex

| model_name | f1_with_feature | f1_without_feature | f1_change | roc_auc_with_feature | roc_auc_without_feature | roc_auc_change |
| --- | --- | --- | --- | --- | --- | --- |
| decision_tree | 1.0000 | 0.9988 | -0.0012 | 1.0000 | 0.9988 | -0.0012 |
| logistic_regression | 0.9999 | 0.9993 | -0.0005 | 1.0000 | 1.0000 | -0.0000 |
| random_forest | 1.0000 | 0.9998 | -0.0002 | 1.0000 | 1.0000 | -0.0000 |

## Interpretation

- Removing URLSimilarityIndex did not cause a test F1-score decline of at least 0.01 for the tested baseline models.
- A performance change in this audit is evidence about dependency on one dataset feature. It does not by itself prove or disprove data leakage.
- Advanced-model comparison should use the audited feature decision and document it before making generalisation claims.

## Status

Feature ablation audit completed. No baseline model artifacts were replaced and no advanced models were trained.

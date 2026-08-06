# Week 6 Advanced Model Training Report

## Outcome

XGBoost and LightGBM were tuned using training and validation data under the normalized-host-group-disjoint protocol. Each locked model was then refitted on train plus validation and evaluated once on the frozen test set. The three Week 5 baselines were retrained under the same audited protocol.

## Leakage-Audited Protocol

- Protocol: `stratified_normalized_host_group_70_15_15_v1`.
- Split grouping: normalized host, with zero host and exact-URL overlap.
- Split ratio: approximately 70% train, 15% validation, 15% test.
- Test data did not participate in preprocessing, tuning, weighting, model selection, or threshold selection.
- Legacy Week 5 scores are historical and not directly comparable.

## Feature Handling

- Input feature columns: 47; transformed columns: 709.
- Excluded identifiers/raw text: `Domain`, `FILENAME`, `Title`, `URL`.
- Quarantined unresolved derived features: `CharContinuationRate`, `TLDLegitimateProb`, `URLCharProb`, `URLSimilarityIndex`.
- Numeric median imputation and TLD one-hot encoding were fitted on training rows only during tuning, then on train plus validation for final refitting.

## Bounded Tuning

- Four predeclared structural candidates per advanced model.
- One train-derived balanced-sample-weight ablation for each structural winner.
- Selection order: validation macro F1, phishing recall, ROC-AUC, then candidate ID as a deterministic tie-breaker.
- No early stopping or open-ended search was used.

### Selected validation configurations

| model_name | candidate_id | weighting | f1_macro | roc_auc |
| --- | --- | --- | --- | --- |
| xgboost | xgb_02_balanced | balanced_sample_weight | 1.0000 | 1.0000 |
| lightgbm | lgb_03_balanced | balanced_sample_weight | 1.0000 | 1.0000 |

## Frozen Test Comparison

| model_name | f1_macro | roc_auc | balanced_accuracy | mcc | phishing_recall | rank_f1_macro |
| --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.9995 | 1.0000 | 0.9995 | 0.9991 | 0.9993 | 4 |
| decision_tree | 0.9988 | 0.9988 | 0.9988 | 0.9977 | 0.9987 | 5 |
| random_forest | 0.9999 | 1.0000 | 0.9998 | 0.9997 | 0.9997 | 3 |
| xgboost | 0.9999 | 1.0000 | 0.9999 | 0.9998 | 0.9999 | 2 |
| lightgbm | 1.0000 | 1.0000 | 1.0000 | 0.9999 | 0.9999 | 1 |

Test ranks are descriptive only; they were not used to select or retune models.

## Reproducibility Evidence

- XGBoost 3.2.0 and LightGBM 4.7.0 are pinned in `requirements.txt`.
- Model manifests record parameters, versions, split/input hashes, feature order, runtime, and artifact hashes.
- Complete preprocessing-plus-model joblib bundles passed prediction round-trip verification; native boosters are supplementary only.

## Limitations

- Host-level grouping does not merge sibling subdomains under one registrable domain because no pinned Public Suffix List is included.
- Very strong webpage-derived features may reflect collection bias; remaining features are valid for this dataset experiment, not proven deployment inputs.
- Weighted probabilities are uncalibrated when a weighted candidate is selected.
- This single-dataset evaluation does not establish temporal or cross-dataset generalisation.

## Artifact Index

- `reports/model_results/advanced_tuning_results.csv`
- `reports/model_results/advanced_results.csv`
- `reports/model_results/model_comparison.csv`
- `models/advanced/` and `models/audited_baseline/`

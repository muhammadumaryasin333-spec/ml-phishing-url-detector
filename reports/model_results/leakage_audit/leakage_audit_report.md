# Wider Target-Proxy and Leakage Audit

## Decision

The legacy row-random split is retained only as historical Week 5 evidence. Week 6 uses a new normalized-host-group-disjoint 70/15/15 protocol. Model and feature choices must use training and validation only; test remains frozen until each model configuration is locked.

## Data Integrity

- Raw rows: 235,795.
- Exact duplicate rows removed before splitting: 0.
- New split asserts zero exact URL and normalized-host overlap.
- Split membership and source-file SHA-256 hashes are recorded in `data/processed/audited/split_manifest.json`.
- Host grouping does not collapse sibling subdomains to registered domains; that requires a pinned Public Suffix List and remains a limitation.

## Audited Split Summary

| split | rows | domains | class | class_count | class_percentage |
| --- | --- | --- | --- | --- | --- |
| train | 165055 | 153614 | 0 | 70662 | 42.8112 |
| train | 165055 | 153614 | 1 | 94393 | 57.1888 |
| validation | 35369 | 32933 | 0 | 15141 | 42.8087 |
| validation | 35369 | 32933 | 1 | 20228 | 57.1913 |
| test | 35371 | 32547 | 0 | 15142 | 42.8091 |
| test | 35371 | 32547 | 1 | 20229 | 57.1909 |

## Legacy Split Overlap

| entity | left_split | right_split | shared_unique_values | right_rows_with_overlap | right_overlap_percentage |
| --- | --- | --- | --- | --- | --- |
| URL | train | validation | 87 | 87 | 0.2460 |
| URL | train | test | 90 | 90 | 0.2545 |
| URL | validation | test | 26 | 26 | 0.0735 |
| Domain | train | validation | 1672 | 2936 | 8.3011 |
| Domain | train | test | 1684 | 2882 | 8.1481 |
| Domain | validation | test | 588 | 1695 | 4.7922 |

## Feature Policy

- Excluded identifiers/raw text: `FILENAME`, `URL`, `Domain`, `Title`.
- Quarantined from primary Week 6 modelling: `CharContinuationRate`, `TLDLegitimateProb`, `URLCharProb`, `URLSimilarityIndex`.
- `TLD` is retained as raw categorical data and encoded by a train-fitted one-hot encoder.
- Remaining webpage and URL-derived measurements are valid only for this dataset experiment until deployment-time extraction is separately audited.

## Univariate Validation Audit

Features with direction-agnostic validation ROC-AUC >= 0.98 are flagged for investigation, not automatically called leakage.

| feature | validation_direction_agnostic_roc_auc | validation_stump_macro_f1 | requires_investigation | feature_policy |
| --- | --- | --- | --- | --- |
| URLSimilarityIndex | 0.9965 | 0.9969 | True | quarantined |
| LineOfCode | 0.9904 | 0.9539 | True | dataset_experiment_only |
| NoOfExternalRef | 0.9885 | 0.9615 | True | dataset_experiment_only |
| NoOfImage | 0.9802 | 0.9419 | True | dataset_experiment_only |
| NoOfJS | 0.9729 | 0.9323 | False | dataset_experiment_only |
| NoOfSelfRef | 0.9714 | 0.9460 | False | dataset_experiment_only |
| NoOfCSS | 0.9568 | 0.9043 | False | dataset_experiment_only |
| HasSocialNet | 0.8942 | 0.8800 | False | dataset_experiment_only |
| HasCopyrightInfo | 0.8760 | 0.8658 | False | dataset_experiment_only |
| HasDescription | 0.8478 | 0.8321 | False | dataset_experiment_only |
| LargestLineLength | 0.8224 | 0.8067 | False | dataset_experiment_only |
| NoOfOtherSpecialCharsInURL | 0.8193 | 0.7605 | False | dataset_experiment_only |

Flagged features: URLSimilarityIndex, LineOfCode, NoOfExternalRef, NoOfImage.

## Interpretation

- No direct numeric copy of the target was found if the direct-copy column is false for every feature in the CSV evidence.
- Strong single-feature separation can reflect dataset construction bias, a legitimate phishing indicator, or a target proxy. Statistics alone cannot distinguish those causes.
- The new protocol removes normalized-host entity overlap and fails closed on the derived features with unresolved fold-safe provenance.
- Legacy and audited scores are not directly comparable because the split and feature policy changed.

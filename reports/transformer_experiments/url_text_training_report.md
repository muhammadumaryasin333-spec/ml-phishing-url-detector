# Week 7 URL-Text Experiment

## Outcome

Character TF-IDF and a frozen SentenceTransformer embedding representation are independently tuned on train/validation and evaluated only after their downstream Logistic Regression configuration is locked.

## Protocol

- Audited split protocol: stratified_normalized_host_group_70_15_15_v1.
- Raw URL strings are treated as text only; this experiment never fetches URLs.
- Normalized-host and exact-URL overlap checks run before every experiment.
- Test labels are excluded from representation fitting, tuning, selection, and threshold choice.
- Selection order: validation macro F1, phishing recall, ROC-AUC, candidate ID.
- Bounded candidates: C in {0.25, 1.0, 4.0}; no weight ablation or open-ended search.

## Representations

- TF-IDF: raw URL character-within-word n-grams (3-5), train-fitted only.
- SentenceTransformer: all-MiniLM-L6-v2 at immutable revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41, 384 float32 normalized dimensions, CPU by default.
- all-MiniLM-L6-v2 is an English sentence encoder, not URL-specific; its 256 wordpiece truncation makes this an exploratory contextual baseline, not a deployment claim.

## Results

- representation=tfidf_logistic_regression; candidate_id=tfidf_lr_c_4_00; f1_macro=0.9977; roc_auc=0.9994; balanced_accuracy=0.9974; phishing_recall=0.9948; mcc=0.9955
- representation=sentence_transformer_logistic_regression; candidate_id=sentence_transformer_lr_c_4_00; f1_macro=0.9802; roc_auc=0.9966; balanced_accuracy=0.9796; phishing_recall=0.9724; mcc=0.9604

## Reproducibility

- Model manifests record selected settings, package versions, split hashes, artifact hashes, and timing.
- Embedding caches contain float32 matrices, immutable row IDs, and hashes; mismatches fail closed.
- No raw URLs, labels, or row-level predictions are saved in the embedding cache or reports.

## Limitations

- Grouping is at normalized-host level, not registered-domain level.
- URL-only results are not directly interchangeable with Week 6 webpage-feature models.
- This single dataset has no temporal or external-dataset generalisation evidence.

## Artifacts

- reports/transformer_experiments/url_text_tuning_results.csv
- reports/transformer_experiments/url_text_results.csv
- reports/transformer_experiments/model_comparison.csv
- models/url_text/

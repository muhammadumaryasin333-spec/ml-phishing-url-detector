# SHAP interpretation notes

These values are model attributions for phishing probability (class 0), relative to a deterministic train-plus-validation background. They are not causal evidence or proof that a feature causes phishing.

TLD one-hot contributions are summed with their signs before feature ranking. No raw URLs or unredacted row identifiers are written to this directory.

The model remains dataset-specific: webpage-derived proxy features and the single-dataset audited evaluation do not establish deployment performance.

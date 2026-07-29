"""Audit the influence of URLSimilarityIndex on Week 5 baseline results.

Run from the project root:

    python src/run_feature_ablation.py
"""

from __future__ import annotations

import pandas as pd

if __package__:
    from .train_baseline_models import (
        PROJECT_ROOT,
        build_models,
        calculate_metrics,
        exclude_train_only_placeholder_columns,
        get_positive_class_scores,
        get_roc_positive_label,
        is_class_imbalanced,
        load_datasets,
        validate_and_split_datasets,
    )
else:
    from train_baseline_models import (
        PROJECT_ROOT,
        build_models,
        calculate_metrics,
        exclude_train_only_placeholder_columns,
        get_positive_class_scores,
        get_roc_positive_label,
        is_class_imbalanced,
        load_datasets,
        validate_and_split_datasets,
    )


ABLATION_DIR = PROJECT_ROOT / "reports" / "model_results" / "feature_ablation"
RESULTS_PATH = ABLATION_DIR / "feature_ablation_results.csv"
REPORT_PATH = ABLATION_DIR / "feature_ablation_report.md"
SUSPICIOUS_FEATURES = ["URLSimilarityIndex"]
METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]


def create_feature_sets(
    features: dict[str, pd.DataFrame],
) -> dict[str, dict[str, pd.DataFrame]]:
    """Return matched feature sets with and without the audited features."""
    missing_features = [
        feature for feature in SUSPICIOUS_FEATURES if feature not in features["train"]
    ]
    if missing_features:
        raise ValueError(
            "Cannot run feature ablation because these features are missing: "
            + ", ".join(missing_features)
        )

    return {
        "with_url_similarity_index": features,
        "without_url_similarity_index": {
            split_name: split_features.drop(columns=SUSPICIOUS_FEATURES)
            for split_name, split_features in features.items()
        },
    }


def format_markdown_table(results: pd.DataFrame) -> str:
    """Create a compact Markdown table for the recorded ablation metrics."""
    columns = ["ablation_condition", "model_name", "dataset_split", *METRIC_COLUMNS]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in results[columns].iterrows():
        rows.append(
            "| "
            + " | ".join(
                [
                    str(row["ablation_condition"]),
                    str(row["model_name"]),
                    str(row["dataset_split"]),
                    *[f"{row[column]:.4f}" for column in METRIC_COLUMNS],
                ]
            )
            + " |"
        )
    return "\n".join([header, separator, *rows])


def build_test_delta_table(results: pd.DataFrame) -> pd.DataFrame:
    """Compare test F1 and ROC-AUC after removing URLSimilarityIndex."""
    test_results = results[results["dataset_split"] == "test"]
    pivot = test_results.pivot(
        index="model_name",
        columns="ablation_condition",
        values=["f1_score", "roc_auc"],
    )
    rows = []
    for model_name in pivot.index:
        rows.append(
            {
                "model_name": model_name,
                "f1_with_feature": pivot.loc[
                    model_name, ("f1_score", "with_url_similarity_index")
                ],
                "f1_without_feature": pivot.loc[
                    model_name, ("f1_score", "without_url_similarity_index")
                ],
                "f1_change": pivot.loc[
                    model_name, ("f1_score", "without_url_similarity_index")
                ]
                - pivot.loc[model_name, ("f1_score", "with_url_similarity_index")],
                "roc_auc_with_feature": pivot.loc[
                    model_name, ("roc_auc", "with_url_similarity_index")
                ],
                "roc_auc_without_feature": pivot.loc[
                    model_name, ("roc_auc", "without_url_similarity_index")
                ],
                "roc_auc_change": pivot.loc[
                    model_name, ("roc_auc", "without_url_similarity_index")
                ]
                - pivot.loc[model_name, ("roc_auc", "with_url_similarity_index")],
            }
        )
    return pd.DataFrame(rows)


def format_delta_table(deltas: pd.DataFrame) -> str:
    """Format the test-set F1 and ROC-AUC changes for the audit report."""
    columns = [
        "model_name",
        "f1_with_feature",
        "f1_without_feature",
        "f1_change",
        "roc_auc_with_feature",
        "roc_auc_without_feature",
        "roc_auc_change",
    ]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in deltas[columns].iterrows():
        rows.append(
            "| "
            + " | ".join(
                [
                    str(row["model_name"]),
                    *[f"{row[column]:.4f}" for column in columns[1:]],
                ]
            )
            + " |"
        )
    return "\n".join([header, separator, *rows])


def write_report(
    results: pd.DataFrame,
    deltas: pd.DataFrame,
    excluded_placeholders: dict[str, dict[str, float]],
) -> None:
    """Write a factual feature-ablation report for the baseline audit."""
    meaningful_drops = deltas[deltas["f1_change"] <= -0.01]["model_name"].tolist()
    if meaningful_drops:
        conclusion = (
            "Removing URLSimilarityIndex caused a meaningful test F1-score decline "
            "for: " + ", ".join(meaningful_drops) + "."
        )
    else:
        conclusion = (
            "Removing URLSimilarityIndex did not cause a test F1-score decline of "
            "at least 0.01 for the tested baseline models."
        )

    lines = [
        "# Week 5 Feature Ablation Audit",
        "",
        "## Purpose",
        "",
        "Test whether the unusually strong Week 5 baseline results depend on "
        "URLSimilarityIndex. The same processed train, validation, and test splits "
        "were used in both conditions.",
        "",
        "## Conditions",
        "",
        "- with_url_similarity_index: current valid baseline feature set.",
        "- without_url_similarity_index: same feature set with URLSimilarityIndex removed.",
        "- Models: Logistic Regression, Decision Tree, and Random Forest.",
        "- No hyperparameter tuning, new data split, advanced model, SHAP, or LIME was used.",
        "",
        "## Shared Feature Safeguard",
        "",
        "- Both conditions retain the existing exclusion of train-only placeholder encodings: "
        + ", ".join(f"`{column}`" for column in excluded_placeholders)
        + ".",
        "- This isolates the effect of URLSimilarityIndex rather than reintroducing "
        "the previously identified placeholder-encoding issue.",
        "",
        "## All Recorded Results",
        "",
        format_markdown_table(results),
        "",
        "## Test-Set Impact of Removing URLSimilarityIndex",
        "",
        format_delta_table(deltas),
        "",
        "## Interpretation",
        "",
        f"- {conclusion}",
        "- A performance change in this audit is evidence about dependency on one "
        "dataset feature. It does not by itself prove or disprove data leakage.",
        "- Advanced-model comparison should use the audited feature decision and "
        "document it before making generalisation claims.",
        "",
        "## Status",
        "",
        "Feature ablation audit completed. No baseline model artifacts were replaced "
        "and no advanced models were trained.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Run the matched baseline feature-ablation experiment."""
    print("Week 5 feature ablation audit")
    print("[1/5] Loading processed dataset splits")
    datasets = load_datasets()
    if datasets is None:
        return 1

    print("[2/5] Validating data and applying existing placeholder safeguards")
    validated_data = validate_and_split_datasets(datasets)
    if validated_data is None:
        return 1
    features, targets, _ = validated_data
    features, excluded_placeholders = exclude_train_only_placeholder_columns(features)

    try:
        feature_sets = create_feature_sets(features)
    except ValueError as error:
        print(f"ERROR: {error}")
        return 1

    class_weight = "balanced" if is_class_imbalanced(targets["train"]) else None
    positive_label = get_roc_positive_label(targets["train"])
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)

    print("[3/5] Training matched baseline models with and without URLSimilarityIndex")
    result_rows = []
    for condition_name, condition_features in feature_sets.items():
        print(
            f"Condition: {condition_name} "
            f"({condition_features['train'].shape[1]} features)"
        )
        models = build_models(condition_features["train"], class_weight)
        for model_name, model in models.items():
            print(f"  Training {model_name}")
            model.fit(condition_features["train"], targets["train"])
            for split_name in ["validation", "test"]:
                predicted = model.predict(condition_features[split_name])
                scores = get_positive_class_scores(
                    model, condition_features[split_name], positive_label
                )
                metrics = calculate_metrics(
                    targets[split_name], predicted, scores, positive_label
                )
                result_rows.append(
                    {
                        "ablation_condition": condition_name,
                        "removed_features": (
                            ""
                            if condition_name.startswith("with_")
                            else ",".join(SUSPICIOUS_FEATURES)
                        ),
                        "feature_count": condition_features["train"].shape[1],
                        "model_name": model_name,
                        "dataset_split": split_name,
                        **metrics,
                    }
                )

    print("[4/5] Saving comparison results")
    results = pd.DataFrame(result_rows)
    results.to_csv(RESULTS_PATH, index=False, float_format="%.6f")

    print("[5/5] Writing feature-ablation report")
    deltas = build_test_delta_table(results)
    write_report(results, deltas, excluded_placeholders)
    print(f"Results: {RESULTS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

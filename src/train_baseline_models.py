"""Train and evaluate Week 5 baseline phishing URL classifiers.

Run from the project root after preprocessing:

    python src/train_baseline_models.py
"""

import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_CACHE_DIR = Path(tempfile.gettempdir()) / "ml_phishing_url_detector_cache"
for cache_directory in [
    RUNTIME_CACHE_DIR,
    RUNTIME_CACHE_DIR / "matplotlib",
    RUNTIME_CACHE_DIR / "xdg",
]:
    cache_directory.mkdir(parents=True, exist_ok=True)
os.environ.setdefault(
    "MPLCONFIGDIR", str(RUNTIME_CACHE_DIR / "matplotlib")
)
os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_CACHE_DIR / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "reports" / "model_results"
CONFUSION_MATRICES_DIR = RESULTS_DIR / "confusion_matrices"
ROC_CURVES_DIR = RESULTS_DIR / "roc_curves"
MODELS_DIR = PROJECT_ROOT / "models" / "baseline"

SPLIT_PATHS = {
    "train": PROCESSED_DIR / "train.csv",
    "validation": PROCESSED_DIR / "validation.csv",
    "test": PROCESSED_DIR / "test.csv",
}
RESULTS_PATH = RESULTS_DIR / "baseline_results.csv"
REPORT_PATH = RESULTS_DIR / "baseline_training_report.md"

TARGET_CANDIDATES = ["label", "Label", "class", "Class", "target", "Target"]
RANDOM_STATE = 42
UNKNOWN_CATEGORY_CODE = -1
MIN_HIGH_CARDINALITY_VALUES = 50
MIN_HELD_OUT_UNKNOWN_RATE = 0.50


def create_output_directories() -> None:
    """Create the directories used for baseline training artifacts."""
    for directory in [RESULTS_DIR, CONFUSION_MATRICES_DIR, ROC_CURVES_DIR, MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def find_target_column(df: pd.DataFrame) -> str | None:
    """Return a supported target column name, matching case-insensitively."""
    for candidate in TARGET_CANDIDATES:
        if candidate in df.columns:
            return candidate

    lower_to_original = {column.lower(): column for column in df.columns}
    for candidate in TARGET_CANDIDATES:
        match = lower_to_original.get(candidate.lower())
        if match is not None:
            return match
    return None


def load_datasets() -> dict[str, pd.DataFrame] | None:
    """Load processed train, validation, and test splits when all are available."""
    missing_paths = [path for path in SPLIT_PATHS.values() if not path.exists()]
    if missing_paths:
        print("ERROR: Processed dataset split files are missing:")
        for path in missing_paths:
            print(f"  - {path.relative_to(PROJECT_ROOT)}")
        print("Preprocessing must be run first using:")
        print("  python src/preprocess_data.py")
        return None

    datasets = {}
    for split_name, path in SPLIT_PATHS.items():
        try:
            datasets[split_name] = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as error:
            print(f"ERROR: Could not read {path.relative_to(PROJECT_ROOT)}: {error}")
            return None
    return datasets


def validate_and_split_datasets(
    datasets: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series], str] | None:
    """Validate target/features and return aligned feature and target data."""
    target_columns = {
        split_name: find_target_column(df) for split_name, df in datasets.items()
    }
    missing_targets = [
        split_name
        for split_name, target_column in target_columns.items()
        if target_column is None
    ]
    if missing_targets:
        print("ERROR: Could not detect a target column in: " + ", ".join(missing_targets))
        print(f"Looked for: {', '.join(TARGET_CANDIDATES)}")
        return None

    features = {
        split_name: df.drop(columns=[target_columns[split_name]])
        for split_name, df in datasets.items()
    }
    targets = {
        split_name: df[target_columns[split_name]] for split_name, df in datasets.items()
    }

    if any(target.isna().any() for target in targets.values()):
        print("ERROR: Target column contains missing values. Resolve this in preprocessing.")
        return None

    train_columns = list(features["train"].columns)
    for split_name in ["validation", "test"]:
        split_columns = list(features[split_name].columns)
        missing_columns = sorted(set(train_columns) - set(split_columns))
        unexpected_columns = sorted(set(split_columns) - set(train_columns))
        if missing_columns or unexpected_columns:
            print(f"ERROR: Feature columns do not match between train and {split_name}.")
            if missing_columns:
                print(f"  Missing in {split_name}: {', '.join(missing_columns)}")
            if unexpected_columns:
                print(f"  Unexpected in {split_name}: {', '.join(unexpected_columns)}")
            print("Re-run preprocessing to create matching dataset splits.")
            return None
        features[split_name] = features[split_name].reindex(columns=train_columns)

    train_classes = set(targets["train"].unique())
    if len(train_classes) != 2:
        print(
            "ERROR: This phishing baseline workflow expects exactly two target classes; "
            f"found {len(train_classes)} in the training split."
        )
        return None

    for split_name in ["validation", "test"]:
        unseen_classes = set(targets[split_name].unique()) - train_classes
        if unseen_classes:
            print(
                f"ERROR: {split_name} contains target classes not present in train: "
                f"{sorted(unseen_classes, key=str)}"
            )
            return None

    return features, targets, str(target_columns["train"])


def exclude_train_only_placeholder_columns(
    features: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float]]]:
    """Exclude high-cardinality values encoded as unknown in held-out splits.

    Preprocessing can encode categories seen only in training as -1 in validation
    and test data. Such values are not meaningful numeric measurements and would
    make a baseline model learn an unavailable training-only mapping.
    """
    excluded_columns = {}
    for column in features["train"].columns:
        train_values = features["train"][column]
        if not pd.api.types.is_numeric_dtype(train_values):
            continue
        if train_values.nunique(dropna=True) < MIN_HIGH_CARDINALITY_VALUES:
            continue
        if (train_values == UNKNOWN_CATEGORY_CODE).any():
            continue

        held_out_unknown_rates = {
            split_name: float(
                (features[split_name][column] == UNKNOWN_CATEGORY_CODE).mean()
            )
            for split_name in ["validation", "test"]
        }
        if all(
            rate >= MIN_HELD_OUT_UNKNOWN_RATE
            for rate in held_out_unknown_rates.values()
        ):
            excluded_columns[column] = held_out_unknown_rates

    if not excluded_columns:
        return features, excluded_columns

    excluded_names = list(excluded_columns)
    usable_features = {
        split_name: df.drop(columns=excluded_names)
        for split_name, df in features.items()
    }
    return usable_features, excluded_columns


def build_preprocessor(
    X_train: pd.DataFrame, scale_numeric: bool
) -> ColumnTransformer:
    """Build a train-fitted preprocessing step for numeric and categorical fields."""
    numeric_columns = X_train.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [
        column for column in X_train.columns if column not in numeric_columns
    ]

    transformers = []
    if numeric_columns:
        numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        numeric_pipeline = Pipeline(steps=numeric_steps)
        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    if not transformers:
        raise ValueError("No usable feature columns were found for training.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def is_class_imbalanced(y_train: pd.Series) -> bool:
    """Return whether target class counts are unequal in the training split."""
    return y_train.value_counts().nunique() > 1


def build_models(X_train: pd.DataFrame, class_weight: str | None) -> dict[str, Pipeline]:
    """Create the three requested baseline classifier pipelines."""
    model_specs = {
        "logistic_regression": (
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
                class_weight=class_weight,
            ),
            True,
        ),
        "decision_tree": (
            DecisionTreeClassifier(
                random_state=RANDOM_STATE,
                class_weight=class_weight,
            ),
            False,
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                class_weight=class_weight,
                n_jobs=-1,
            ),
            False,
        ),
    }
    return {
        model_name: Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train, scale_numeric)),
                ("model", classifier),
            ]
        )
        for model_name, (classifier, scale_numeric) in model_specs.items()
    }


def get_roc_positive_label(y_train: pd.Series) -> object:
    """Use phishing class 0 when present; otherwise use the second sorted class."""
    classes = sorted(y_train.unique(), key=str)
    if 0 in classes:
        return 0

    lower_to_original = {str(label).strip().lower(): label for label in classes}
    for candidate in ["phishing", "malicious", "phish"]:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    return classes[1]


def calculate_metrics(
    y_true: pd.Series,
    y_predicted: np.ndarray,
    y_scores: np.ndarray,
    positive_label: object,
) -> dict[str, float]:
    """Calculate macro classification metrics and binary ROC-AUC."""
    binary_target = (y_true == positive_label).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_predicted),
        "precision": precision_score(y_true, y_predicted, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_predicted, average="macro", zero_division=0),
        "f1_score": f1_score(y_true, y_predicted, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(binary_target, y_scores),
    }


def get_positive_class_scores(
    model: Pipeline, X: pd.DataFrame, positive_label: object
) -> np.ndarray:
    """Return predicted probabilities for the configured ROC positive class."""
    class_labels = list(model.classes_)
    if positive_label not in class_labels:
        raise ValueError(
            f"Configured ROC positive class {positive_label!r} is not in model classes "
            f"{class_labels}."
        )
    positive_index = class_labels.index(positive_label)
    return model.predict_proba(X)[:, positive_index]


def print_evaluation(
    model_name: str,
    split_name: str,
    y_true: pd.Series,
    y_predicted: np.ndarray,
) -> None:
    """Print the classification report required for each evaluation split."""
    print(f"\n{model_name} classification report ({split_name}):")
    print(classification_report(y_true, y_predicted, zero_division=0))


def save_confusion_matrix(
    model_name: str,
    y_true: pd.Series,
    y_predicted: np.ndarray,
) -> Path:
    """Save the test-set confusion matrix for one baseline model."""
    labels = sorted(pd.unique(y_true), key=str)
    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_predicted,
        labels=labels,
        display_labels=[str(label) for label in labels],
        cmap="Blues",
        colorbar=False,
        ax=axis,
    )
    axis.set_title(f"{model_name.replace('_', ' ').title()} - Test Confusion Matrix")
    figure.tight_layout()
    output_path = CONFUSION_MATRICES_DIR / f"{model_name}_confusion_matrix.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_roc_curve(
    model_name: str,
    y_true: pd.Series,
    y_scores: np.ndarray,
    positive_label: object,
    roc_auc: float,
) -> Path:
    """Save the test-set ROC curve for one baseline model."""
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=positive_label)
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(fpr, tpr, color="navy", linewidth=2, label=f"ROC-AUC = {roc_auc:.4f}")
    axis.plot([0, 1], [0, 1], color="gray", linestyle="--", label="No-skill baseline")
    axis.set_title(f"{model_name.replace('_', ' ').title()} - Test ROC Curve")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    output_path = ROC_CURVES_DIR / f"{model_name}_roc_curve.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path


def format_results_table(results: pd.DataFrame) -> str:
    """Return a Markdown table without requiring an additional dependency."""
    columns = ["model_name", "accuracy", "precision", "recall", "f1_score", "roc_auc"]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, result in results[columns].iterrows():
        rows.append(
            "| "
            + " | ".join(
                [
                    str(result["model_name"]),
                    *[f"{result[column]:.4f}" for column in columns[1:]],
                ]
            )
            + " |"
        )
    return "\n".join([header, separator, *rows])


def write_training_report(
    results: pd.DataFrame,
    datasets: dict[str, pd.DataFrame],
    target_column: str,
    positive_label: object,
    class_weight: str | None,
    feature_count: int,
    excluded_columns: dict[str, dict[str, float]],
) -> tuple[pd.Series, pd.Series]:
    """Write the academic Week 5 baseline training report."""
    test_results = results[results["dataset_split"] == "test"].copy()
    best_by_f1 = test_results.sort_values(
        ["f1_score", "roc_auc"], ascending=False
    ).iloc[0]
    best_by_auc = test_results.sort_values(
        ["roc_auc", "f1_score"], ascending=False
    ).iloc[0]
    class_weight_note = "balanced" if class_weight else "not applied (balanced counts)"

    lines = [
        "# Week 5 Baseline Model Training Report",
        "",
        "## Purpose",
        "",
        "Create a reproducible, untuned machine-learning benchmark for phishing URL "
        "detection before evaluating advanced ensemble and transformer-based methods.",
        "",
        "## Dataset files used",
        "",
        *[
            f"- `{path.relative_to(PROJECT_ROOT)}` — {len(datasets[split_name]):,} rows"
            for split_name, path in SPLIT_PATHS.items()
        ],
        f"- Detected target column: `{target_column}`",
        "",
        "## Models trained",
        "",
        "- Logistic Regression (`max_iter=1000`)",
        "- Decision Tree Classifier (default baseline settings)",
        "- Random Forest Classifier (`n_estimators=100`)",
        f"- `random_state=42` used for all applicable models; class weighting: {class_weight_note}.",
        "",
        "## Feature handling",
        "",
        f"- Trained with {feature_count} feature columns.",
        "- Numeric imputation is fitted on training data; numeric scaling is applied "
        "only to Logistic Regression. Categorical fields, if present, are imputed "
        "and one-hot encoded inside each training-fitted pipeline.",
    ]
    if excluded_columns:
        lines.extend(
            [
                "- Excluded train-only placeholder encodings: "
                + ", ".join(f"`{column}`" for column in excluded_columns)
                + ".",
                "- Each excluded column was high-cardinality in training and encoded "
                "as `-1` for at least 50% of both held-out splits. The raw category "
                "values cannot be recovered safely from the processed CSV files.",
            ]
        )
    lines.extend(
        [
            "",
        "## Evaluation metrics",
        "",
        "Accuracy, macro precision, macro recall, macro F1-score, ROC-AUC, "
        "classification reports, test-set confusion matrices, and test-set ROC curves.",
        "Macro averages prevent either class from being favoured in the summary metrics. "
        f"The ROC curve treats `{positive_label}` as the positive phishing class when available.",
        "",
        "## Test set results",
        "",
        format_results_table(test_results),
        "",
        "## Best baseline model",
        "",
        "Selection rule: highest test macro F1-score, with ROC-AUC used as the tie-breaker.",
        f"- Best by this rule: **{best_by_f1['model_name']}** "
        f"(F1={best_by_f1['f1_score']:.4f}, ROC-AUC={best_by_f1['roc_auc']:.4f}).",
        f"- Best ROC-AUC: **{best_by_auc['model_name']}** "
        f"(ROC-AUC={best_by_auc['roc_auc']:.4f}, F1={best_by_auc['f1_score']:.4f}).",
        "",
        "## Observations",
        "",
        "- Preprocessing is fitted inside each model pipeline using training data only, "
        "which avoids validation/test leakage from imputation or categorical encoding. "
        "Numeric scaling is applied only to Logistic Regression because tree models do not require it.",
        "- High-cardinality train-only placeholder encodings are excluded before fitting "
        "so models cannot learn unavailable category mappings.",
        "- Validation metrics provide a development comparison; test metrics are retained "
        "for the final Week 5 baseline comparison.",
        "- No hyperparameter search, calibration, or explainability method was used in "
        "this baseline stage.",
        "",
        "## Limitations of baseline models",
        "",
        "- Results are based on one fixed train/validation/test split and default model settings.",
        "- Existing high-cardinality categorical values may have been encoded during the "
        "preprocessing stage, so their original URL/text semantics are unavailable here.",
        "- The near-perfect scores and use of `URLSimilarityIndex` as the first Decision "
        "Tree split require a target-proxy/leakage audit before making generalisation claims.",
        "- These tabular baselines do not model URL or website text context as transformers can.",
        "- Model probabilities are not calibrated and results should not be interpreted as deployment readiness.",
        "",
        "## Next step",
        "",
        "Compare advanced models such as XGBoost/LightGBM in a later project stage, then "
        "evaluate transformer-based representations and explainability methods. These are "
        "not included in Week 5 baseline training.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return best_by_f1, best_by_auc


def main() -> int:
    """Run baseline training, evaluation, and artifact generation."""
    print("Week 5 baseline model training")
    print("[1/7] Checking processed dataset splits")
    datasets = load_datasets()
    if datasets is None:
        return 1

    print("[2/7] Validating target and feature columns")
    validated_data = validate_and_split_datasets(datasets)
    if validated_data is None:
        return 1
    features, targets, target_column = validated_data
    features, excluded_columns = exclude_train_only_placeholder_columns(features)
    if features["train"].empty:
        print("ERROR: No usable feature columns remain after validation safeguards.")
        return 1
    print(f"Detected target column: {target_column}")
    print(
        "Split sizes: "
        + ", ".join(f"{name}={len(df):,}" for name, df in datasets.items())
    )
    if excluded_columns:
        excluded_details = ", ".join(
            f"{column} (validation={rates['validation']:.1%}, test={rates['test']:.1%})"
            for column, rates in excluded_columns.items()
        )
        print(f"Excluded train-only placeholder encodings: {excluded_details}")
    print(f"Usable feature columns: {features['train'].shape[1]}")

    class_weight = "balanced" if is_class_imbalanced(targets["train"]) else None
    positive_label = get_roc_positive_label(targets["train"])
    print(
        f"Class weighting: {class_weight or 'not applied'}; "
        f"ROC positive class: {positive_label!r}"
    )

    print("[3/7] Preparing output directories and baseline pipelines")
    create_output_directories()
    try:
        models = build_models(features["train"], class_weight)
    except ValueError as error:
        print(f"ERROR: {error}")
        return 1

    results = []
    print("[4/7] Training Logistic Regression, Decision Tree, and Random Forest")
    for model_name, model in models.items():
        print(f"\nTraining {model_name.replace('_', ' ').title()}...")
        model.fit(features["train"], targets["train"])
        joblib.dump(model, MODELS_DIR / f"{model_name}.joblib")
        print(f"Saved model: {MODELS_DIR.relative_to(PROJECT_ROOT) / f'{model_name}.joblib'}")

        for split_name in ["validation", "test"]:
            y_true = targets[split_name]
            y_predicted = model.predict(features[split_name])
            y_scores = get_positive_class_scores(
                model, features[split_name], positive_label
            )
            metrics = calculate_metrics(y_true, y_predicted, y_scores, positive_label)
            results.append(
                {
                    "model_name": model_name,
                    "dataset_split": split_name,
                    **metrics,
                }
            )
            print_evaluation(model_name, split_name, y_true, y_predicted)

            if split_name == "test":
                print("Saving test-set confusion matrix and ROC curve")
                save_confusion_matrix(model_name, y_true, y_predicted)
                save_roc_curve(
                    model_name,
                    y_true,
                    y_scores,
                    positive_label,
                    metrics["roc_auc"],
                )

    print("[5/7] Saving validation and test metrics")
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False, float_format="%.6f")

    print("[6/7] Writing baseline training report")
    best_by_f1, best_by_auc = write_training_report(
        results_df,
        datasets,
        target_column,
        positive_label,
        class_weight,
        features["train"].shape[1],
        excluded_columns,
    )

    print("[7/7] Baseline training complete")
    print(f"Results: {RESULTS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(
        "Best test model by macro F1-score: "
        f"{best_by_f1['model_name']} ({best_by_f1['f1_score']:.4f})"
    )
    print(
        "Best test model by ROC-AUC: "
        f"{best_by_auc['model_name']} ({best_by_auc['roc_auc']:.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

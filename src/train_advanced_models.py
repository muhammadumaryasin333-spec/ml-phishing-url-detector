"""Train audited Week 6 baselines, XGBoost, and LightGBM classifiers."""

from __future__ import annotations

import json
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
import xgboost
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

if __package__:
    from .audited_data import (
        AUDITED_SPLIT_PATHS,
        IDENTIFIER_COLUMNS,
        PROJECT_ROOT,
        QUARANTINED_FEATURES,
        RANDOM_STATE,
        SPLIT_MANIFEST_PATH,
        load_audited_splits,
        prepare_model_data,
        sha256_file,
    )
    from .train_baseline_models import build_models
else:
    from audited_data import (
        AUDITED_SPLIT_PATHS,
        IDENTIFIER_COLUMNS,
        PROJECT_ROOT,
        QUARANTINED_FEATURES,
        RANDOM_STATE,
        SPLIT_MANIFEST_PATH,
        load_audited_splits,
        prepare_model_data,
        sha256_file,
    )
    from train_baseline_models import build_models


RUNTIME_CACHE_DIR = Path(tempfile.gettempdir()) / "ml_phishing_url_detector_cache"
for cache_directory in [
    RUNTIME_CACHE_DIR,
    RUNTIME_CACHE_DIR / "matplotlib",
    RUNTIME_CACHE_DIR / "xdg",
]:
    cache_directory.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_CACHE_DIR / "xdg"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_DIR = PROJECT_ROOT / "reports" / "model_results"
ADVANCED_RESULTS_PATH = RESULTS_DIR / "advanced_results.csv"
TUNING_RESULTS_PATH = RESULTS_DIR / "advanced_tuning_results.csv"
COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"
REPORT_PATH = RESULTS_DIR / "advanced_training_report.md"
ADVANCED_CONFUSION_DIR = RESULTS_DIR / "advanced_confusion_matrices"
ADVANCED_ROC_DIR = RESULTS_DIR / "advanced_roc_curves"
ADVANCED_MODELS_DIR = PROJECT_ROOT / "models" / "advanced"
AUDITED_BASELINE_MODELS_DIR = PROJECT_ROOT / "models" / "audited_baseline"

PROTOCOL_ID = "stratified_normalized_host_group_70_15_15_v1"
POSITIVE_LABEL = 0
CLASS_LABELS = [0, 1]
N_JOBS = min(4, os.cpu_count() or 1)

XGBOOST_CANDIDATES: dict[str, dict[str, Any]] = {
    "xgb_01": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0,
        "reg_alpha": 0,
        "reg_lambda": 1,
    },
    "xgb_02": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0,
        "reg_alpha": 0,
        "reg_lambda": 1,
    },
    "xgb_03": {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_child_weight": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "gamma": 0,
        "reg_alpha": 0,
        "reg_lambda": 1,
    },
    "xgb_04": {
        "n_estimators": 200,
        "learning_rate": 0.10,
        "max_depth": 4,
        "min_child_weight": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "gamma": 0,
        "reg_alpha": 0,
        "reg_lambda": 1,
    },
}

LIGHTGBM_CANDIDATES: dict[str, dict[str, Any]] = {
    "lgb_01": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0,
        "reg_lambda": 1,
    },
    "lgb_02": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0,
        "reg_lambda": 1,
    },
    "lgb_03": {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 100,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_alpha": 0,
        "reg_lambda": 1,
    },
    "lgb_04": {
        "n_estimators": 200,
        "learning_rate": 0.10,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 1.0,
        "subsample_freq": 1,
        "colsample_bytree": 1.0,
        "reg_alpha": 0,
        "reg_lambda": 1,
    },
}


def create_output_directories() -> None:
    """Create Week 6 report and model output directories."""
    for directory in [
        RESULTS_DIR,
        ADVANCED_CONFUSION_DIR,
        ADVANCED_ROC_DIR,
        ADVANCED_MODELS_DIR,
        AUDITED_BASELINE_MODELS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def build_advanced_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build numeric and categorical transforms fitted only on supplied rows."""
    numeric_columns = X_train.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [
        column for column in X_train.columns if column not in numeric_columns
    ]
    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="missing"),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        ),
                    ]
                ),
                categorical_columns,
            )
        )
    if not transformers:
        raise ValueError("No features are available for advanced preprocessing.")
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
    )


def build_advanced_estimator(
    model_name: str, parameters: dict[str, Any]
) -> XGBClassifier | LGBMClassifier:
    """Construct one deterministic CPU estimator from a bounded candidate."""
    if model_name == "xgboost":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device="cpu",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            verbosity=0,
            **parameters,
        )
    if model_name == "lightgbm":
        return LGBMClassifier(
            objective="binary",
            metric="binary_logloss",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
            **parameters,
        )
    raise ValueError(f"Unsupported advanced model: {model_name}")


def positive_class_scores(model: Any, X: Any) -> np.ndarray:
    """Return phishing-class probabilities using the fitted class order."""
    classes = list(model.classes_)
    if POSITIVE_LABEL not in classes:
        raise ValueError(f"Positive class {POSITIVE_LABEL} is absent from {classes}.")
    return model.predict_proba(X)[:, classes.index(POSITIVE_LABEL)]


def calculate_full_metrics(
    y_true: pd.Series | np.ndarray,
    predicted: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float | int]:
    """Calculate stable overall, per-class, and confusion metrics."""
    y_array = np.asarray(y_true)
    precision, recall, f1_values, support = precision_recall_fscore_support(
        y_array,
        predicted,
        labels=CLASS_LABELS,
        zero_division=0,
    )
    matrix = confusion_matrix(y_array, predicted, labels=CLASS_LABELS)
    phishing_target = (y_array == POSITIVE_LABEL).astype(int)
    return {
        "accuracy": accuracy_score(y_array, predicted),
        "balanced_accuracy": balanced_accuracy_score(y_array, predicted),
        "precision_macro": precision_score(
            y_array, predicted, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_array, predicted, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(y_array, predicted, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_array, predicted),
        "roc_auc": roc_auc_score(phishing_target, scores),
        "phishing_precision": precision[0],
        "phishing_recall": recall[0],
        "phishing_f1": f1_values[0],
        "phishing_support": int(support[0]),
        "legitimate_precision": precision[1],
        "legitimate_recall": recall[1],
        "legitimate_f1": f1_values[1],
        "legitimate_support": int(support[1]),
        "cm_phishing_as_phishing": int(matrix[0, 0]),
        "cm_phishing_as_legitimate": int(matrix[0, 1]),
        "cm_legitimate_as_phishing": int(matrix[1, 0]),
        "cm_legitimate_as_legitimate": int(matrix[1, 1]),
    }


def evaluate_estimator(
    model: Any, X: Any, y: pd.Series
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray, float]:
    """Evaluate one fitted estimator and record prediction runtime."""
    start = time.perf_counter()
    predicted = model.predict(X)
    scores = positive_class_scores(model, X)
    predict_seconds = time.perf_counter() - start
    metrics = calculate_full_metrics(y, predicted, scores)
    return metrics, predicted, scores, predict_seconds


def selection_key(row: dict[str, Any]) -> tuple[float, float, float, str]:
    """Return deterministic validation-only selection values."""
    return (
        -float(row["f1_macro"]),
        -float(row["phishing_recall"]),
        -float(row["roc_auc"]),
        str(row["candidate_id"]),
    )


def select_best_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the strongest validation row without reading test metrics."""
    if not rows:
        raise ValueError("At least one validation candidate is required.")
    return min(rows, key=selection_key)


def tune_advanced_model(
    model_name: str,
    candidate_specs: dict[str, dict[str, Any]],
    X_train: Any,
    y_train: pd.Series,
    X_validation: Any,
    y_validation: pd.Series,
    input_feature_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Tune structural candidates and one balanced-weight ablation on validation."""
    result_rows = []
    structural_rows = []
    for candidate_id, parameters in candidate_specs.items():
        print(f"  {candidate_id}: unweighted")
        model = build_advanced_estimator(model_name, parameters)
        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        metrics, _, _, predict_seconds = evaluate_estimator(
            model, X_validation, y_validation
        )
        row = {
            "protocol_id": PROTOCOL_ID,
            "model_name": model_name,
            "candidate_id": candidate_id,
            "weighting": "none",
            "fit_scope": "train",
            "dataset_split": "validation",
            "is_selected": False,
            "n_train_rows": len(y_train),
            "n_eval_rows": len(y_validation),
            "n_input_features": input_feature_count,
            "positive_label": POSITIVE_LABEL,
            **metrics,
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "parameters_json": json.dumps(parameters, sort_keys=True),
        }
        result_rows.append(row)
        structural_rows.append(row)

    structural_winner = select_best_candidate(structural_rows)
    winning_parameters = candidate_specs[str(structural_winner["candidate_id"])]
    weighted_id = f"{structural_winner['candidate_id']}_balanced"
    print(f"  {weighted_id}: train-derived balanced sample weights")
    weighted_model = build_advanced_estimator(model_name, winning_parameters)
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    start = time.perf_counter()
    weighted_model.fit(X_train, y_train, sample_weight=sample_weights)
    fit_seconds = time.perf_counter() - start
    metrics, _, _, predict_seconds = evaluate_estimator(
        weighted_model, X_validation, y_validation
    )
    weighted_row = {
        "protocol_id": PROTOCOL_ID,
        "model_name": model_name,
        "candidate_id": weighted_id,
        "weighting": "balanced_sample_weight",
        "fit_scope": "train",
        "dataset_split": "validation",
        "is_selected": False,
        "n_train_rows": len(y_train),
        "n_eval_rows": len(y_validation),
        "n_input_features": input_feature_count,
        "positive_label": POSITIVE_LABEL,
        **metrics,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "parameters_json": json.dumps(winning_parameters, sort_keys=True),
    }
    result_rows.append(weighted_row)

    winner = select_best_candidate([structural_winner, weighted_row])
    for row in result_rows:
        row["is_selected"] = row["candidate_id"] == winner["candidate_id"]
    selected_config = {
        "candidate_id": winner["candidate_id"],
        "parameters": winning_parameters,
        "weighting": winner["weighting"],
    }
    return result_rows, winner, selected_config


def save_test_plots(
    model_name: str,
    y_true: pd.Series,
    predicted: np.ndarray,
    scores: np.ndarray,
    roc_auc: float,
) -> None:
    """Save test confusion matrix and ROC curve for an advanced model."""
    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        predicted,
        labels=CLASS_LABELS,
        display_labels=["phishing (0)", "legitimate (1)"],
        cmap="Blues",
        colorbar=False,
        ax=axis,
    )
    axis.set_title(f"{model_name.title()} - Audited Test Confusion Matrix")
    figure.tight_layout()
    figure.savefig(
        ADVANCED_CONFUSION_DIR / f"{model_name}_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=POSITIVE_LABEL)
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(fpr, tpr, linewidth=2, label=f"ROC-AUC = {roc_auc:.4f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="No skill")
    axis.set_title(f"{model_name.title()} - Audited Test ROC Curve")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        ADVANCED_ROC_DIR / f"{model_name}_roc_curve.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_advanced_artifacts(
    model_name: str,
    preprocessor: ColumnTransformer,
    model: Any,
    selected_config: dict[str, Any],
    features: dict[str, pd.DataFrame],
    final_fit_rows: int,
    transformed_feature_count: int,
    fit_seconds: float,
    predict_seconds: float,
) -> None:
    """Save a complete joblib bundle, native booster, and evidence manifest."""
    bundle_path = ADVANCED_MODELS_DIR / f"{model_name}.joblib"
    native_path = ADVANCED_MODELS_DIR / (
        f"{model_name}.ubj" if model_name == "xgboost" else f"{model_name}.txt"
    )
    bundle = {
        "protocol_id": PROTOCOL_ID,
        "preprocessor": preprocessor,
        "model": model,
        "positive_label": POSITIVE_LABEL,
        "class_labels": CLASS_LABELS,
        "input_feature_names": features["train"].columns.tolist(),
        "quarantined_features": sorted(QUARANTINED_FEATURES),
    }
    joblib.dump(bundle, bundle_path)
    if model_name == "xgboost":
        model.save_model(native_path)
    else:
        model.booster_.save_model(native_path)

    loaded_bundle = joblib.load(bundle_path)
    verification_rows = features["test"].iloc[:256]
    original_matrix = preprocessor.transform(verification_rows)
    loaded_matrix = loaded_bundle["preprocessor"].transform(verification_rows)
    expected = model.predict_proba(original_matrix)
    actual = loaded_bundle["model"].predict_proba(loaded_matrix)
    if not np.allclose(expected, actual, rtol=1e-12, atol=1e-12):
        raise RuntimeError(f"Saved {model_name} bundle failed prediction verification.")

    manifest = {
        "protocol_id": PROTOCOL_ID,
        "model_name": model_name,
        "selected_candidate": selected_config,
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS,
        "class_map": {"0": "phishing", "1": "legitimate"},
        "fit_rows": final_fit_rows,
        "input_feature_count": len(features["train"].columns),
        "transformed_feature_count": transformed_feature_count,
        "input_feature_names": features["train"].columns.tolist(),
        "input_feature_dtypes": {
            column: str(dtype) for column, dtype in features["train"].dtypes.items()
        },
        "identifier_columns_excluded": sorted(IDENTIFIER_COLUMNS),
        "quarantined_features": sorted(QUARANTINED_FEATURES),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__,
        },
        "platform": platform.platform(),
        "fit_seconds": fit_seconds,
        "test_predict_seconds": predict_seconds,
        "inputs": {
            name: {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(path),
            }
            for name, path in AUDITED_SPLIT_PATHS.items()
        },
        "split_manifest_sha256": sha256_file(SPLIT_MANIFEST_PATH),
        "artifacts": {
            "joblib": {
                "path": str(bundle_path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(bundle_path),
            },
            "native_supplementary": {
                "path": str(native_path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(native_path),
                "requires_external_preprocessor": True,
            },
        },
        "round_trip_prediction_rows_verified": len(verification_rows),
    }
    manifest_path = ADVANCED_MODELS_DIR / f"{model_name}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def fit_audited_baselines(
    features: dict[str, pd.DataFrame], targets: dict[str, pd.Series]
) -> list[dict[str, Any]]:
    """Retrain all baselines fairly under the audited protocol."""
    rows = []
    validation_models = build_models(features["train"], class_weight=None)
    for model_name, model in validation_models.items():
        print(f"  {model_name}: train -> validation")
        start = time.perf_counter()
        model.fit(features["train"], targets["train"])
        fit_seconds = time.perf_counter() - start
        metrics, _, _, predict_seconds = evaluate_estimator(
            model, features["validation"], targets["validation"]
        )
        rows.append(
            {
                "protocol_id": PROTOCOL_ID,
                "model_name": model_name,
                "candidate_id": "untuned_baseline",
                "weighting": "none",
                "fit_scope": "train",
                "dataset_split": "validation",
                "is_selected": True,
                "n_train_rows": len(targets["train"]),
                "n_eval_rows": len(targets["validation"]),
                "n_input_features": features["train"].shape[1],
                "positive_label": POSITIVE_LABEL,
                **metrics,
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "parameters_json": json.dumps(
                    model.named_steps["model"].get_params(), default=str, sort_keys=True
                ),
            }
        )

    combined_features = pd.concat(
        [features["train"], features["validation"]], ignore_index=True
    )
    combined_targets = pd.concat(
        [targets["train"], targets["validation"]], ignore_index=True
    )
    final_models = build_models(combined_features, class_weight=None)
    for model_name, model in final_models.items():
        print(f"  {model_name}: train+validation -> frozen test")
        start = time.perf_counter()
        model.fit(combined_features, combined_targets)
        fit_seconds = time.perf_counter() - start
        metrics, _, _, predict_seconds = evaluate_estimator(
            model, features["test"], targets["test"]
        )
        rows.append(
            {
                "protocol_id": PROTOCOL_ID,
                "model_name": model_name,
                "candidate_id": "untuned_baseline",
                "weighting": "none",
                "fit_scope": "train_validation",
                "dataset_split": "test",
                "is_selected": True,
                "n_train_rows": len(combined_targets),
                "n_eval_rows": len(targets["test"]),
                "n_input_features": combined_features.shape[1],
                "positive_label": POSITIVE_LABEL,
                **metrics,
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "parameters_json": json.dumps(
                    model.named_steps["model"].get_params(), default=str, sort_keys=True
                ),
            }
        )
        joblib.dump(model, AUDITED_BASELINE_MODELS_DIR / f"{model_name}.joblib")
    return rows


def format_markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    """Format selected result columns without another dependency."""
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def write_training_report(
    advanced_results: pd.DataFrame,
    comparison: pd.DataFrame,
    feature_count: int,
    transformed_feature_count: int,
) -> None:
    """Write the Week 6 protocol, tuning, results, and limitations."""
    selected_validation = advanced_results[
        advanced_results["dataset_split"] == "validation"
    ]
    comparison_view = comparison[
        [
            "model_name",
            "f1_macro",
            "roc_auc",
            "balanced_accuracy",
            "mcc",
            "phishing_recall",
            "rank_f1_macro",
        ]
    ]
    selected_view = selected_validation[
        ["model_name", "candidate_id", "weighting", "f1_macro", "roc_auc"]
    ]
    lines = [
        "# Week 6 Advanced Model Training Report",
        "",
        "## Outcome",
        "",
        "XGBoost and LightGBM were tuned using training and validation data under "
        "the normalized-host-group-disjoint protocol. Each locked model was then "
        "refitted on train plus validation and evaluated once on the frozen test set. "
        "The three Week 5 baselines were retrained under the same audited protocol.",
        "",
        "## Leakage-Audited Protocol",
        "",
        f"- Protocol: `{PROTOCOL_ID}`.",
        "- Split grouping: normalized host, with zero host and exact-URL overlap.",
        "- Split ratio: approximately 70% train, 15% validation, 15% test.",
        "- Test data did not participate in preprocessing, tuning, weighting, model "
        "selection, or threshold selection.",
        "- Legacy Week 5 scores are historical and not directly comparable.",
        "",
        "## Feature Handling",
        "",
        f"- Input feature columns: {feature_count}; transformed columns: "
        f"{transformed_feature_count}.",
        "- Excluded identifiers/raw text: "
        + ", ".join(f"`{name}`" for name in sorted(IDENTIFIER_COLUMNS))
        + ".",
        "- Quarantined unresolved derived features: "
        + ", ".join(f"`{name}`" for name in sorted(QUARANTINED_FEATURES))
        + ".",
        "- Numeric median imputation and TLD one-hot encoding were fitted on training "
        "rows only during tuning, then on train plus validation for final refitting.",
        "",
        "## Bounded Tuning",
        "",
        "- Four predeclared structural candidates per advanced model.",
        "- One train-derived balanced-sample-weight ablation for each structural winner.",
        "- Selection order: validation macro F1, phishing recall, ROC-AUC, then "
        "candidate ID as a deterministic tie-breaker.",
        "- No early stopping or open-ended search was used.",
        "",
        "### Selected validation configurations",
        "",
        format_markdown_table(selected_view, selected_view.columns.tolist()),
        "",
        "## Frozen Test Comparison",
        "",
        format_markdown_table(comparison_view, comparison_view.columns.tolist()),
        "",
        "Test ranks are descriptive only; they were not used to select or retune models.",
        "",
        "## Reproducibility Evidence",
        "",
        "- XGBoost 3.2.0 and LightGBM 4.7.0 are pinned in `requirements.txt`.",
        "- Model manifests record parameters, versions, split/input hashes, feature "
        "order, runtime, and artifact hashes.",
        "- Complete preprocessing-plus-model joblib bundles passed prediction "
        "round-trip verification; native boosters are supplementary only.",
        "",
        "## Limitations",
        "",
        "- Host-level grouping does not merge sibling subdomains under one registrable "
        "domain because no pinned Public Suffix List is included.",
        "- Very strong webpage-derived features may reflect collection bias; remaining "
        "features are valid for this dataset experiment, not proven deployment inputs.",
        "- Weighted probabilities are uncalibrated when a weighted candidate is selected.",
        "- This single-dataset evaluation does not establish temporal or cross-dataset "
        "generalisation.",
        "",
        "## Artifact Index",
        "",
        "- `reports/model_results/advanced_tuning_results.csv`",
        "- `reports/model_results/advanced_results.csv`",
        "- `reports/model_results/model_comparison.csv`",
        "- `models/advanced/` and `models/audited_baseline/`",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Run the complete leakage-audited Week 6 training workflow."""
    print("Week 6 audited advanced model training")
    print("[1/8] Loading and validating frozen audited splits")
    try:
        splits = load_audited_splits()
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}")
        print("Run first: python src/run_leakage_audit.py")
        return 1
    features, targets = prepare_model_data(splits)
    create_output_directories()

    print("[2/8] Fitting train-only advanced preprocessor")
    tuning_preprocessor = build_advanced_preprocessor(features["train"])
    X_train = tuning_preprocessor.fit_transform(features["train"])
    X_validation = tuning_preprocessor.transform(features["validation"])
    transformed_feature_count = X_train.shape[1]
    print(
        f"Input features: {features['train'].shape[1]}; "
        f"transformed features: {transformed_feature_count}"
    )

    print("[3/8] Tuning XGBoost on validation only")
    xgb_rows, xgb_validation_winner, xgb_config = tune_advanced_model(
        "xgboost",
        XGBOOST_CANDIDATES,
        X_train,
        targets["train"],
        X_validation,
        targets["validation"],
        features["train"].shape[1],
    )

    print("[4/8] Tuning LightGBM on validation only")
    lgb_rows, lgb_validation_winner, lgb_config = tune_advanced_model(
        "lightgbm",
        LIGHTGBM_CANDIDATES,
        X_train,
        targets["train"],
        X_validation,
        targets["validation"],
        features["train"].shape[1],
    )
    tuning_rows = [*xgb_rows, *lgb_rows]
    tuning_results = pd.DataFrame(tuning_rows)
    tuning_results.to_csv(TUNING_RESULTS_PATH, index=False, float_format="%.6f")

    print("[5/8] Locking configurations, refitting, and opening frozen test once")
    combined_features = pd.concat(
        [features["train"], features["validation"]], ignore_index=True
    )
    combined_targets = pd.concat(
        [targets["train"], targets["validation"]], ignore_index=True
    )
    final_preprocessor = build_advanced_preprocessor(combined_features)
    X_combined = final_preprocessor.fit_transform(combined_features)
    X_test = final_preprocessor.transform(features["test"])
    advanced_rows = [xgb_validation_winner, lgb_validation_winner]

    for model_name, selected_config in [
        ("xgboost", xgb_config),
        ("lightgbm", lgb_config),
    ]:
        print(f"  {model_name}: train+validation -> frozen test")
        model = build_advanced_estimator(model_name, selected_config["parameters"])
        fit_kwargs = {}
        if selected_config["weighting"] == "balanced_sample_weight":
            fit_kwargs["sample_weight"] = compute_sample_weight(
                class_weight="balanced", y=combined_targets
            )
        start = time.perf_counter()
        model.fit(X_combined, combined_targets, **fit_kwargs)
        fit_seconds = time.perf_counter() - start
        metrics, predicted, scores, predict_seconds = evaluate_estimator(
            model, X_test, targets["test"]
        )
        test_row = {
            "protocol_id": PROTOCOL_ID,
            "model_name": model_name,
            "candidate_id": selected_config["candidate_id"],
            "weighting": selected_config["weighting"],
            "fit_scope": "train_validation",
            "dataset_split": "test",
            "is_selected": True,
            "n_train_rows": len(combined_targets),
            "n_eval_rows": len(targets["test"]),
            "n_input_features": combined_features.shape[1],
            "positive_label": POSITIVE_LABEL,
            **metrics,
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "parameters_json": json.dumps(
                selected_config["parameters"], sort_keys=True
            ),
        }
        advanced_rows.append(test_row)
        save_test_plots(
            model_name, targets["test"], predicted, scores, float(metrics["roc_auc"])
        )
        save_advanced_artifacts(
            model_name=model_name,
            preprocessor=final_preprocessor,
            model=model,
            selected_config=selected_config,
            features=features,
            final_fit_rows=len(combined_targets),
            transformed_feature_count=X_combined.shape[1],
            fit_seconds=fit_seconds,
            predict_seconds=predict_seconds,
        )

    advanced_results = pd.DataFrame(advanced_rows)
    advanced_results.to_csv(ADVANCED_RESULTS_PATH, index=False, float_format="%.6f")

    print("[6/8] Retraining all baselines under the identical audited protocol")
    baseline_rows = fit_audited_baselines(features, targets)

    print("[7/8] Saving fair frozen-test comparison")
    baseline_test = pd.DataFrame(baseline_rows)
    baseline_test = baseline_test[baseline_test["dataset_split"] == "test"]
    advanced_test = advanced_results[advanced_results["dataset_split"] == "test"]
    comparison = pd.concat([baseline_test, advanced_test], ignore_index=True)
    comparison["rank_f1_macro"] = comparison["f1_macro"].rank(
        method="min", ascending=False
    ).astype(int)
    comparison["rank_roc_auc"] = comparison["roc_auc"].rank(
        method="min", ascending=False
    ).astype(int)
    comparison.to_csv(COMPARISON_PATH, index=False, float_format="%.6f")

    print("[8/8] Writing Week 6 report")
    write_training_report(
        advanced_results=advanced_results,
        comparison=comparison,
        feature_count=features["train"].shape[1],
        transformed_feature_count=X_combined.shape[1],
    )
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Comparison: {COMPARISON_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

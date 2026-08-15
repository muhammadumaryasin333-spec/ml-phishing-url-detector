"""Generate reproducible probability-space SHAP evidence for audited LightGBM."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any, Iterable

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn

if __package__:
    from .audited_data import (
        AUDITED_SPLIT_PATHS,
        MEMBERSHIP_COLUMN,
        PROJECT_ROOT,
        QUARANTINED_FEATURES,
        RANDOM_STATE,
        SPLIT_MANIFEST_PATH,
        TARGET_COLUMN,
        load_audited_splits,
        prepare_model_data,
        sha256_file,
    )
else:
    from audited_data import (
        AUDITED_SPLIT_PATHS,
        MEMBERSHIP_COLUMN,
        PROJECT_ROOT,
        QUARANTINED_FEATURES,
        RANDOM_STATE,
        SPLIT_MANIFEST_PATH,
        TARGET_COLUMN,
        load_audited_splits,
        prepare_model_data,
        sha256_file,
    )


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


ADVANCED_MODELS_DIR = PROJECT_ROOT / "models" / "advanced"
MODEL_NAME = "lightgbm"
BUNDLE_PATH = ADVANCED_MODELS_DIR / f"{MODEL_NAME}.joblib"
MODEL_MANIFEST_PATH = ADVANCED_MODELS_DIR / f"{MODEL_NAME}_manifest.json"
EXPLANATIONS_DIR = PROJECT_ROOT / "reports" / "explainability"

BACKGROUND_ROWS = 128
EXPLANATION_ROWS = 256
ADDITIVITY_TOLERANCE = 1e-6
PHISHING_LABEL = 0
LEGITIMATE_LABEL = 1


def hash_strings(values: Iterable[object]) -> str:
    """Hash an ordered sequence without placing raw identifiers in reports."""
    encoded = "\n".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def deterministic_sample_rows(
    frame: pd.DataFrame, count: int, *, seed: int
) -> pd.DataFrame:
    """Return a reproducible bounded sample independent of input row order."""
    if count <= 0:
        raise ValueError("Sample count must be positive.")
    if MEMBERSHIP_COLUMN not in frame:
        raise ValueError(f"Missing immutable row identifier: {MEMBERSHIP_COLUMN}")
    if frame[MEMBERSHIP_COLUMN].astype("string").duplicated().any():
        raise ValueError("Immutable row identifiers must be unique before sampling.")

    ordered = frame.sort_values(MEMBERSHIP_COLUMN, kind="stable").reset_index(
        drop=True
    )
    selected_count = min(count, len(ordered))
    rng = np.random.default_rng(seed)
    if TARGET_COLUMN in ordered and selected_count >= 2:
        labels = set(ordered[TARGET_COLUMN].dropna().astype(int))
        if labels != {PHISHING_LABEL, LEGITIMATE_LABEL}:
            raise ValueError("Explanation sampling requires both target classes.")
        phishing_rows = ordered.index[
            ordered[TARGET_COLUMN].eq(PHISHING_LABEL)
        ].to_numpy()
        phishing_count = round(selected_count * len(phishing_rows) / len(ordered))
        phishing_count = max(1, min(len(phishing_rows), phishing_count))
        legitimate_count = selected_count - phishing_count
        legitimate_rows = ordered.index[
            ordered[TARGET_COLUMN].eq(LEGITIMATE_LABEL)
        ].to_numpy()
        if legitimate_count < 1 or legitimate_count > len(legitimate_rows):
            raise ValueError("Explanation sample cannot preserve both target classes.")
        positions = np.sort(
            np.concatenate(
                [
                    rng.choice(phishing_rows, size=phishing_count, replace=False),
                    rng.choice(legitimate_rows, size=legitimate_count, replace=False),
                ]
            )
        )
    else:
        positions = np.sort(
            rng.choice(len(ordered), size=selected_count, replace=False)
        )
    return ordered.iloc[positions].reset_index(drop=True)


def select_local_cases(cohort: pd.DataFrame) -> list[tuple[str, int]]:
    """Choose representative and error cases by a fixed post-hoc rule."""
    required = {"actual_label", "predicted_label", "phishing_probability"}
    if not required <= set(cohort.columns):
        raise ValueError("Local-case cohort is missing prediction columns.")
    candidates: list[tuple[str, int]] = []
    for label, name in [
        (PHISHING_LABEL, "representative_phishing"),
        (LEGITIMATE_LABEL, "representative_legitimate"),
    ]:
        positions = cohort.index[cohort["actual_label"].eq(label)].tolist()
        if not positions:
            raise ValueError(f"Explanation cohort has no class {label} row.")
        candidates.append((name, int(positions[0])))
    boundary_position = int(
        (cohort["phishing_probability"] - 0.5).abs().idxmin()
    )
    candidates.append(("closest_to_boundary", boundary_position))
    for actual, predicted, name in [
        (PHISHING_LABEL, LEGITIMATE_LABEL, "false_negative"),
        (LEGITIMATE_LABEL, PHISHING_LABEL, "false_positive"),
    ]:
        positions = cohort.index[
            cohort["actual_label"].eq(actual)
            & cohort["predicted_label"].eq(predicted)
        ].tolist()
        if positions:
            candidates.append((name, int(positions[0])))
    return candidates


def normalise_probability_shap_values(
    values: Any, *, rows: int, columns: int, class_index: int
) -> np.ndarray:
    """Resolve SHAP 0.51 binary output variants to one class contribution matrix."""
    if isinstance(values, list):
        if len(values) != 2:
            raise ValueError("Expected exactly two class SHAP arrays.")
        matrix = np.asarray(values[class_index])
    else:
        array = np.asarray(values)
        if array.shape == (rows, columns):
            matrix = array
        elif array.shape == (rows, columns, 2):
            matrix = array[:, :, class_index]
        elif array.shape == (2, rows, columns):
            matrix = array[class_index]
        else:
            raise ValueError(
                "Unexpected SHAP output shape; expected (rows, columns), "
                "(rows, columns, 2), or (2, rows, columns), got "
                f"{array.shape}."
            )
    if matrix.shape != (rows, columns):
        raise ValueError(
            f"Resolved SHAP values have shape {matrix.shape}, expected {(rows, columns)}."
        )
    if not np.isfinite(matrix).all():
        raise ValueError("SHAP values contain non-finite values.")
    return matrix.astype(float, copy=False)


def normalise_expected_value(value: Any, *, class_index: int) -> float:
    """Resolve SHAP expected-value variants for the selected class."""
    array = np.asarray(value, dtype=float)
    if array.ndim == 0 or array.size == 1:
        result = float(array.reshape(-1)[0])
    elif array.shape == (2,):
        result = float(array[class_index])
    else:
        raise ValueError(f"Unexpected SHAP expected-value shape: {array.shape}.")
    if not np.isfinite(result):
        raise ValueError("SHAP expected value is non-finite.")
    return result


def build_feature_mapping(preprocessor: Any) -> pd.DataFrame:
    """Map every transformed column exactly once to its input feature group."""
    transformed_names = list(preprocessor.get_feature_names_out())
    rows: list[dict[str, str]] = []
    for transformer_name, transformer, columns in preprocessor.transformers_:
        if transformer_name == "remainder" or transformer == "drop":
            continue
        output_slice = preprocessor.output_indices_.get(transformer_name)
        if output_slice is None:
            raise ValueError(f"Missing output slice for transformer {transformer_name!r}.")
        output_names = transformed_names[output_slice]
        input_columns = list(columns)
        if transformer_name == "numeric":
            if len(output_names) != len(input_columns):
                raise ValueError("Numeric transformer changed column cardinality.")
            pairs = zip(output_names, input_columns, strict=True)
        elif transformer_name == "categorical":
            encoder = transformer.named_steps.get("encoder")
            if encoder is None:
                raise ValueError("Categorical transformer has no fitted encoder.")
            grouped_columns = [
                column
                for column, categories in zip(
                    input_columns, encoder.categories_, strict=True
                )
                for _ in categories
            ]
            if len(output_names) != len(grouped_columns):
                raise ValueError("Categorical encoded-column count is inconsistent.")
            pairs = zip(output_names, grouped_columns, strict=True)
        else:
            raise ValueError(f"Unsupported fitted transformer {transformer_name!r}.")
        for transformed_feature, original_feature in pairs:
            rows.append(
                {
                    "transformed_feature": str(transformed_feature),
                    "original_feature": str(original_feature),
                    "transformer": transformer_name,
                }
            )
    mapping = pd.DataFrame(rows)
    if len(mapping) != len(transformed_names):
        raise ValueError("Every transformed feature must map exactly once.")
    if mapping["transformed_feature"].duplicated().any():
        raise ValueError("A transformed feature maps to more than one input feature.")
    if mapping["transformed_feature"].tolist() != transformed_names:
        raise ValueError("Feature mapping order differs from fitted preprocessor output.")
    return mapping


def aggregate_feature_contributions(
    phishing_values: np.ndarray, mapping: pd.DataFrame
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Sum signed dummy contributions before ranking original feature groups."""
    if phishing_values.ndim != 2 or phishing_values.shape[1] != len(mapping):
        raise ValueError("SHAP matrix and feature mapping dimensions differ.")
    original_features = mapping["original_feature"].drop_duplicates().tolist()
    aggregated = np.empty((phishing_values.shape[0], len(original_features)), dtype=float)
    summary_rows = []
    for position, feature in enumerate(original_features):
        source_columns = np.flatnonzero(
            mapping["original_feature"].to_numpy() == feature
        )
        signed_values = phishing_values[:, source_columns].sum(axis=1)
        aggregated[:, position] = signed_values
        summary_rows.append(
            {
                "feature": feature,
                "transformed_column_count": int(len(source_columns)),
                "mean_abs_shap_value_phishing": float(np.abs(signed_values).mean()),
                "mean_signed_shap_value_phishing": float(signed_values.mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["mean_abs_shap_value_phishing", "feature"],
        ascending=[False, True],
        kind="stable",
    )
    return aggregated, original_features, summary.reset_index(drop=True)


def assert_model_inputs(
    bundle: dict[str, Any],
    manifest: dict[str, Any],
    features: dict[str, pd.DataFrame],
) -> None:
    """Fail closed when the saved model, split files, or feature contract differ."""
    if manifest.get("model_name") != MODEL_NAME:
        raise ValueError(
            "Explanation manifest does not identify the locked LightGBM model."
        )
    expected_bundle_hash = manifest.get("artifacts", {}).get("joblib", {}).get(
        "sha256"
    )
    if (
        not isinstance(expected_bundle_hash, str)
        or sha256_file(BUNDLE_PATH) != expected_bundle_hash
    ):
        raise ValueError("Saved LightGBM bundle hash does not match its manifest.")
    if manifest.get("split_manifest_sha256") != sha256_file(SPLIT_MANIFEST_PATH):
        raise ValueError(
            "Audited split manifest hash does not match the model manifest."
        )
    for split_name, path in AUDITED_SPLIT_PATHS.items():
        expected_hash = manifest.get("inputs", {}).get(split_name, {}).get("sha256")
        if not isinstance(expected_hash, str) or sha256_file(path) != expected_hash:
            raise ValueError(
                f"{split_name} split hash does not match the model manifest."
            )

    expected_columns = list(manifest.get("input_feature_names", []))
    if not expected_columns:
        raise ValueError("Model manifest has no input feature contract.")
    if bundle.get("input_feature_names") != expected_columns:
        raise ValueError(
            "Joblib bundle input feature order differs from model manifest."
        )
    if bundle.get("quarantined_features") != manifest.get("quarantined_features"):
        raise ValueError(
            "Joblib bundle quarantine policy differs from model manifest."
        )
    if sorted(bundle["quarantined_features"]) != sorted(QUARANTINED_FEATURES):
        raise ValueError("Runtime quarantine policy differs from the locked model.")
    if bundle.get("positive_label") != PHISHING_LABEL:
        raise ValueError("Locked model must define phishing as class 0.")
    if bundle.get("class_labels") != [PHISHING_LABEL, LEGITIMATE_LABEL]:
        raise ValueError("Locked class labels must be [0, 1].")
    if manifest.get("class_map") != {"0": "phishing", "1": "legitimate"}:
        raise ValueError("Model manifest class map is not the locked phishing contract.")
    for split_name, split_features in features.items():
        if split_features.columns.tolist() != expected_columns:
            raise ValueError(
                f"{split_name} input features differ from the model contract."
            )

    model = bundle.get("model")
    classes = getattr(model, "classes_", None)
    if np.asarray(classes).tolist() != [PHISHING_LABEL, LEGITIMATE_LABEL]:
        raise ValueError("Estimator class order must be exactly [0, 1].")
    if bundle.get("preprocessor") is None:
        raise ValueError("Saved model bundle has no fitted preprocessor.")


def save_global_plots(
    aggregated_values: np.ndarray,
    feature_names: list[str],
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save group-aggregated global SHAP bar and beeswarm evidence."""
    top_features = summary.head(20).iloc[::-1]
    figure, axis = plt.subplots(figsize=(9, 7))
    axis.barh(
        top_features["feature"], top_features["mean_abs_shap_value_phishing"],
        color="#b91c1c",
    )
    axis.set_xlabel("Mean |SHAP contribution to phishing probability|")
    axis.set_title("LightGBM global feature importance — phishing class (0)")
    figure.tight_layout()
    figure.savefig(
        output_dir / "shap_bar_plot.png", dpi=300, bbox_inches="tight"
    )
    plt.close(figure)

    ranking = summary.head(15)["feature"].tolist()
    indices = [feature_names.index(feature) for feature in ranking]
    rng = np.random.default_rng(RANDOM_STATE)
    figure, axis = plt.subplots(figsize=(10, 7))
    for position, (feature, index) in enumerate(zip(ranking, indices, strict=True)):
        jitter = rng.uniform(-0.28, 0.28, size=aggregated_values.shape[0])
        axis.scatter(
            aggregated_values[:, index],
            np.full(aggregated_values.shape[0], position) + jitter,
            alpha=0.35,
            s=12,
            color="#1d4ed8",
            linewidths=0,
        )
    axis.axvline(0, color="#52525b", linewidth=1)
    axis.set_yticks(range(len(ranking)), ranking)
    axis.invert_yaxis()
    axis.set_xlabel("SHAP contribution to phishing probability")
    axis.set_title("LightGBM global SHAP beeswarm — original feature groups")
    figure.tight_layout()
    figure.savefig(output_dir / "shap_summary_plot.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_local_waterfalls(
    shap_module: Any,
    aggregated_values: np.ndarray,
    feature_names: list[str],
    phishing_base_value: float,
    cohort: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Save predeclared, non-cherry-picked local phishing-class waterfalls."""
    local_cases = select_local_cases(cohort)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for case_name, position in local_cases:
        values = aggregated_values[position]
        reconstructed = float(phishing_base_value + values.sum())
        expected_probability = float(cohort.iloc[position]["phishing_probability"])
        error = abs(reconstructed - expected_probability)
        explanation = shap_module.Explanation(
            values=values,
            base_values=phishing_base_value,
            feature_names=feature_names,
        )
        shap_module.plots.waterfall(explanation, max_display=15, show=False)
        figure = plt.gcf()
        figure.suptitle(
            f"Phishing probability explanation — {case_name.replace('_', ' ')}",
            y=1.02,
        )
        figure.tight_layout()
        figure.savefig(
            output_dir / f"{case_name}_waterfall.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(figure)
        for feature, contribution in zip(feature_names, values, strict=True):
            output_rows.append(
                {
                    "case_id": case_name,
                    "local_position": position,
                    "member_id_sha256": cohort.iloc[position]["member_id_sha256"],
                    "actual_label": int(cohort.iloc[position]["actual_label"]),
                    "phishing_probability": expected_probability,
                    "base_probability": phishing_base_value,
                    "reconstructed_probability": reconstructed,
                    "additivity_error": error,
                    "feature": feature,
                    "shap_value_phishing": float(contribution),
                }
            )
    return pd.DataFrame(output_rows)


def write_notes(output_dir: Path) -> None:
    """State interpretation limits alongside the saved evidence."""
    output_dir.joinpath("explainability_notes.md").write_text(
        "# SHAP interpretation notes\n\n"
        "These values are model attributions for phishing probability (class 0), "
        "relative to a deterministic train-plus-validation background. They are not "
        "causal evidence or proof that a feature causes phishing.\n\n"
        "TLD one-hot contributions are summed with their signs before feature ranking. "
        "No raw URLs or unredacted row identifiers are written to this directory.\n\n"
        "The model remains dataset-specific: webpage-derived proxy features and the "
        "single-dataset audited evaluation do not establish deployment performance.\n",
        encoding="utf-8",
    )


def generate_explanations(output_dir: Path = EXPLANATIONS_DIR) -> dict[str, Any]:
    """Verify the locked model and save bounded, reproducible SHAP evidence."""
    try:
        import shap
    except ImportError as error:
        raise RuntimeError(
            "SHAP 0.51.0 is required. Install the project dependencies before "
            "running explanations."
        ) from error

    if not BUNDLE_PATH.exists() or not MODEL_MANIFEST_PATH.exists():
        raise FileNotFoundError("Locked LightGBM bundle or manifest is missing.")
    manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    bundle = joblib.load(BUNDLE_PATH)
    splits = load_audited_splits()
    features, _targets = prepare_model_data(splits)
    assert_model_inputs(bundle, manifest, features)

    output_dir.mkdir(parents=True, exist_ok=True)
    fit_rows = pd.concat([splits["train"], splits["validation"]], ignore_index=True)
    background_rows = deterministic_sample_rows(
        fit_rows, BACKGROUND_ROWS, seed=RANDOM_STATE
    )
    test_rows = deterministic_sample_rows(
        splits["test"], EXPLANATION_ROWS, seed=RANDOM_STATE + 1
    )
    background_features = background_rows.loc[:, bundle["input_feature_names"]]
    explanation_features = test_rows.loc[:, bundle["input_feature_names"]]
    preprocessor = bundle["preprocessor"]
    model = bundle["model"]
    background_matrix = preprocessor.transform(background_features)
    explanation_matrix = preprocessor.transform(explanation_features)
    background_dense = (
        background_matrix.toarray()
        if hasattr(background_matrix, "toarray")
        else np.asarray(background_matrix)
    )
    explanation_dense = (
        explanation_matrix.toarray()
        if hasattr(explanation_matrix, "toarray")
        else np.asarray(explanation_matrix)
    )
    if not np.isfinite(background_dense).all() or not np.isfinite(
        explanation_dense
    ).all():
        raise ValueError("Transformed model inputs contain non-finite values.")

    transformed_names = list(preprocessor.get_feature_names_out())
    if explanation_dense.shape[1] != len(transformed_names):
        raise ValueError("Transformed explanation matrix differs from feature names.")
    class_index = list(model.classes_).index(LEGITIMATE_LABEL)
    explainer = shap.TreeExplainer(
        model,
        data=background_dense,
        feature_perturbation="interventional",
        model_output="probability",
    )
    legitimate_values = normalise_probability_shap_values(
        explainer.shap_values(explanation_dense),
        rows=len(explanation_dense),
        columns=len(transformed_names),
        class_index=class_index,
    )
    legitimate_base_value = normalise_expected_value(
        explainer.expected_value, class_index=class_index
    )
    phishing_values = -legitimate_values
    phishing_base_value = 1.0 - legitimate_base_value
    probabilities = model.predict_proba(explanation_matrix)
    phishing_index = list(model.classes_).index(PHISHING_LABEL)
    phishing_probabilities = probabilities[:, phishing_index]
    reconstructed = phishing_base_value + phishing_values.sum(axis=1)
    errors = np.abs(reconstructed - phishing_probabilities)
    maximum_error = float(errors.max())
    if maximum_error > ADDITIVITY_TOLERANCE:
        raise RuntimeError(
            f"Probability-space SHAP additivity failed: {maximum_error:.3e} exceeds "
            f"{ADDITIVITY_TOLERANCE:.1e}."
        )

    mapping = build_feature_mapping(preprocessor)
    aggregated_values, original_features, summary = aggregate_feature_contributions(
        phishing_values, mapping
    )
    cohort = pd.DataFrame(
        {
            "member_id_sha256": test_rows[MEMBERSHIP_COLUMN].map(
                lambda value: hash_strings([value])
            ),
            "actual_label": test_rows[TARGET_COLUMN].to_numpy(),
            "predicted_label": model.predict(explanation_matrix),
            "phishing_probability": phishing_probabilities,
            "reconstructed_probability": reconstructed,
            "additivity_error": errors,
        }
    )

    mapping.to_csv(output_dir / "feature_transformation_map.csv", index=False)
    summary.to_csv(output_dir / "shap_global_importance.csv", index=False)
    cohort.to_csv(output_dir / "explanation_cohort.csv", index=False)
    save_global_plots(aggregated_values, original_features, summary, output_dir)
    local_output_dir = output_dir / "local_explanations"
    local = save_local_waterfalls(
        shap,
        aggregated_values,
        original_features,
        phishing_base_value,
        cohort,
        local_output_dir,
    )
    local.to_csv(local_output_dir / "local_cases.csv", index=False)
    write_notes(output_dir)

    generated_manifest = {
        "model_name": MODEL_NAME,
        "model_bundle_sha256": sha256_file(BUNDLE_PATH),
        "model_manifest_sha256": sha256_file(MODEL_MANIFEST_PATH),
        "split_manifest_sha256": sha256_file(SPLIT_MANIFEST_PATH),
        "input_feature_names_sha256": hash_strings(bundle["input_feature_names"]),
        "quarantined_features_sha256": hash_strings(
            sorted(bundle["quarantined_features"])
        ),
        "class_semantics": {
            "explainer_output": "legitimate probability (class 1)",
            "reported_output": "phishing probability (class 0)",
            "conversion": (
                "phishing_base=1-legitimate_base; "
                "phishing_shap=-legitimate_shap"
            ),
        },
        "sampling": {
            "background": {
                "source": "train_plus_validation",
                "rows": len(background_rows),
                "seed": RANDOM_STATE,
                "membership_sha256": hash_strings(background_rows[MEMBERSHIP_COLUMN]),
            },
            "explanation": {
                "source": "frozen_test_after_model_lock",
                "rows": len(test_rows),
                "seed": RANDOM_STATE + 1,
                "membership_sha256": hash_strings(test_rows[MEMBERSHIP_COLUMN]),
            },
            "local_cases": local[["case_id", "local_position"]]
            .drop_duplicates()
            .to_dict(orient="records"),
        },
        "feature_counts": {
            "input": len(bundle["input_feature_names"]),
            "transformed": len(transformed_names),
            "aggregated": len(original_features),
        },
        "additivity": {
            "space": "phishing_probability",
            "tolerance": ADDITIVITY_TOLERANCE,
            "maximum_absolute_error": maximum_error,
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "shap": shap.__version__,
        },
        "privacy": "Raw URLs and unredacted row identifiers are not written.",
    }
    (output_dir / "explanation_manifest.json").write_text(
        json.dumps(generated_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return generated_manifest


def main() -> int:
    """Generate the Week 8 LightGBM explanation artifacts."""
    manifest = generate_explanations()
    print(f"SHAP explanations: {EXPLANATIONS_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Maximum probability additivity error: {manifest['additivity']['maximum_absolute_error']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

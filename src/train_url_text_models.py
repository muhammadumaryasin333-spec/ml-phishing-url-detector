"""Train leakage-safe URL-text classifiers on the audited split protocol.

This module never fetches or opens a submitted URL.  It treats URL values as
plain text and keeps the audited test split outside tuning and selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

if __package__:
    from .audited_data import (
        AUDITED_SPLIT_PATHS,
        MEMBERSHIP_COLUMN,
        PROJECT_ROOT,
        RANDOM_STATE,
        SPLIT_MANIFEST_PATH,
        TARGET_COLUMN,
        load_audited_splits,
        sha256_file,
        validate_audited_splits,
    )
else:
    from audited_data import (
        AUDITED_SPLIT_PATHS,
        MEMBERSHIP_COLUMN,
        PROJECT_ROOT,
        RANDOM_STATE,
        SPLIT_MANIFEST_PATH,
        TARGET_COLUMN,
        load_audited_splits,
        sha256_file,
        validate_audited_splits,
    )


PROTOCOL_ID = "stratified_normalized_host_group_70_15_15_v1"
POSITIVE_LABEL = 0
CLASS_LABELS = [0, 1]
URL_COLUMN = "URL"

RESULTS_DIR = PROJECT_ROOT / "reports" / "transformer_experiments"
MODELS_DIR = PROJECT_ROOT / "models" / "url_text"
EMBEDDING_CACHE_DIR = PROJECT_ROOT / "models" / ".cache" / "url_text_embeddings"
TUNING_RESULTS_PATH = RESULTS_DIR / "url_text_tuning_results.csv"
RESULTS_PATH = RESULTS_DIR / "url_text_results.csv"
COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"
REPORT_PATH = RESULTS_DIR / "url_text_training_report.md"
TABULAR_COMPARISON_PATH = PROJECT_ROOT / "reports" / "model_results" / "model_comparison.csv"

TFIDF_VECTORISER_CONFIG: dict[str, Any] = {
    "analyzer": "char_wb",
    "ngram_range": [3, 5],
    "min_df": 2,
    "max_features": 150000,
    "sublinear_tf": True,
    "lowercase": False,
    "norm": "l2",
    "dtype": "float32",
}
TFIDF_CANDIDATES: dict[str, dict[str, float]] = {
    "tfidf_lr_c_0_25": {"C": 0.25},
    "tfidf_lr_c_1_00": {"C": 1.0},
    "tfidf_lr_c_4_00": {"C": 4.0},
}

SENTENCE_TRANSFORMER_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
SENTENCE_TRANSFORMER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
SENTENCE_TRANSFORMER_LICENSE = "apache-2.0"
SENTENCE_TRANSFORMER_DIMENSIONS = 384
SENTENCE_TRANSFORMER_MAX_WORDPIECES = 256
SENTENCE_TRANSFORMER_BATCH_SIZE = 128
SENTENCE_TRANSFORMER_CONFIG: dict[str, Any] = {
    "model_id": SENTENCE_TRANSFORMER_MODEL_ID,
    "revision": SENTENCE_TRANSFORMER_REVISION,
    "license": SENTENCE_TRANSFORMER_LICENSE,
    "language_scope": "English",
    "intended_use": "general sentence and short-paragraph embeddings",
    "embedding_dimensions": SENTENCE_TRANSFORMER_DIMENSIONS,
    "max_wordpieces": SENTENCE_TRANSFORMER_MAX_WORDPIECES,
    "batch_size": SENTENCE_TRANSFORMER_BATCH_SIZE,
    "normalize_embeddings": True,
    "precision": "float32",
    "trust_remote_code": False,
}
EMBEDDING_CANDIDATES: dict[str, dict[str, float]] = {
    "sentence_transformer_lr_c_0_25": {"C": 0.25},
    "sentence_transformer_lr_c_1_00": {"C": 1.0},
    "sentence_transformer_lr_c_4_00": {"C": 4.0},
}


class UrlEncoder(Protocol):
    """Minimum SentenceTransformer interface used by the cached encoder path."""

    max_seq_length: int

    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray: ...


def build_tfidf_vectorizer() -> TfidfVectorizer:
    """Build the fixed sparse URL character representation."""
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=150000,
        sublinear_tf=True,
        lowercase=False,
        norm="l2",
        dtype=np.float32,
    )


def build_logistic_regression(C: float) -> LogisticRegression:
    """Build a deterministic binary classifier for one bounded C candidate."""
    return LogisticRegression(
        C=C,
        class_weight=None,
        solver="liblinear",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )


def extract_url_texts(split: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return ordered URL texts and immutable row IDs, failing closed on bad rows."""
    required_columns = {URL_COLUMN, MEMBERSHIP_COLUMN, TARGET_COLUMN}
    missing_columns = sorted(required_columns - set(split.columns))
    if missing_columns:
        raise ValueError("Missing URL-text columns: " + ", ".join(missing_columns))
    if split[URL_COLUMN].isna().any():
        raise ValueError("URL text must not contain missing values.")
    urls = split[URL_COLUMN].astype("string").str.strip()
    if urls.eq("").any():
        raise ValueError("URL text must not contain empty values.")
    row_ids = split[MEMBERSHIP_COLUMN].astype("string")
    if row_ids.isna().any() or row_ids.duplicated().any():
        raise ValueError("URL-text membership identifiers must be present and unique.")
    return urls.reset_index(drop=True), row_ids.reset_index(drop=True)


def hash_url_rows(row_ids: pd.Series, urls: pd.Series) -> str:
    """Hash ordered row IDs and URL text without persisting URL text in metadata."""
    if len(row_ids) != len(urls):
        raise ValueError("Cannot hash URL text with mismatched row-ID length.")
    digest = hashlib.sha256()
    for row_id, url in zip(row_ids.astype("string"), urls.astype("string"), strict=True):
        digest.update(json.dumps([str(row_id), str(url)], ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def hash_row_ids(row_ids: pd.Series) -> str:
    """Hash ordered identifiers used to detect cache row-order mismatch."""
    digest = hashlib.sha256()
    for row_id in row_ids.astype("string"):
        digest.update(str(row_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def select_best_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select exclusively from validation metrics with a deterministic tie-breaker."""
    if not rows:
        raise ValueError("At least one validation candidate is required.")
    return min(
        rows,
        key=lambda row: (
            -float(row["f1_macro"]),
            -float(row["phishing_recall"]),
            -float(row["roc_auc"]),
            str(row["candidate_id"]),
        ),
    )


def positive_class_scores(model: LogisticRegression, X: Any) -> np.ndarray:
    """Return phishing probabilities, where label 0 is the phishing class."""
    classes = list(model.classes_)
    if POSITIVE_LABEL not in classes:
        raise ValueError(f"Phishing class {POSITIVE_LABEL} is absent from {classes}.")
    return model.predict_proba(X)[:, classes.index(POSITIVE_LABEL)]


def calculate_full_metrics(
    y_true: pd.Series | np.ndarray,
    predicted: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float | int]:
    """Calculate fixed, class-aware metrics with an explicit class order."""
    y_array = np.asarray(y_true)
    precision, recall, f1_values, support = precision_recall_fscore_support(
        y_array, predicted, labels=CLASS_LABELS, zero_division=0
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


def evaluate_classifier(
    model: LogisticRegression, X: Any, y: pd.Series
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray, float]:
    """Evaluate one fitted downstream classifier without changing its threshold."""
    started_at = time.perf_counter()
    predicted = model.predict(X)
    scores = positive_class_scores(model, X)
    predict_seconds = time.perf_counter() - started_at
    return calculate_full_metrics(y, predicted, scores), predicted, scores, predict_seconds


def sample_split(split: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    """Make a deterministic class-stratified smoke subset without new split membership."""
    if max_rows < 2:
        raise ValueError("Smoke sampling requires at least two rows.")
    if max_rows >= len(split):
        return split.copy().reset_index(drop=True)
    class_counts = split[TARGET_COLUMN].value_counts().sort_index()
    if set(class_counts.index) != {0, 1}:
        raise ValueError("Smoke sampling requires both audited target classes.")
    desired_zero = round(max_rows * int(class_counts[0]) / len(split))
    desired_zero = max(1, min(int(class_counts[0]), desired_zero))
    desired_one = max_rows - desired_zero
    if desired_one < 1 or desired_one > int(class_counts[1]):
        raise ValueError("Smoke row count cannot preserve both target classes.")
    rng = np.random.default_rng(seed)
    selected = []
    for label, count in [(0, desired_zero), (1, desired_one)]:
        candidates = split.index[split[TARGET_COLUMN] == label].to_numpy()
        selected.extend(rng.choice(candidates, size=count, replace=False).tolist())
    return split.loc[sorted(selected)].reset_index(drop=True)


def tune_logistic_candidates(
    representation: str,
    candidates: dict[str, dict[str, float]],
    X_train: Any,
    y_train: pd.Series,
    X_validation: Any,
    y_validation: pd.Series,
    feature_count: int,
    representation_parameters: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run bounded downstream tuning on train/validation only."""
    rows = []
    for candidate_id, parameters in candidates.items():
        classifier = build_logistic_regression(parameters["C"])
        started_at = time.perf_counter()
        classifier.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started_at
        metrics, _, _, predict_seconds = evaluate_classifier(
            classifier, X_validation, y_validation
        )
        rows.append(
            {
                "protocol_id": PROTOCOL_ID,
                "representation": representation,
                "candidate_id": candidate_id,
                "fit_scope": "train",
                "dataset_split": "validation",
                "is_selected": False,
                "n_train_rows": len(y_train),
                "n_eval_rows": len(y_validation),
                "n_features": feature_count,
                "positive_label": POSITIVE_LABEL,
                **metrics,
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "parameters_json": json.dumps(
                    {"classifier": parameters, "representation": representation_parameters},
                    sort_keys=True,
                ),
            }
        )
    winner = select_best_candidate(rows)
    for row in rows:
        row["is_selected"] = row["candidate_id"] == winner["candidate_id"]
    return rows, winner


def _cache_key(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def embedding_cache_paths(cache_root: Path, split_name: str) -> dict[str, Path]:
    """Return deterministic, revision-scoped cache locations for one split."""
    key = _cache_key(SENTENCE_TRANSFORMER_CONFIG)
    directory = cache_root / f"all_minilm_l6_v2_{key}" / split_name
    return {
        "directory": directory,
        "matrix": directory / "embeddings.npy",
        "row_ids": directory / "row_ids.npy",
        "metadata": directory / "metadata.json",
    }


def _cache_metadata(
    split_name: str,
    row_ids: pd.Series,
    urls: pd.Series,
    matrix: np.ndarray,
    device: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "split_name": split_name,
        "input_sha256": hash_url_rows(row_ids, urls),
        "row_ids_sha256": hash_row_ids(row_ids),
        "row_count": len(row_ids),
        "embedding_shape": list(matrix.shape),
        "embedding_dtype": str(matrix.dtype),
        "embedding_config": SENTENCE_TRANSFORMER_CONFIG,
        "device": device,
    }


def _load_embedding_cache(
    paths: dict[str, Path],
    expected_metadata: dict[str, Any],
    expected_row_ids: pd.Series,
) -> np.ndarray:
    required_paths = [paths["matrix"], paths["row_ids"], paths["metadata"]]
    if not all(path.exists() for path in required_paths):
        raise FileNotFoundError("Embedding cache is incomplete.")
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    for key in [
        "schema_version",
        "split_name",
        "input_sha256",
        "row_ids_sha256",
        "row_count",
        "embedding_shape",
        "embedding_dtype",
        "embedding_config",
        "device",
        "device",
    ]:
        if metadata.get(key) != expected_metadata[key]:
            raise ValueError(f"Embedding cache metadata mismatch for {key}.")
    cached_row_ids = np.load(paths["row_ids"], allow_pickle=False)
    expected_ids = expected_row_ids.astype(str).to_numpy(dtype=str)
    if not np.array_equal(cached_row_ids.astype(str), expected_ids):
        raise ValueError("Embedding cache row order does not match the audited split.")
    matrix = np.load(paths["matrix"], allow_pickle=False)
    if (
        matrix.dtype != np.float32
        or matrix.ndim != 2
        or matrix.shape[0] != len(expected_row_ids)
        or matrix.shape[1] != SENTENCE_TRANSFORMER_DIMENSIONS
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("Embedding cache matrix is invalid.")
    if metadata.get("embedding_shape") != list(matrix.shape):
        raise ValueError("Embedding cache shape metadata does not match its matrix.")
    return matrix


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(value)
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def _atomic_save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", suffix=".npy", dir=path.parent, delete=False) as handle:
        np.save(handle, array, allow_pickle=False)
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def encode_with_cache(
    encoder: UrlEncoder | None,
    split_name: str,
    urls: pd.Series,
    row_ids: pd.Series,
    cache_root: Path,
    device: str,
    refresh_cache: bool = False,
) -> tuple[np.ndarray, bool, float]:
    """Load verified embeddings or encode and atomically cache a matching matrix."""
    paths = embedding_cache_paths(cache_root, split_name)
    placeholder = np.empty((len(urls), SENTENCE_TRANSFORMER_DIMENSIONS), dtype=np.float32)
    expected_metadata = _cache_metadata(split_name, row_ids, urls, placeholder, device)
    cache_exists = any(
        path.exists() for name, path in paths.items() if name != "directory"
    )
    if cache_exists and not refresh_cache:
        return _load_embedding_cache(paths, expected_metadata, row_ids), True, 0.0
    if encoder is None:
        raise RuntimeError(
            "No encoder is available for a cache miss. Install sentence-transformers "
            "or reuse a complete, matching cache."
        )
    started_at = time.perf_counter()
    matrix = np.asarray(
        encoder.encode(
            urls.astype(str).tolist(),
            batch_size=SENTENCE_TRANSFORMER_BATCH_SIZE,
            device=device,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            precision="float32",
        ),
        dtype=np.float32,
    )
    encode_seconds = time.perf_counter() - started_at
    if (
        matrix.ndim != 2
        or matrix.shape[0] != len(urls)
        or matrix.shape[1] != SENTENCE_TRANSFORMER_DIMENSIONS
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("SentenceTransformer returned invalid embedding shape or values.")
    metadata = _cache_metadata(split_name, row_ids, urls, matrix, device)
    _atomic_save_array(paths["matrix"], matrix)
    _atomic_save_array(paths["row_ids"], row_ids.astype(str).to_numpy(dtype=str))
    _atomic_write_text(paths["metadata"], json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return matrix, False, encode_seconds


def load_sentence_transformer(device: str, cache_root: Path) -> UrlEncoder:
    """Load the immutable encoder revision without trusting remote executable code."""
    try:
        import sentence_transformers
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "SentenceTransformer path requires sentence-transformers==5.6.0. "
            "Install the declared project dependency before running it."
        ) from error
    if sentence_transformers.__version__ != "5.6.0":
        raise RuntimeError(
            "SentenceTransformer path requires version 5.6.0 for the recorded protocol; "
            f"found {sentence_transformers.__version__}."
        )
    return SentenceTransformer(
        SENTENCE_TRANSFORMER_MODEL_ID,
        revision=SENTENCE_TRANSFORMER_REVISION,
        cache_folder=str(cache_root / "model_files"),
        trust_remote_code=False,
        device=device,
    )


def _final_result_row(
    representation: str,
    winner: dict[str, Any],
    metrics: dict[str, float | int],
    fit_seconds: float,
    predict_seconds: float,
    n_train_rows: int,
    n_eval_rows: int,
    feature_count: int,
    representation_parameters: dict[str, Any],
) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "representation": representation,
        "candidate_id": winner["candidate_id"],
        "fit_scope": "train_validation",
        "dataset_split": "test",
        "is_selected": True,
        "n_train_rows": n_train_rows,
        "n_eval_rows": n_eval_rows,
        "n_features": feature_count,
        "positive_label": POSITIVE_LABEL,
        **metrics,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "parameters_json": json.dumps(
            {
                "classifier": {"C": float(winner["parameters"]["C"])},
                "representation": representation_parameters,
            },
            sort_keys=True,
        ),
    }


def _atomic_dump_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", suffix=".joblib", dir=path.parent, delete=False) as handle:
        temporary_path = Path(handle.name)
    joblib.dump(value, temporary_path)
    os.replace(temporary_path, path)


def save_model_bundle(
    representation: str,
    bundle: dict[str, Any],
    verification_X: Any,
    manifest: dict[str, Any],
) -> None:
    """Save a deployable bundle and verify a prediction round trip before manifest."""
    bundle_path = MODELS_DIR / f"{representation}.joblib"
    manifest_path = MODELS_DIR / f"{representation}_manifest.json"
    _atomic_dump_joblib(bundle_path, bundle)
    loaded_bundle = joblib.load(bundle_path)
    classifier = loaded_bundle["classifier"]
    if representation == "tfidf_logistic_regression":
        transformed = loaded_bundle["vectorizer"].transform(verification_X)
    else:
        transformed = verification_X
    expected = bundle["classifier"].predict_proba(transformed)
    actual = classifier.predict_proba(transformed)
    if not np.allclose(expected, actual, rtol=1e-12, atol=1e-12):
        raise RuntimeError(f"Saved {representation} bundle failed prediction verification.")
    manifest["artifacts"] = {
        "joblib": {
            "path": str(bundle_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(bundle_path),
        }
    }
    manifest["round_trip_prediction_rows_verified"] = len(verification_X)
    _atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _base_manifest(
    representation: str,
    winner: dict[str, Any],
    feature_count: int,
    fit_seconds: float,
    predict_seconds: float,
) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "representation": representation,
        "selected_candidate": {
            "candidate_id": winner["candidate_id"],
            "parameters": winner["parameters"],
        },
        "random_state": RANDOM_STATE,
        "class_map": {"0": "phishing", "1": "legitimate"},
        "feature_count": feature_count,
        "fit_seconds": fit_seconds,
        "test_predict_seconds": predict_seconds,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "platform": platform.platform(),
        "inputs": {
            name: {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(path),
            }
            for name, path in AUDITED_SPLIT_PATHS.items()
        },
        "split_manifest_sha256": sha256_file(SPLIT_MANIFEST_PATH),
    }


def _validate_output_targets(replace_outputs: bool, representations: set[str]) -> None:
    targets = [TUNING_RESULTS_PATH, RESULTS_PATH, COMPARISON_PATH, REPORT_PATH]
    if "tfidf" in representations:
        targets.extend(
            [
                MODELS_DIR / "tfidf_logistic_regression.joblib",
                MODELS_DIR / "tfidf_logistic_regression_manifest.json",
            ]
        )
    if "sentence-transformer" in representations:
        targets.extend(
            [
                MODELS_DIR / "sentence_transformer_logistic_regression.joblib",
                MODELS_DIR / "sentence_transformer_logistic_regression_manifest.json",
            ]
        )
    existing = [path for path in targets if path.exists()]
    if existing and not replace_outputs:
        raise FileExistsError(
            "Refusing to replace existing URL-text artifacts without --replace-outputs: "
            + ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in existing)
        )


def _atomic_write_dataframe(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        frame.to_csv(handle, index=False, float_format="%.6f")
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def build_cross_representation_comparison(final_results: pd.DataFrame) -> pd.DataFrame:
    """Create a clearly scoped tabular-versus-URL-text comparison table."""
    tabular = pd.read_csv(TABULAR_COMPARISON_PATH)
    tabular = tabular.loc[
        (tabular["dataset_split"] == "test") & tabular["is_selected"].astype(bool)
    ].copy()
    tabular_rows = pd.DataFrame(
        {
            "model": tabular["model_name"],
            "input_scope": "URL and webpage-derived tabular features",
            "f1_macro": tabular["f1_macro"],
            "roc_auc": tabular["roc_auc"],
            "balanced_accuracy": tabular["balanced_accuracy"],
            "phishing_recall": tabular["phishing_recall"],
            "test_rows": tabular["n_eval_rows"],
        }
    )
    url_rows = pd.DataFrame(
        {
            "model": final_results["representation"],
            "input_scope": "Raw URL text only",
            "f1_macro": final_results["f1_macro"],
            "roc_auc": final_results["roc_auc"],
            "balanced_accuracy": final_results["balanced_accuracy"],
            "phishing_recall": final_results["phishing_recall"],
            "test_rows": final_results["n_eval_rows"],
        }
    )
    comparison = pd.concat([tabular_rows, url_rows], ignore_index=True)
    comparison["protocol_id"] = PROTOCOL_ID
    comparison["comparison_note"] = (
        "Same audited split; input scopes differ, so metrics are descriptive rather than interchangeable."
    )
    return comparison.sort_values(
        ["f1_macro", "roc_auc", "model"], ascending=[False, False, True], kind="stable"
    ).reset_index(drop=True)


def write_training_report(
    tuning_results: pd.DataFrame,
    final_results: pd.DataFrame,
    smoke_rows: int | None,
) -> None:
    """Write protocol evidence without persisting row-level URLs or predictions."""
    lines = [
        "# Week 7 URL-Text Experiment",
        "",
        "## Outcome",
        "",
        "Character TF-IDF and a frozen SentenceTransformer embedding representation "
        "are independently tuned on train/validation and evaluated only after their "
        "downstream Logistic Regression configuration is locked.",
        "",
        "## Protocol",
        "",
        f"- Audited split protocol: {PROTOCOL_ID}.",
        "- Raw URL strings are treated as text only; this experiment never fetches URLs.",
        "- Normalized-host and exact-URL overlap checks run before every experiment.",
        "- Test labels are excluded from representation fitting, tuning, selection, and threshold choice.",
        "- Selection order: validation macro F1, phishing recall, ROC-AUC, candidate ID.",
        "- Bounded candidates: C in {0.25, 1.0, 4.0}; no weight ablation or open-ended search.",
        "",
        "## Representations",
        "",
        "- TF-IDF: raw URL character-within-word n-grams (3-5), train-fitted only.",
        "- SentenceTransformer: all-MiniLM-L6-v2 at immutable revision "
        f"{SENTENCE_TRANSFORMER_REVISION}, 384 float32 normalized dimensions, CPU by default.",
        "- all-MiniLM-L6-v2 is an English sentence encoder, not URL-specific; its 256 wordpiece "
        "truncation makes this an exploratory contextual baseline, not a deployment claim.",
        "",
        "## Results",
        "",
    ]
    display_columns = [
        "representation",
        "candidate_id",
        "f1_macro",
        "roc_auc",
        "balanced_accuracy",
        "phishing_recall",
        "mcc",
    ]
    for _, row in final_results[display_columns].iterrows():
        values = [
            f"{value:.4f}" if isinstance(value, float) else str(value)
            for value in row.tolist()
        ]
        lines.append("- " + "; ".join(f"{key}={value}" for key, value in zip(display_columns, values, strict=True)))
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            "- Model manifests record selected settings, package versions, split hashes, artifact hashes, and timing.",
            "- Embedding caches contain float32 matrices, immutable row IDs, and hashes; mismatches fail closed.",
            "- No raw URLs, labels, or row-level predictions are saved in the embedding cache or reports.",
            "",
            "## Limitations",
            "",
            "- Grouping is at normalized-host level, not registered-domain level.",
            "- URL-only results are not directly interchangeable with Week 6 webpage-feature models.",
            "- This single dataset has no temporal or external-dataset generalisation evidence.",
            "",
            "## Artifacts",
            "",
            "- reports/transformer_experiments/url_text_tuning_results.csv",
            "- reports/transformer_experiments/url_text_results.csv",
            "- reports/transformer_experiments/model_comparison.csv",
            "- models/url_text/",
        ]
    )
    if smoke_rows is not None:
        lines.extend(
            [
                "",
                "## Smoke-run warning",
                "",
                f"This run sampled at most {smoke_rows} rows per audited split. Its metrics are "
                "only a wiring check and must not be compared with full-data results.",
            ]
        )
    _atomic_write_text(REPORT_PATH, "\n".join(lines) + "\n")


def run_experiment(
    splits: dict[str, pd.DataFrame],
    representations: set[str],
    *,
    smoke_rows: int | None = None,
    device: str = "cpu",
    cache_root: Path = EMBEDDING_CACHE_DIR,
    refresh_embedding_cache: bool = False,
    replace_outputs: bool = False,
    encoder: UrlEncoder | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run train/validation selection then a single locked test evaluation per representation."""
    allowed_representations = {"tfidf", "sentence-transformer"}
    if not representations or not representations <= allowed_representations:
        raise ValueError("Representations must be a non-empty subset of tfidf, sentence-transformer.")
    validate_audited_splits(splits)
    working_splits = {
        name: (
            sample_split(splits[name], smoke_rows, RANDOM_STATE + index)
            if smoke_rows is not None
            else splits[name].copy().reset_index(drop=True)
        )
        for index, name in enumerate(["train", "validation", "test"])
    }
    _validate_output_targets(replace_outputs, representations)
    url_texts: dict[str, pd.Series] = {}
    row_ids: dict[str, pd.Series] = {}
    targets: dict[str, pd.Series] = {}
    for name in ["train", "validation", "test"]:
        url_texts[name], row_ids[name] = extract_url_texts(working_splits[name])
        targets[name] = working_splits[name][TARGET_COLUMN].reset_index(drop=True)
    combined_urls = pd.concat([url_texts["train"], url_texts["validation"]], ignore_index=True)
    combined_targets = pd.concat([targets["train"], targets["validation"]], ignore_index=True)
    combined_row_ids = pd.concat([row_ids["train"], row_ids["validation"]], ignore_index=True)
    run_scope = {
        "smoke_rows_per_split_max": smoke_rows,
        "ordered_membership_sha256": {
            name: hash_row_ids(row_ids[name]) for name in ["train", "validation", "test"]
        },
    }
    tuning_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    if "tfidf" in representations:
        vectorizer = build_tfidf_vectorizer()
        train_matrix = vectorizer.fit_transform(url_texts["train"])
        validation_matrix = vectorizer.transform(url_texts["validation"])
        rows, winner = tune_logistic_candidates(
            "tfidf_logistic_regression",
            TFIDF_CANDIDATES,
            train_matrix,
            targets["train"],
            validation_matrix,
            targets["validation"],
            train_matrix.shape[1],
            TFIDF_VECTORISER_CONFIG,
        )
        tuning_rows.extend(rows)
        winner["parameters"] = TFIDF_CANDIDATES[winner["candidate_id"]]

        final_vectorizer = build_tfidf_vectorizer()
        combined_matrix = final_vectorizer.fit_transform(combined_urls)
        test_matrix = final_vectorizer.transform(url_texts["test"])
        classifier = build_logistic_regression(winner["parameters"]["C"])
        started_at = time.perf_counter()
        classifier.fit(combined_matrix, combined_targets)
        fit_seconds = time.perf_counter() - started_at
        metrics, _, _, predict_seconds = evaluate_classifier(classifier, test_matrix, targets["test"])
        final_rows.append(
            _final_result_row(
                "tfidf_logistic_regression",
                winner,
                metrics,
                fit_seconds,
                predict_seconds,
                len(combined_targets),
                len(targets["test"]),
                combined_matrix.shape[1],
                TFIDF_VECTORISER_CONFIG,
            )
        )
        verification_urls = url_texts["test"].iloc[: min(256, len(url_texts["test"]))]
        manifest = _base_manifest(
            "tfidf_logistic_regression", winner, combined_matrix.shape[1], fit_seconds, predict_seconds
        )
        manifest["representation_parameters"] = TFIDF_VECTORISER_CONFIG
        manifest["run_scope"] = run_scope
        save_model_bundle(
            "tfidf_logistic_regression",
            {
                "protocol_id": PROTOCOL_ID,
                "representation": "tfidf_logistic_regression",
                "positive_label": POSITIVE_LABEL,
                "class_labels": CLASS_LABELS,
                "vectorizer": final_vectorizer,
                "classifier": classifier,
            },
            verification_urls,
            manifest,
        )

    if "sentence-transformer" in representations:
        active_encoder = (
            encoder
            if encoder is not None
            else load_sentence_transformer(device, cache_root)
        )
        assert active_encoder is not None
        train_embeddings, _, train_encode_seconds = encode_with_cache(
            active_encoder,
            "train",
            url_texts["train"],
            row_ids["train"],
            cache_root,
            device,
            refresh_embedding_cache,
        )
        validation_embeddings, _, validation_encode_seconds = encode_with_cache(
            active_encoder,
            "validation",
            url_texts["validation"],
            row_ids["validation"],
            cache_root,
            device,
            refresh_embedding_cache,
        )
        rows, winner = tune_logistic_candidates(
            "sentence_transformer_logistic_regression",
            EMBEDDING_CANDIDATES,
            train_embeddings,
            targets["train"],
            validation_embeddings,
            targets["validation"],
            train_embeddings.shape[1],
            SENTENCE_TRANSFORMER_CONFIG,
        )
        tuning_rows.extend(rows)
        winner["parameters"] = EMBEDDING_CANDIDATES[winner["candidate_id"]]

        combined_embeddings = np.vstack([train_embeddings, validation_embeddings])
        classifier = build_logistic_regression(winner["parameters"]["C"])
        started_at = time.perf_counter()
        classifier.fit(combined_embeddings, combined_targets)
        fit_seconds = time.perf_counter() - started_at
        test_embeddings, _, test_encode_seconds = encode_with_cache(
            active_encoder,
            "test",
            url_texts["test"],
            row_ids["test"],
            cache_root,
            device,
            refresh_embedding_cache,
        )
        metrics, _, _, predict_seconds = evaluate_classifier(classifier, test_embeddings, targets["test"])
        final_rows.append(
            _final_result_row(
                "sentence_transformer_logistic_regression",
                winner,
                metrics,
                fit_seconds,
                predict_seconds,
                len(combined_targets),
                len(targets["test"]),
                combined_embeddings.shape[1],
                SENTENCE_TRANSFORMER_CONFIG,
            )
        )
        verification_embeddings = test_embeddings[: min(256, len(test_embeddings))]
        manifest = _base_manifest(
            "sentence_transformer_logistic_regression",
            winner,
            combined_embeddings.shape[1],
            fit_seconds,
            predict_seconds,
        )
        manifest["sentence_transformer"] = {
            **SENTENCE_TRANSFORMER_CONFIG,
            "device": device,
            "encoder_max_seq_length": int(getattr(active_encoder, "max_seq_length", -1)),
            "external_weights_required": True,
            "cache_root": str(cache_root.relative_to(PROJECT_ROOT)),
            "combined_train_validation_row_ids_sha256": hash_row_ids(combined_row_ids),
            "sentence_transformers_version": "5.6.0",
            "encoding_seconds": {
                "train": train_encode_seconds,
                "validation": validation_encode_seconds,
                "test": test_encode_seconds,
            },
        }
        manifest["run_scope"] = run_scope
        save_model_bundle(
            "sentence_transformer_logistic_regression",
            {
                "protocol_id": PROTOCOL_ID,
                "representation": "sentence_transformer_logistic_regression",
                "positive_label": POSITIVE_LABEL,
                "class_labels": CLASS_LABELS,
                "sentence_transformer": SENTENCE_TRANSFORMER_CONFIG,
                "classifier": classifier,
            },
            verification_embeddings,
            manifest,
        )

    tuning_results = pd.DataFrame(tuning_rows)
    final_results = pd.DataFrame(final_rows)
    _atomic_write_dataframe(TUNING_RESULTS_PATH, tuning_results)
    _atomic_write_dataframe(RESULTS_PATH, final_results)
    _atomic_write_dataframe(COMPARISON_PATH, build_cross_representation_comparison(final_results))
    write_training_report(tuning_results, final_results, smoke_rows)
    return tuning_results, final_results


def parse_args() -> argparse.Namespace:
    """Parse explicit full-run and bounded smoke-run controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=["tfidf", "sentence-transformer"],
        default=["tfidf", "sentence-transformer"],
        help="Representations to run. Default: both full-data paths.",
    )
    parser.add_argument(
        "--smoke-rows",
        type=int,
        default=None,
        help="Deterministically sample at most this many rows per fixed split; not comparable to a full run.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu"],
        help="Explicit SentenceTransformer device. CPU is the reproducible default.",
    )
    parser.add_argument(
        "--refresh-embedding-cache",
        action="store_true",
        help="Re-encode and replace existing embedding-cache entries.",
    )
    parser.add_argument(
        "--replace-outputs",
        action="store_true",
        help="Allow replacement of existing URL-text reports and model bundles.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the Week 7 URL-text workflow without silently changing protocol."""
    args = parse_args()
    try:
        splits = load_audited_splits()
        _, final_results = run_experiment(
            splits,
            set(args.representations),
            smoke_rows=args.smoke_rows,
            device=args.device,
            refresh_embedding_cache=args.refresh_embedding_cache,
            replace_outputs=args.replace_outputs,
        )
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1
    print("Week 7 URL-text experiment complete")
    print(final_results[["representation", "f1_macro", "roc_auc", "phishing_recall"]].to_string(index=False))
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

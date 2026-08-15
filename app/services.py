"""Multi-model URL and guarded live-page inference services."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import os
import select
from statistics import median
import subprocess
import sys
import threading
from typing import Any

import joblib
import numpy as np

from .features import FeatureExtractionResult, extract_features
from .fetcher import FetchError, FetchResult, fetch_html


PHISHING_LABEL = 0
LEGITIMATE_LABEL = 1
EXPECTED_CLASS_LABELS = (PHISHING_LABEL, LEGITIMATE_LABEL)
MODEL_IDS = (
    "automatic", "all", "tfidf", "minilm", "lightgbm", "xgboost",
    "logistic_regression", "decision_tree", "random_forest",
)
DEEP_MODEL_IDS = frozenset(MODEL_IDS[4:])


class ModelArtifactError(RuntimeError):
    """Raised when a required saved model is absent or incompatible."""


class AnalysisUnavailableError(RuntimeError):
    """Raised when the explicitly selected analysis cannot be completed."""


@dataclass(frozen=True)
class Signal:
    ngram: str
    contribution: float
    direction: str


@dataclass(frozen=True)
class ModelScore:
    model_id: str
    display_name: str
    input_scope: str
    status: str
    phishing_probability: float | None = None
    predicted_label: str | None = None


@dataclass(frozen=True)
class AnalysisResult:
    predicted_label: str
    phishing_probability: float
    confidence: float
    signals: tuple[Signal, ...]
    selected_model: str = "tfidf"
    agreement: float = 1.0
    feature_coverage: float = 0.0
    deep_scan_status: str = "not_requested"
    model_scores: tuple[ModelScore, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    display_name: str
    input_scope: str


MODEL_INFO = (
    ModelInfo("automatic", "Automatic consensus", "URL text + live webpage"),
    ModelInfo("all", "Compare all models", "URL text + live webpage"),
    ModelInfo("tfidf", "TF-IDF", "URL text only"),
    ModelInfo("minilm", "MiniLM", "URL text only"),
    ModelInfo("lightgbm", "LightGBM", "47 approximate live features"),
    ModelInfo("xgboost", "XGBoost", "47 approximate live features"),
    ModelInfo("logistic_regression", "Logistic Regression", "47 approximate live features"),
    ModelInfo("decision_tree", "Decision Tree", "47 approximate live features"),
    ModelInfo("random_forest", "Random Forest", "47 approximate live features"),
)
MODEL_INFO_BY_ID = {item.model_id: item for item in MODEL_INFO}
MINILM_WEIGHTS_SHA256 = "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
MINILM_REQUIRED_FILES = (
    "1_Pooling/config.json", "config.json", "config_sentence_transformers.json",
    "model.safetensors", "modules.json", "sentence_bert_config.json",
    "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt",
)


def _phishing_probability(classifier: Any, matrix: Any) -> float:
    try:
        classes = tuple(int(label) for label in classifier.classes_)
    except (AttributeError, TypeError, ValueError) as error:
        raise ModelArtifactError("Model classes are invalid.") from error
    if classes != EXPECTED_CLASS_LABELS:
        raise ModelArtifactError("Model classes must be ordered exactly as [0, 1].")
    probabilities = np.asarray(classifier.predict_proba(matrix), dtype=float)
    if probabilities.shape != (1, 2) or not np.isfinite(probabilities).all():
        raise ModelArtifactError("Model returned invalid probabilities.")
    score = float(probabilities[0, 0])
    if not 0.0 <= score <= 1.0:
        raise ModelArtifactError("Model returned a probability outside [0, 1].")
    return score


def _validate_vendored_minilm(model_path: Path) -> None:
    if any(not (model_path / relative_path).is_file() for relative_path in MINILM_REQUIRED_FILES):
        raise ModelArtifactError("Vendored MiniLM model is incomplete.")
    with (model_path / "model.safetensors").open("rb") as weights:
        digest = hashlib.file_digest(weights, "sha256").hexdigest()
    if digest != MINILM_WEIGHTS_SHA256:
        raise ModelArtifactError("Vendored MiniLM weights failed integrity validation.")


class URLTextModelService:
    """Validated TF-IDF bundle and local contribution inspection."""

    def __init__(self, vectorizer: Any, classifier: Any, model_name: str) -> None:
        if not hasattr(vectorizer, "transform") or not hasattr(vectorizer, "get_feature_names_out"):
            raise ModelArtifactError("URL-text bundle has an invalid TF-IDF vectorizer.")
        if not hasattr(classifier, "coef_") or not hasattr(classifier, "predict_proba"):
            raise ModelArtifactError("URL-text bundle has an invalid classifier.")
        self._vectorizer = vectorizer
        self._classifier = classifier
        self.model_name = model_name
        if tuple(int(label) for label in classifier.classes_) != EXPECTED_CLASS_LABELS:
            raise ModelArtifactError("TF-IDF classes must be ordered exactly as [0, 1].")

    @classmethod
    def load(cls, artifact_path: Path) -> "URLTextModelService":
        bundle = _load_mapping_bundle(artifact_path)
        if bundle.get("class_labels") != [0, 1] or bundle.get("positive_label") != 0:
            raise ModelArtifactError("TF-IDF bundle has incompatible class semantics.")
        if bundle.get("representation") != "tfidf_logistic_regression":
            raise ModelArtifactError("Expected the TF-IDF Logistic Regression bundle.")
        return cls(bundle.get("vectorizer"), bundle.get("classifier"), "tfidf_logistic_regression")

    def analyze(self, url: str) -> AnalysisResult:
        score, row = self.score(url)
        return AnalysisResult(
            predicted_label="phishing" if score >= 0.5 else "legitimate",
            phishing_probability=score,
            confidence=max(score, 1.0 - score),
            signals=tuple(self._top_signals(row)),
        )

    def score(self, url: str) -> tuple[float, Any]:
        row = self._vectorizer.transform([url])
        return _phishing_probability(self._classifier, row), row

    def _top_signals(self, row: Any, limit: int = 5) -> list[Signal]:
        feature_names = self._vectorizer.get_feature_names_out()
        coefficients = np.asarray(self._classifier.coef_, dtype=float)
        if row.shape[1] != len(feature_names) or coefficients.shape != (1, row.shape[1]):
            raise ModelArtifactError("TF-IDF feature names and coefficients do not align.")
        active_indices = row.indices
        if not len(active_indices):
            return []
        weighted = row.data * -coefficients[0][active_indices]
        ranked = np.argsort(np.abs(weighted))[::-1][:limit]
        return [
            Signal(
                ngram=str(feature_names[active_indices[position]]),
                contribution=float(weighted[position]),
                direction="increases phishing risk" if weighted[position] > 0 else "reduces phishing risk",
            )
            for position in ranked
        ]


class MiniLMProcessEncoder:
    """Persistent isolated encoder used to avoid native runtime collisions."""

    def __init__(self, model_path: Path) -> None:
        environment = os.environ.copy()
        environment.setdefault("HF_HUB_OFFLINE", "1")
        environment.setdefault("TRANSFORMERS_OFFLINE", "1")
        self._process = subprocess.Popen(
            [sys.executable, "-m", "app.minilm_worker", str(model_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=environment,
        )
        self._lock = threading.Lock()
        ready = self._read_message(timeout=60.0)
        if ready.get("status") != "ready":
            self.close()
            raise ModelArtifactError("MiniLM worker did not start correctly.")

    def _read_message(self, timeout: float) -> dict[str, Any]:
        if self._process.stdout is None:
            raise ModelArtifactError("MiniLM worker output is unavailable.")
        readable, _, _ = select.select([self._process.stdout], [], [], timeout)
        if not readable:
            raise ModelArtifactError("MiniLM inference timed out.")
        line = self._process.stdout.readline()
        if not line:
            raise ModelArtifactError("MiniLM worker stopped unexpectedly.")
        try:
            message = json.loads(line)
        except json.JSONDecodeError as error:
            raise ModelArtifactError("MiniLM worker returned invalid output.") from error
        if not isinstance(message, dict):
            raise ModelArtifactError("MiniLM worker returned invalid output.")
        return message

    def encode(self, urls: list[str], **_: Any) -> np.ndarray:
        if len(urls) != 1 or self._process.stdin is None:
            raise ModelArtifactError("MiniLM worker accepts one URL per request.")
        with self._lock:
            if self._process.poll() is not None:
                raise ModelArtifactError("MiniLM worker is unavailable.")
            self._process.stdin.write(json.dumps({"url": urls[0]}) + "\n")
            self._process.stdin.flush()
            response = self._read_message(timeout=30.0)
        if "error" in response:
            raise ModelArtifactError("MiniLM inference failed.")
        return np.asarray([response.get("embedding")], dtype=np.float32)

    def close(self) -> None:
        if self._process.poll() is not None:
            return
        if self._process.stdin is not None:
            self._process.stdin.close()
        try:
            self._process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            self._process.wait(timeout=5.0)


def _load_mapping_bundle(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ModelArtifactError(f"Required model artifact is unavailable: {path.name}")
    try:
        value = joblib.load(path)
    except Exception as error:
        raise ModelArtifactError(f"Model artifact could not be loaded: {path.name}") from error
    if not isinstance(value, dict):
        raise ModelArtifactError(f"Expected a mapping model bundle: {path.name}")
    return value


class MultiModelService:
    """Load every trained artifact once and coordinate explicit inference scopes."""

    def __init__(
        self,
        *,
        tfidf: URLTextModelService,
        minilm_bundle: dict[str, Any],
        encoder: Any,
        models: dict[str, Any],
        page_fetcher: Callable[[str], FetchResult] = fetch_html,
    ) -> None:
        self.tfidf = tfidf
        self.minilm_bundle = minilm_bundle
        self.encoder = encoder
        self.models = models
        self.page_fetcher = page_fetcher
        self.model_name = "multi_model_consensus"
        expected = set(MODEL_IDS) - {"automatic", "all", "tfidf", "minilm"}
        if set(models) != expected:
            raise ModelArtifactError("Loaded tabular model inventory is incomplete.")

    @classmethod
    def load(cls, models_root: Path) -> "MultiModelService":
        tfidf = URLTextModelService.load(models_root / "url_text" / "tfidf_logistic_regression.joblib")
        minilm_bundle = _load_mapping_bundle(
            models_root / "url_text" / "sentence_transformer_logistic_regression.joblib"
        )
        if minilm_bundle.get("class_labels") != [0, 1] or minilm_bundle.get("positive_label") != 0:
            raise ModelArtifactError("MiniLM bundle has incompatible class semantics.")
        models: dict[str, Any] = {
            "lightgbm": _load_mapping_bundle(models_root / "advanced" / "lightgbm.joblib"),
            "xgboost": _load_mapping_bundle(models_root / "advanced" / "xgboost.joblib"),
        }
        for name in ("logistic_regression", "decision_tree", "random_forest"):
            models[name] = joblib.load(models_root / "audited_baseline" / f"{name}.joblib")
        config = minilm_bundle.get("sentence_transformer", {})
        model_id = config.get("model_id")
        revision = config.get("revision")
        if not isinstance(model_id, str) or not isinstance(revision, str):
            raise ModelArtifactError("MiniLM bundle is missing its pinned model identity.")
        vendored_model = models_root / "url_text" / "all-MiniLM-L6-v2"
        _validate_vendored_minilm(vendored_model)
        encoder = MiniLMProcessEncoder(vendored_model)
        return cls(tfidf=tfidf, minilm_bundle=minilm_bundle, encoder=encoder, models=models)

    @property
    def model_info(self) -> tuple[ModelInfo, ...]:
        return MODEL_INFO

    def close(self) -> None:
        close = getattr(self.encoder, "close", None)
        if callable(close):
            close()

    def _score_minilm(self, url: str) -> float:
        matrix = np.asarray(
            self.encoder.encode(
                [url], batch_size=1, device="cpu", show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True, precision="float32",
            ),
            dtype=np.float32,
        )
        if matrix.shape != (1, 384) or not np.isfinite(matrix).all():
            raise ModelArtifactError("MiniLM returned invalid embeddings.")
        return _phishing_probability(self.minilm_bundle["classifier"], matrix)

    def _score_tabular(self, model_id: str, features: FeatureExtractionResult) -> float:
        model = self.models[model_id]
        frame = features.audited
        if isinstance(model, dict):
            expected_names = model.get("input_feature_names")
            if expected_names != frame.columns.tolist():
                raise ModelArtifactError(f"{model_id} feature contract does not match extraction output.")
            matrix = model["preprocessor"].transform(frame)
            return _phishing_probability(model["model"], matrix)
        expected_names = list(getattr(model, "feature_names_in_", []))
        if expected_names != frame.columns.tolist():
            raise ModelArtifactError(f"{model_id} feature contract does not match extraction output.")
        return _phishing_probability(model, frame)

    @staticmethod
    def _model_score(model_id: str, score: float) -> ModelScore:
        info = MODEL_INFO_BY_ID[model_id]
        return ModelScore(
            model_id=model_id,
            display_name=info.display_name,
            input_scope=info.input_scope,
            status="ok",
            phishing_probability=score,
            predicted_label="phishing" if score >= 0.5 else "legitimate",
        )

    def analyze(self, url: str, selected_model: str = "automatic", deep_scan: bool = True) -> AnalysisResult:
        if selected_model not in MODEL_IDS:
            raise ValueError("Unknown model selection.")
        tfidf_score, tfidf_row = self.tfidf.score(url)
        score_map = {"tfidf": tfidf_score}
        results = [self._model_score("tfidf", tfidf_score)]
        if selected_model != "tfidf":
            minilm_score = self._score_minilm(url)
            score_map["minilm"] = minilm_score
            if selected_model == "minilm":
                results = [self._model_score("minilm", minilm_score)]
            else:
                results.append(self._model_score("minilm", minilm_score))
        warnings = [
            "Scores are uncalibrated dataset outputs, not verified real-world probabilities."
        ]
        feature_coverage = 0.0
        deep_status = "not_requested"
        needs_deep = deep_scan and selected_model not in {"tfidf", "minilm"}
        if needs_deep:
            try:
                fetched = self.page_fetcher(url)
                extracted = extract_features(url, fetched)
                feature_coverage = extracted.coverage
                deep_status = "complete"
                model_ids = [
                    "lightgbm", "xgboost", "logistic_regression",
                    "decision_tree", "random_forest",
                ]
                for model_id in model_ids:
                    score = self._score_tabular(model_id, extracted)
                    score_map[model_id] = score
                    results.append(self._model_score(model_id, score))
                warnings.append(
                    f"Live-page features are approximate; {len(extracted.approximate_features)} semantic or derived fields do not have a published executable reference."
                )
                if extracted.final_url_changed:
                    warnings.append("The fetched page redirected to a different URL before extraction.")
            except FetchError as error:
                deep_status = "failed"
                warnings.append(str(error))
                if selected_model in DEEP_MODEL_IDS or selected_model == "all":
                    raise AnalysisUnavailableError(str(error)) from error

        if selected_model in {"tfidf", "minilm"}:
            final_score = score_map[selected_model]
            participating = [score_map[selected_model]]
        elif selected_model in DEEP_MODEL_IDS:
            if selected_model not in score_map:
                raise AnalysisUnavailableError("Selected webpage model was not available.")
            final_score = score_map[selected_model]
            participating = [final_score]
        else:
            minilm_score = score_map["minilm"]
            url_consensus = (tfidf_score + minilm_score) / 2.0
            page_ids = [
                model_id for model_id in (
                    "lightgbm", "xgboost", "logistic_regression",
                    "decision_tree", "random_forest",
                ) if model_id in score_map
            ]
            if page_ids:
                page_consensus = median(score_map[model_id] for model_id in page_ids)
                final_score = 0.35 * url_consensus + 0.65 * page_consensus
                participating = [tfidf_score, minilm_score, *(score_map[item] for item in page_ids)]
            else:
                final_score = url_consensus
                participating = [tfidf_score, minilm_score]

        majority_label = final_score >= 0.5
        agreement = sum((score >= 0.5) == majority_label for score in participating) / len(participating)
        uncertain = 0.35 <= final_score <= 0.65 or agreement < 0.70
        if needs_deep and deep_status != "complete":
            uncertain = True
        predicted = "uncertain" if uncertain else ("phishing" if majority_label else "legitimate")
        return AnalysisResult(
            predicted_label=predicted,
            phishing_probability=final_score,
            confidence=max(final_score, 1.0 - final_score),
            signals=tuple(self.tfidf._top_signals(tfidf_row)) if selected_model != "minilm" else (),
            selected_model=selected_model,
            agreement=agreement,
            feature_coverage=feature_coverage,
            deep_scan_status=deep_status,
            model_scores=tuple(results),
            warnings=tuple(warnings),
        )

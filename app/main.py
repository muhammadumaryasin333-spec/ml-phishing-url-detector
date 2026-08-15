"""FastAPI entry point for multi-model phishing analysis."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
import ipaddress
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .services import (
    AnalysisResult,
    AnalysisUnavailableError,
    ModelArtifactError,
    MultiModelService,
)


APP_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIRECTORY.parent
STATIC_DIRECTORY = APP_DIRECTORY / "static"
DEFAULT_MODELS_ROOT = PROJECT_ROOT / "models"
DEFAULT_ALLOWED_HOSTS = ("localhost", "127.0.0.1", "testserver")
MAX_URL_LENGTH = 2_048
MAX_REQUEST_BODY_BYTES = 4_096


class AnalysisRequest(BaseModel):
    """One URL, explicit model choice, and live-page preference."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    url: str = Field(min_length=1, max_length=MAX_URL_LENGTH)
    model: Literal[
        "automatic", "all", "tfidf", "minilm", "lightgbm", "xgboost",
        "logistic_regression", "decision_tree", "random_forest",
    ] = "automatic"
    deep_scan: bool = True

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        if any(character.isspace() for character in value):
            raise ValueError("URL must not contain whitespace.")
        normalized = value.replace("[.]", ".")
        try:
            parsed = urlsplit(normalized)
            port = parsed.port
        except ValueError as error:
            raise ValueError("URL must include a valid hostname and port.") from error
        if parsed.scheme.casefold() not in {"http", "https"}:
            raise ValueError("Only http and https URLs are supported.")
        if not parsed.hostname:
            raise ValueError("URL must include a valid hostname.")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("URLs containing embedded credentials are not supported.")
        if port not in {None, 80, 443}:
            raise ValueError("Only standard HTTP and HTTPS ports are supported.")
        _validate_hostname(parsed.hostname)
        return normalized


class SignalResponse(BaseModel):
    ngram: str = Field(min_length=1, max_length=256)
    contribution: float
    direction: str


class ModelScoreResponse(BaseModel):
    model_id: str
    display_name: str
    input_scope: str
    status: str
    phishing_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    predicted_label: str | None = None


class AnalysisResponse(BaseModel):
    predicted_label: str
    phishing_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    probability_note: str
    analysis_scope: str
    signals: list[SignalResponse]
    selected_model: str
    agreement: float = Field(ge=0.0, le=1.0)
    feature_coverage: float = Field(ge=0.0, le=1.0)
    deep_scan_status: str
    model_scores: list[ModelScoreResponse]
    warnings: list[str]


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    model_name: str
    inference_scope: str


class ModelInfoResponse(BaseModel):
    model_id: str
    display_name: str
    input_scope: str


class ModelsResponse(BaseModel):
    models: list[ModelInfoResponse]


AnalysisServiceFactory = Callable[[Path], Any]


def _validate_hostname(hostname: str) -> None:
    host = hostname.rstrip(".")
    if not host:
        raise ValueError("URL must include a valid hostname.")
    try:
        ipaddress.ip_address(host)
        return
    except ValueError:
        pass
    try:
        ascii_host = host.encode("idna").decode("ascii")
    except UnicodeError as error:
        raise ValueError("URL must include a valid hostname.") from error
    if len(ascii_host) > 253:
        raise ValueError("URL hostname is too long.")
    for label in ascii_host.split("."):
        if (
            not label or len(label) > 63 or label.startswith("-") or label.endswith("-")
            or not all(character.isascii() and (character.isalnum() or character == "-") for character in label)
        ):
            raise ValueError("URL must include a valid hostname.")


def get_analysis_service(request: Request) -> Any:
    service = getattr(request.app.state, "analysis_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Analysis service is unavailable.")
    return service


def build_response(result: AnalysisResult) -> AnalysisResponse:
    scope = (
        "URL text plus a guarded, bounded live HTML request. JavaScript is not executed."
        if result.deep_scan_status != "not_requested"
        else "URL text only. The submitted site was not contacted."
    )
    return AnalysisResponse(
        predicted_label=result.predicted_label,
        phishing_probability=result.phishing_probability,
        confidence=result.confidence,
        probability_note="Uncalibrated dataset model score; not a verified real-world probability.",
        analysis_scope=scope,
        signals=[SignalResponse(ngram=item.ngram, contribution=item.contribution, direction=item.direction) for item in result.signals],
        selected_model=result.selected_model,
        agreement=result.agreement,
        feature_coverage=result.feature_coverage,
        deep_scan_status=result.deep_scan_status,
        model_scores=[
            ModelScoreResponse(
                model_id=item.model_id,
                display_name=item.display_name,
                input_scope=item.input_scope,
                status=item.status,
                phishing_probability=item.phishing_probability,
                predicted_label=item.predicted_label,
            )
            for item in result.model_scores
        ],
        warnings=list(result.warnings),
    )


def create_app(
    *,
    model_path: Path | None = None,
    service_factory: AnalysisServiceFactory = MultiModelService.load,
    allowed_hosts: Sequence[str] = DEFAULT_ALLOWED_HOSTS,
) -> FastAPI:
    resolved_models_root = model_path or DEFAULT_MODELS_ROOT
    trusted_hosts = tuple(allowed_hosts)
    if not trusted_hosts or "*" in trusted_hosts:
        raise ValueError("Trusted hosts must be explicit and must not include '*'.")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.analysis_service = service_factory(resolved_models_root)
        try:
            yield
        finally:
            close = getattr(app.state.analysis_service, "close", None)
            if callable(close):
                close()
            app.state.analysis_service = None

    application = FastAPI(
        title="Phishing URL Analyser", version="2.0.0", docs_url=None,
        redoc_url=None, lifespan=lifespan,
    )
    application.add_middleware(TrustedHostMiddleware, allowed_hosts=list(trusted_hosts))

    @application.middleware("http")
    async def add_security_headers(request: Request, call_next: Callable) -> Response:
        content_length = request.headers.get("content-length")
        if (
            request.method == "POST" and content_length is not None
            and content_length.isdecimal() and int(content_length) > MAX_REQUEST_BODY_BYTES
        ):
            response: Response = JSONResponse(status_code=413, content={"detail": "Request is too large."})
        else:
            response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'; "
            "img-src 'self'; script-src 'self'; style-src 'self'"
        )
        response.headers["Permissions-Policy"] = "camera=(), geolocation=(), microphone=()"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @application.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, __: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": "Invalid request."})

    @application.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIRECTORY / "index.html")

    @application.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        return Response(status_code=204)

    @application.get("/health", response_model=HealthResponse)
    async def health(service: Any = Depends(get_analysis_service)) -> HealthResponse:
        return HealthResponse(
            status="ok", model_ready=True, model_name=service.model_name,
            inference_scope="url_text_and_guarded_live_page",
        )

    @application.get("/api/models", response_model=ModelsResponse)
    async def models(service: Any = Depends(get_analysis_service)) -> ModelsResponse:
        return ModelsResponse(
            models=[
                ModelInfoResponse(
                    model_id=item.model_id,
                    display_name=item.display_name,
                    input_scope=item.input_scope,
                )
                for item in getattr(service, "model_info", ())
            ]
        )

    @application.post("/api/analyze", response_model=AnalysisResponse)
    async def analyze(
        payload: AnalysisRequest,
        service: Any = Depends(get_analysis_service),
    ) -> AnalysisResponse:
        try:
            result = await asyncio.to_thread(
                service.analyze, payload.url, payload.model, payload.deep_scan
            )
            return build_response(result)
        except AnalysisUnavailableError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        except ModelArtifactError as error:
            raise HTTPException(status_code=503, detail="Analysis service is unavailable.") from error

    application.mount("/static", StaticFiles(directory=STATIC_DIRECTORY), name="static")
    return application


app = create_app()

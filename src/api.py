"""FastAPI service. Run: uvicorn src.api:app --reload   (interactive docs at /docs)

Environment:
  MODEL_PATH       path to the model bundle            (default models/model.joblib)
  AUDIT_LOG_PATH   JSONL decision log; empty disables  (default logs/decisions.jsonl)
  API_KEYS         comma-separated keys; if set, /decision, /decision/batch and /stats need X-API-Key
"""

from __future__ import annotations

import os
import secrets
import threading
import time
import uuid
from collections import Counter, deque
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, Response, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field

from .audit import AuditLog
from .engine import DecisionEngine
from .model import load_bundle

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/model.joblib"))
MAX_BATCH = 100

EXAMPLE_REQUEST = {
    "pan_hash": "a3f9c1d2e4b5",
    "age_years": 34,
    "annual_income": 540000,
    "loan_amount": 450000,
    "annuity": 38000,
    "goods_price": 420000,
    "years_employed": 6.5,
}


class DecisionRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [EXAMPLE_REQUEST]})

    pan_hash: str = Field(min_length=1, max_length=128, description="Hashed PAN. Never send a raw PAN.")
    age_years: int = Field(ge=0, le=120)
    annual_income: float = Field(ge=0)
    loan_amount: float = Field(gt=0)
    annuity: float = Field(gt=0, description="Loan annuity (repayment amount), same unit as the training data")
    goods_price: float | None = Field(default=None, gt=0)
    years_employed: float | None = Field(default=None, ge=0)
    ext_source_1: float | None = Field(default=None, ge=0, le=1)
    ext_source_2: float | None = Field(default=None, ge=0, le=1, description="If absent, derived from the bureau score")
    ext_source_3: float | None = Field(default=None, ge=0, le=1)


class Reason(BaseModel):
    feature: str
    reason: str
    shap: float | None = Field(default=None, description="SHAP contribution to log-odds of default (model reasons only)")


class RuleResult(BaseModel):
    rule: str
    passed: bool


class DecisionResponse(BaseModel):
    decision: str = Field(description="APPROVE | REFER | DECLINE")
    pd: float | None = Field(description="Calibrated probability of default; null when a hard rule declined first")
    reasons: list[Reason]
    rule_results: list[RuleResult]
    failed_rules: list[str]
    bureau_score: int
    kyc_verified: bool
    thresholds: dict[str, float]
    model_version: str
    request_id: str
    latency_ms: float


class BatchRequest(BaseModel):
    items: list[DecisionRequest] = Field(min_length=1, max_length=MAX_BATCH)


class BatchResponse(BaseModel):
    results: list[DecisionResponse]


class Stats:
    """In-process counters. Good enough for a demo; production would export Prometheus metrics."""

    def __init__(self):
        self.lock = threading.Lock()
        self.decisions: Counter = Counter()
        self.latencies: deque = deque(maxlen=1000)
        self.started = time.time()

    def record(self, decision: str, latency_ms: float) -> None:
        with self.lock:
            self.decisions[decision] += 1
            self.latencies.append(latency_ms)

    def snapshot(self) -> dict:
        with self.lock:
            lat = list(self.latencies)
            return {
                "uptime_s": round(time.time() - self.started),
                "decisions": dict(self.decisions),
                "latency_ms": {"p50": round(float(np.percentile(lat, 50)), 2), "p95": round(float(np.percentile(lat, 95)), 2)}
                if lat
                else None,
            }


def create_app(
    engine: DecisionEngine | None = None,
    audit: AuditLog | None = None,
    api_keys: list[str] | None = None,
) -> FastAPI:
    """Tests pass an engine; in production the engine is loaded lazily from MODEL_PATH."""
    app = FastAPI(
        title="Credit Decisioning Engine",
        version="2.0.0",
        description="Eligibility rules -> calibrated PD model -> policy decision with SHAP reason codes. "
        "Bureau and e-KYC are mock adapters. Demonstration project, not for real lending decisions.",
    )
    state = {"engine": engine}
    audit = audit if audit is not None else AuditLog(os.environ.get("AUDIT_LOG_PATH", "logs/decisions.jsonl") or None)
    keys = api_keys if api_keys is not None else [k for k in os.environ.get("API_KEYS", "").split(",") if k]
    stats = Stats()
    key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def get_engine() -> DecisionEngine:
        if state["engine"] is None:
            if not MODEL_PATH.exists():
                raise HTTPException(503, f"model not found at {MODEL_PATH}; run `python -m src.pipeline`")
            state["engine"] = DecisionEngine(load_bundle(MODEL_PATH))
        return state["engine"]

    def require_key(key: str | None = Security(key_header)) -> None:
        if keys and not (key and any(secrets.compare_digest(key, k) for k in keys)):
            raise HTTPException(401, "missing or invalid X-API-Key")

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    def score(req: DecisionRequest, request_id: str) -> dict:
        start = time.perf_counter()
        payload = req.model_dump()
        result = get_engine().decide(payload)
        latency = (time.perf_counter() - start) * 1000
        result.update(request_id=request_id, latency_ms=round(latency, 2))
        audit.write(request_id, payload, result, latency)
        stats.record(result["decision"], latency)
        return result

    @app.get("/health", tags=["ops"])
    def health():
        try:
            return {"status": "ok", "model_version": get_engine().bundle.version}
        except HTTPException:
            return {"status": "no_model", "model_version": None}

    @app.get("/model-info", tags=["ops"])
    def model_info():
        b = get_engine().bundle
        return {
            "model_version": b.version,
            "data_source": b.meta.get("data_source"),
            "resume_safe": b.meta.get("data_source") == "home_credit",
            "thresholds": b.meta["policy"],
            "rules": b.meta.get("rules"),
            "features": b.meta.get("feature_columns"),
            "test_auc": b.meta.get("test_auc"),
            "calibrated": b.calibration is not None,
        }

    @app.get("/stats", tags=["ops"], dependencies=[Depends(require_key)])
    def get_stats():
        return stats.snapshot()

    @app.post("/decision", response_model=DecisionResponse, tags=["decision"], dependencies=[Depends(require_key)])
    def decision(req: DecisionRequest, request: Request):
        return score(req, request.state.request_id)

    @app.post("/decision/batch", response_model=BatchResponse, tags=["decision"], dependencies=[Depends(require_key)])
    def decision_batch(batch: BatchRequest, request: Request, response: Response):
        rid = request.state.request_id
        return {"results": [score(item, f"{rid}-{i}") for i, item in enumerate(batch.items)]}

    return app


app = create_app()

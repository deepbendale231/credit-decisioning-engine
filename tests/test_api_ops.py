import json

from conftest import GOOD
from fastapi.testclient import TestClient

from src.api import create_app
from src.audit import AuditLog
from src.engine import DecisionEngine
from src.policy import Policy


def make(bundle, tmp_path=None, keys=None, t=(0.05, 0.15)):
    audit = AuditLog(tmp_path / "audit.jsonl" if tmp_path else None)
    return TestClient(create_app(DecisionEngine(bundle, Policy(*t)), audit=audit, api_keys=keys)), audit


def test_request_id_is_generated_and_echoed(bundle):
    c, _ = make(bundle)
    assert len(c.get("/health").headers["X-Request-ID"]) == 16
    r = c.post("/decision", json=GOOD, headers={"X-Request-ID": "abc-123"})
    assert r.headers["X-Request-ID"] == "abc-123" and r.json()["request_id"] == "abc-123"


def test_audit_log_records_replayable_decision(bundle, tmp_path):
    c, audit = make(bundle, tmp_path)
    body = c.post("/decision", json=GOOD).json()
    [rec] = audit.read()
    assert rec["decision"] == body["decision"] and rec["pd"] == body["pd"] and rec["model_version"] == "test-model"
    assert rec["pan_prefix"] == "GOODPAN1"[:8] and "pan_hash" not in json.dumps(rec["request"])
    assert set(rec["features"]) >= {"EXT_SOURCE_MEAN", "CREDIT_INCOME_RATIO"}
    # replay: the logged inputs reproduce the logged decision
    replay = DecisionEngine(bundle, Policy(0.05, 0.15)).decide({**rec["request"], "pan_hash": "GOODPAN1"})
    assert replay["decision"] == rec["decision"] and replay["pd"] == rec["pd"]


def test_disabled_audit_writes_nothing(bundle, tmp_path):
    c, audit = make(bundle)
    c.post("/decision", json=GOOD)
    assert audit.read() == [] and not list(tmp_path.iterdir())


def test_api_key_enforced_when_configured(bundle):
    c, _ = make(bundle, keys=["s3cret"])
    assert c.get("/health").status_code == 200
    assert c.post("/decision", json=GOOD).status_code == 401
    assert c.post("/decision", json=GOOD, headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.post("/decision", json=GOOD, headers={"X-API-Key": "s3cret"}).status_code == 200
    assert c.get("/stats").status_code == 401


def test_batch_scores_each_item_and_caps_size(bundle, tmp_path):
    c, audit = make(bundle, tmp_path)
    items = [GOOD, {**GOOD, "age_years": 19}, {**GOOD, "pan_hash": "KYCFAIL-9"}]
    r = c.post("/decision/batch", json={"items": items}).json()
    assert [x["decision"] for x in r["results"]][1:] == ["DECLINE", "DECLINE"]
    assert len({x["request_id"] for x in r["results"]}) == 3 and len(audit.read()) == 3
    assert c.post("/decision/batch", json={"items": [GOOD] * 101}).status_code == 422
    assert c.post("/decision/batch", json={"items": []}).status_code == 422


def test_model_info_and_stats(bundle):
    c, _ = make(bundle)
    c.post("/decision", json=GOOD)
    info = c.get("/model-info").json()
    assert info["model_version"] == "test-model" and info["calibrated"] and info["resume_safe"] is False
    assert info["rules"]["min_age"] == 21
    stats = c.get("/stats").json()
    assert sum(stats["decisions"].values()) == 1 and stats["latency_ms"]["p95"] >= 0


def test_no_model_gives_503_and_health_says_so(tmp_path, monkeypatch):
    import src.api as api

    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "missing.joblib")
    c = TestClient(api.create_app(audit=AuditLog(None), api_keys=[]))
    assert c.get("/health").json() == {"status": "no_model", "model_version": None}
    assert c.post("/decision", json=GOOD).status_code == 503


def test_openapi_documents_example_and_tags(bundle):
    c, _ = make(bundle)
    spec = c.get("/openapi.json").json()
    assert "/decision" in spec["paths"] and "/decision/batch" in spec["paths"]
    assert spec["components"]["schemas"]["DecisionRequest"]["examples"][0]["age_years"] == 34

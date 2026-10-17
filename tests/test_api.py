from fastapi.testclient import TestClient

from src.api import create_app
from src.engine import DecisionEngine
from src.policy import Policy

GOOD = {
    "pan_hash": "GOODPAN1",
    "age_years": 35,
    "annual_income": 600_000,
    "loan_amount": 300_000,
    "annuity": 60_000,
    "years_employed": 8,
}


def client_for(bundle, t1, t2):
    return TestClient(create_app(DecisionEngine(bundle, Policy(t1, t2))))


def test_health(bundle):
    body = client_for(bundle, 0.05, 0.15).get("/health").json()
    assert body == {"status": "ok", "model_version": "test-model"}


def test_approve(bundle):
    r = client_for(bundle, 1.01, 1.01).post("/decision", json=GOOD)
    body = r.json()
    assert r.status_code == 200
    assert body["decision"] == "APPROVE"
    assert 0 <= body["pd"] <= 1
    assert body["failed_rules"] == []
    assert body["model_version"] == "test-model"
    assert len(body["reasons"]) <= 3


def test_refer(bundle):
    body = client_for(bundle, 0.0, 1.01).post("/decision", json=GOOD).json()
    assert body["decision"] == "REFER"
    assert body["pd"] is not None


def test_decline_by_model(bundle):
    body = client_for(bundle, 0.0, 0.0).post("/decision", json=GOOD).json()
    assert body["decision"] == "DECLINE"
    assert body["failed_rules"] == []


def test_decline_with_rule_hit_skips_model(bundle):
    body = client_for(bundle, 1.01, 1.01).post("/decision", json={**GOOD, "age_years": 20}).json()
    assert body["decision"] == "DECLINE"
    assert body["failed_rules"] == ["age_21_60"]
    assert body["pd"] is None
    assert body["reasons"][0]["reason"].startswith("Applicant age")


def test_decline_when_kyc_fails(bundle):
    body = client_for(bundle, 1.01, 1.01).post("/decision", json={**GOOD, "pan_hash": "KYCFAIL-1"}).json()
    assert body["failed_rules"] == ["kyc_verified"]
    assert body["kyc_verified"] is False


def test_zero_income_declines_cleanly(bundle):
    r = client_for(bundle, 1.01, 1.01).post("/decision", json={**GOOD, "annual_income": 0})
    assert r.status_code == 200
    assert "income_floor" in r.json()["failed_rules"]


def test_validation_error(bundle):
    r = client_for(bundle, 0.05, 0.15).post("/decision", json={**GOOD, "loan_amount": -5})
    assert r.status_code == 422


def test_same_applicant_same_answer(bundle):
    c = client_for(bundle, 0.05, 0.15)
    drop = lambda d: {k: v for k, v in d.items() if k not in ("request_id", "latency_ms")}
    assert drop(c.post("/decision", json=GOOD).json()) == drop(c.post("/decision", json=GOOD).json())

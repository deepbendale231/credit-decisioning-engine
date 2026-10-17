from dataclasses import asdict

import pytest
from sklearn.model_selection import train_test_split

from src import synthetic
from src.evaluation import make_reference
from src.features import build_features
from src.model import ModelBundle, fit_platt, train_model
from src.rules import RulesConfig, rules_pass_mask

DRIFT = ["EXT_SOURCE_MEAN", "AGE_YEARS", "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "AMT_INCOME_TOTAL", "YEARS_EMPLOYED"]

GOOD = {
    "pan_hash": "GOODPAN1",
    "age_years": 35,
    "annual_income": 600_000,
    "loan_amount": 300_000,
    "annuity": 60_000,
    "years_employed": 8,
}


@pytest.fixture(scope="session")
def bundle():
    """Small calibrated, monotone model on synthetic data: fast, deterministic, needs no real data."""
    app = synthetic.make_application_train(n=8000, seed=3)
    X = build_features(app)
    y = app["TARGET"].to_numpy()
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)
    model = train_model(Xtr, ytr, Xva, yva, n_estimators=80, early_stopping_rounds=10)
    platt = fit_platt(model.predict_proba(Xva[Xva.columns.intersection(Xtr.columns)][list(Xtr.columns)])[:, 1], yva)
    b = ModelBundle(model, "test-model", calibration=platt)
    b.meta = {
        "policy": {"t1": 0.05, "t2": 0.15},
        "rules": asdict(RulesConfig()),
        "data_source": "synthetic",
        "reference": {
            "pd": make_reference(b.predict_pd(Xva)[rules_pass_mask(Xva.assign(KYC_VERIFIED=1.0))]),
            "features": {f: make_reference(Xtr[f]) for f in DRIFT},
        },
    }
    return b


@pytest.fixture(scope="session")
def good():
    return dict(GOOD)

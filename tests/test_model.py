import numpy as np
import pandas as pd

from src.features import build_features
from src.model import fit_platt, load_bundle, save_bundle


def _row(**kw):
    base = dict(
        AMT_INCOME_TOTAL=200_000.0,
        AMT_CREDIT=400_000.0,
        AMT_ANNUITY=25_000.0,
        AMT_GOODS_PRICE=380_000.0,
        DAYS_BIRTH=-365.25 * 35,
        DAYS_EMPLOYED=-365.25 * 4,
        EXT_SOURCE_1=0.5,
        EXT_SOURCE_2=0.5,
        EXT_SOURCE_3=0.5,
    )
    return pd.DataFrame([{**base, **kw}])


def test_monotone_constraint_holds_for_external_score(bundle):
    grid = np.linspace(0.05, 0.95, 40)
    rows = pd.concat([build_features(_row(EXT_SOURCE_1=g, EXT_SOURCE_2=g, EXT_SOURCE_3=g)) for g in grid], ignore_index=True)
    pds = bundle.predict_raw(rows)
    assert (np.diff(pds) <= 1e-9).all(), "predicted risk must never rise as the external score rises"


def test_calibration_preserves_ranking_and_bounds(bundle):
    X = build_features(_row(EXT_SOURCE_1=np.nan).pipe(lambda d: pd.concat([d] * 5, ignore_index=True)))
    X["AMT_CREDIT"] = [1e5, 2e5, 4e5, 8e5, 1.6e6]
    raw, cal = bundle.predict_raw(X), bundle.predict_pd(X)
    assert ((cal > 0) & (cal < 1)).all()
    assert (np.argsort(raw, kind="stable") == np.argsort(cal, kind="stable")).all()


def test_platt_fit_recovers_identity_on_calibrated_scores():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.4, 50_000)
    y = (rng.random(50_000) < p).astype(int)
    c = fit_platt(p, y)
    assert abs(c["a"] - 1) < 0.1 and abs(c["b"]) < 0.15


def test_explain_returns_only_risk_raising_reasons_at_most_three(bundle):
    X = build_features(_row(EXT_SOURCE_1=0.05, EXT_SOURCE_2=0.05, EXT_SOURCE_3=0.05, AMT_CREDIT=2_000_000.0))
    reasons = bundle.explain(X)[0]
    assert 1 <= len(reasons) <= 3
    assert all(r["shap"] > 0 for r in reasons)


def test_missing_input_is_reported_as_missing_not_low(bundle):
    X = build_features(_row(EXT_SOURCE_1=np.nan, EXT_SOURCE_3=np.nan, DAYS_EMPLOYED=np.nan))
    texts = " ".join(r["reason"] for r in bundle.explain(X, top_k=14)[0])
    assert "Information not available" in texts or "details are missing" in texts


def test_save_load_roundtrip(bundle, tmp_path):
    save_bundle(bundle, tmp_path / "m.joblib")
    loaded = load_bundle(tmp_path / "m.joblib")
    X = build_features(_row())
    assert loaded.version == bundle.version and loaded.calibration == bundle.calibration
    assert np.allclose(loaded.predict_pd(X), bundle.predict_pd(X))
    assert loaded.meta["policy"] == bundle.meta["policy"]

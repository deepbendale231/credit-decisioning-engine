import numpy as np
import pytest

from src.policy import (
    APPROVE,
    DECLINE,
    REFER,
    Policy,
    TunedRules,
    decide,
    decide_many,
    pareto_envelope,
    pick_approve_cutoff,
    pick_refer_cutoff,
    rules_grid,
    tune_rules_for_approval,
    tune_rules_only,
)


def test_decide_bands():
    p = Policy(0.05, 0.15)
    assert decide(0.0499, p) == APPROVE
    assert decide(0.05, p) == REFER  # t1 is the first REFER value
    assert decide(0.1499, p) == REFER
    assert decide(0.15, p) == DECLINE


def test_decide_many_matches_decide():
    p = Policy(0.05, 0.15)
    pds = np.array([0.01, 0.05, 0.1, 0.15, 0.9])
    assert list(decide_many(pds, p)) == [decide(x, p) for x in pds]


def test_policy_rejects_inverted_thresholds():
    with pytest.raises(ValueError):
        Policy(0.2, 0.1)


def test_approve_cutoff_respects_target():
    rng = np.random.default_rng(0)
    pd_ = rng.uniform(0, 1, 20_000)
    y = (rng.random(20_000) < pd_).astype(int)  # calibrated by construction
    t1 = pick_approve_cutoff(pd_, y, target_default_rate=0.10)
    approved = pd_ < t1
    assert 0 < approved.mean() < 1
    assert y[approved].mean() <= 0.10


def test_approve_cutoff_when_target_unreachable():
    assert pick_approve_cutoff(np.array([0.5, 0.6]), np.array([1, 1]), 0.0) == 0.0


def test_refer_band_is_sized_by_capacity():
    rng = np.random.default_rng(1)
    pd_ = rng.uniform(0, 1, 20_000)
    t1 = 0.4
    t2 = pick_refer_cutoff(pd_, t1, refer_share=0.10)
    refer_share = np.mean((pd_ >= t1) & (pd_ < t2))
    assert t1 < t2 < 1.0 and abs(refer_share - 0.10) < 0.01


def test_refer_band_edge_cases():
    pd_ = np.linspace(0, 1, 1000)
    assert pick_refer_cutoff(pd_, 0.5, refer_share=0.0) == 0.5
    assert pick_refer_cutoff(pd_, 0.95, refer_share=0.5) > 1.0 - 1e-6  # runs past the riskiest applicant
    assert Policy(0.5, pick_refer_cutoff(pd_, 0.5, 0.1)).t2 > 0.5


def _bureau_frame(n=6000, seed=0):
    from src import synthetic
    from src.features import build_features

    app = synthetic.make_application_train(n=n, seed=seed)
    X = build_features(app)
    X["KYC_VERIFIED"] = 1.0
    return X, app["TARGET"].to_numpy()


def test_rules_grid_includes_external_score_floor_and_floor_cuts_default():
    X, y = _bureau_frame()
    rows = rules_grid(X, y)
    assert {r["cfg"].min_ext_score for r in rows} >= {0.0, 0.4}
    base = next(r for r in rows if r["cfg"].min_ext_score == 0.0)
    floored = next(r for r in rows if r["cfg"].rules == base["cfg"].rules and r["cfg"].min_ext_score == 0.5)
    assert floored["approval_rate"] < base["approval_rate"] and floored["default_rate"] < base["default_rate"]


def test_tuned_rules_mask_matches_grid_row_and_missing_score_fails_floor():
    X, y = _bureau_frame()
    row = rules_grid(X, y)[37]
    assert row["cfg"].mask(X).mean() == row["approval_rate"]
    X2 = X.copy()
    X2["EXT_SOURCE_MEAN"] = np.nan
    assert not TunedRules(row["cfg"].rules, 0.3).mask(X2).any()
    assert TunedRules(row["cfg"].rules, 0.0).mask(X2).any()  # floor off: missing score is not a failure


def test_tune_rules_only_flags_unreachable_target():
    X, y = _bureau_frame()
    rows = rules_grid(X, y)
    ok = tune_rules_only(rows, target_default_rate=0.5)
    assert ok["reached"] and ok["default_rate"] <= 0.5
    best = tune_rules_only(rows, target_default_rate=0.0)
    assert not best["reached"] and best["default_rate"] == min(r["default_rate"] for r in rows)


def test_tune_rules_for_approval_minimises_default_at_matched_approval():
    X, y = _bureau_frame()
    rows = rules_grid(X, y)
    pick = tune_rules_for_approval(rows, 0.5, tolerance=0.0025)
    assert pick["reached"] and pick["approval_rate"] >= 0.5 - 0.0025
    assert pick["default_rate"] == min(r["default_rate"] for r in rows if r["approval_rate"] >= 0.5 - 0.0025)
    strict = tune_rules_for_approval(rows, 0.5, tolerance=0.0)
    assert strict["approval_rate"] >= 0.5 and strict["default_rate"] >= pick["default_rate"]
    assert not tune_rules_for_approval(rows, 2.0)["reached"]


def test_pareto_envelope_is_undominated():
    X, y = _bureau_frame()
    env = pareto_envelope(rules_grid(X, y))
    approvals = [r["approval_rate"] for r in env]
    defaults = [r["default_rate"] for r in env]
    assert approvals == sorted(approvals) and defaults == sorted(defaults)

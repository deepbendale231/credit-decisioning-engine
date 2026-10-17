import numpy as np
import pandas as pd

from src.fairness import age_band, approval_gap


def test_age_band_edges():
    bands = age_band(pd.Series([20.9, 21, 30.9, 31, 45.9, 46, 60.9, 61]))
    assert bands.isna().tolist() == [True, False, False, False, False, False, False, True]
    assert bands.dropna().astype(str).tolist() == ["21-30", "21-30", "31-45", "31-45", "46-60", "46-60"]


def test_gap_and_ratio_on_known_data():
    group = pd.Series(["F"] * 1000 + ["M"] * 1000)
    approved = np.array([True] * 600 + [False] * 400 + [True] * 400 + [False] * 600)
    y = np.zeros(2000)
    out = approval_gap(group, approved, y)
    assert out["gap_pts"] == 20.0 and out["adverse_impact_ratio"] == 0.667


def test_small_groups_are_listed_but_not_in_gap():
    group = pd.Series(["F"] * 500 + ["M"] * 500 + ["X"] * 5)
    approved = np.array([True] * 250 + [False] * 250 + [True] * 250 + [False] * 250 + [False] * 5)
    out = approval_gap(group, approved, np.zeros(1005))
    assert out["gap_pts"] == 0.0 and len(out["groups"]) == 3


def test_bootstrap_ci_brackets_gap():
    rng = np.random.default_rng(0)
    group = pd.Series(rng.choice(["F", "M"], 4000))
    approved = rng.random(4000) < np.where(group == "F", 0.55, 0.50)
    out = approval_gap(group, approved, rng.integers(0, 2, 4000), n_boot=100, seed=1)
    lo, hi = out["gap_pts_ci95"]
    assert lo <= out["gap_pts"] <= hi


def test_group_without_approvals_has_no_default_rate():
    group = pd.Series(["F"] * 200 + ["M"] * 200)
    approved = np.array([True] * 200 + [False] * 200)
    rows = {r["group"]: r for r in approval_gap(group, approved, np.zeros(400))["groups"]}
    assert rows["M"]["default_rate_among_approved"] is None

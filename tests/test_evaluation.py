import numpy as np
import pytest

from src.evaluation import (
    bootstrap_ci,
    calibration_table,
    decile_table,
    discrimination,
    ks_statistic,
    make_reference,
    psi,
    psi_against,
    psi_status,
)


def _scored(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.beta(2, 20, n)
    return (rng.random(n) < p).astype(int), p


def test_perfect_and_random_discrimination():
    y = np.array([0, 0, 1, 1])
    assert discrimination(y, np.array([0.1, 0.2, 0.8, 0.9]))["auc"] == 1.0
    assert ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    rng = np.random.default_rng(1)
    yy, pp = rng.integers(0, 2, 20_000), rng.random(20_000)
    assert abs(discrimination(yy, pp)["auc"] - 0.5) < 0.02


def test_decile_table_is_ordered_and_complete():
    y, p = _scored()
    t = decile_table(y, p)
    assert len(t) == 10 and t["n"].sum() == len(y)
    assert t["default_rate"].is_monotonic_decreasing or t["default_rate"].iloc[0] > t["default_rate"].iloc[-1]
    assert t["cum_capture"].iloc[-1] == pytest.approx(1.0) and t["lift"].iloc[0] > 1


def test_calibrated_scores_have_low_ece():
    y, p = _scored(60_000)
    _, ece = calibration_table(y, p)
    assert ece < 0.01
    _, ece_bad = calibration_table(y, np.clip(p * 3, 0, 1))
    assert ece_bad > 5 * ece


def test_psi_identical_is_zero_and_shift_is_flagged():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 20_000)
    assert psi(a, a) < 1e-6
    assert psi_status(psi(a, rng.normal(0, 1, 20_000))) == "stable"
    assert psi_status(psi(a, rng.normal(1.5, 1, 20_000))) == "action"


def test_psi_counts_missing_as_its_own_bin():
    rng = np.random.default_rng(3)
    a = rng.normal(size=10_000)
    ref = make_reference(a)
    shifted = a.copy()
    shifted[:3000] = np.nan
    assert psi_against(ref, shifted) > 0.25


def test_bootstrap_ci_contains_point_and_is_seeded():
    y, p = _scored(5000)
    from sklearn.metrics import roc_auc_score

    a = bootstrap_ci(lambda i: roc_auc_score(y[i], p[i]), len(y), n_boot=80, seed=1)
    b = bootstrap_ci(lambda i: roc_auc_score(y[i], p[i]), len(y), n_boot=80, seed=1)
    assert a == b and a["lo"] <= a["point"] <= a["hi"]

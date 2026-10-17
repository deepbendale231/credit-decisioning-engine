"""Model-quality and monitoring metrics: discrimination, calibration, lift, drift, bootstrap CIs."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve


def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    """Max separation between the cumulative score distributions of defaulters and non-defaulters."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def discrimination(y: np.ndarray, p: np.ndarray) -> dict:
    auc = float(roc_auc_score(y, p))
    return {
        "auc": round(auc, 4),
        "gini": round(2 * auc - 1, 4),
        "ks": round(ks_statistic(y, p), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "log_loss": round(float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), 4),
    }


def decile_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Riskiest decile first: default rate, lift over the base rate, share of all defaults captured."""
    frame = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    frame = frame.sort_values("p", ascending=False, kind="stable").reset_index(drop=True)
    frame["decile"] = (np.arange(len(frame)) * n_bins // len(frame)) + 1
    base = frame["y"].mean()
    out = frame.groupby("decile").agg(n=("y", "size"), avg_pd=("p", "mean"), default_rate=("y", "mean"), defaults=("y", "sum"))
    out["lift"] = out["default_rate"] / base
    out["cum_capture"] = out["defaults"].cumsum() / frame["y"].sum()
    return out.reset_index().drop(columns="defaults")


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> tuple[pd.DataFrame, float]:
    """Quantile-binned predicted vs observed default rate, and expected calibration error (ECE)."""
    frame = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), n_bins, labels=False)
    table = frame.groupby("bin").agg(n=("y", "size"), mean_pd=("p", "mean"), observed=("y", "mean")).reset_index(drop=True)
    ece = float((table["n"] * (table["mean_pd"] - table["observed"]).abs()).sum() / table["n"].sum())
    return table, ece


# ---------------------------------------------------------------- drift (PSI)
PSI_WATCH, PSI_ACTION = 0.10, 0.25  # common industry rules of thumb


def make_reference(values, bins: int = 10) -> dict:
    """Freeze a distribution for later PSI comparison: inner bin edges + bin shares + missing share."""
    v = np.asarray(values, float)
    present = v[~np.isnan(v)]
    edges = np.unique(np.quantile(present, np.linspace(0, 1, bins + 1)[1:-1]))
    return {"edges": edges.tolist(), "dist": _shares(v, edges).tolist()}


def _shares(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    v = np.asarray(values, float)
    missing = np.isnan(v)
    counts = np.bincount(np.searchsorted(edges, v[~missing], side="right"), minlength=len(edges) + 1)
    return np.append(counts, missing.sum()) / max(len(v), 1)


def psi_against(reference: dict, actual) -> float:
    """Population Stability Index of `actual` against a frozen reference."""
    exp = np.clip(np.asarray(reference["dist"], float), 1e-4, None)
    act = np.clip(_shares(np.asarray(actual, float), np.asarray(reference["edges"], float)), 1e-4, None)
    return float(np.sum((act - exp) * np.log(act / exp)))


def psi(expected, actual, bins: int = 10) -> float:
    return psi_against(make_reference(expected, bins), actual)


def psi_status(value: float) -> str:
    return "action" if value >= PSI_ACTION else "watch" if value >= PSI_WATCH else "stable"


# ------------------------------------------------------------------ bootstrap
def bootstrap_ci(stat: Callable[[np.ndarray], float], n: int, n_boot: int = 300, seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile bootstrap of a statistic that takes an index array into the test set."""
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        try:
            draws.append(stat(rng.integers(0, n, n)))
        except ValueError:  # e.g. a resample with a single class
            continue
    lo, hi = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(stat(np.arange(n))), "lo": float(lo), "hi": float(hi), "n_boot": len(draws)}

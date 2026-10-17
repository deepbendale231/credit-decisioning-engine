"""Approval-gap audit. This MEASURES gaps; it does not certify the model as fair.

Gender is not a model feature. Age is (it is a legitimate underwriting variable and is also bounded by
a hard rule), so the age-band gap is expected to be non-zero and is reported against default rates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from .evaluation import bootstrap_ci
from .features import FEATURE_COLUMNS

AGE_LABELS = ["21-30", "31-45", "46-60"]
FOUR_FIFTHS = 0.8  # US EEOC heuristic; shown only as a familiar reference point, not a legal test here


def age_band(age_years: pd.Series) -> pd.Series:
    """Label age bands 21-30 / 31-45 / 46-60; ages outside 21-60 get NaN (rule-declined)."""
    return pd.cut(age_years, bins=[20.999, 30.999, 45.999, 60.999], labels=AGE_LABELS)


def _summarise(frame: pd.DataFrame, min_group_size: int, with_auc: bool = True) -> tuple[list[dict], float | None, float | None]:
    rows = []
    for name, g in frame.groupby("group", observed=True):
        n_app = int(g["approved"].sum())
        auc = float(roc_auc_score(g["y"], g["pd"])) if with_auc and "pd" in g and g["y"].nunique() == 2 else None
        rows.append(
            {
                "group": str(name),
                "n": len(g),
                "approval_rate": round(float(g["approved"].mean()), 4),
                "default_rate_among_approved": round(float(g.loc[g["approved"], "y"].mean()), 4) if n_app else None,
                "observed_default_rate_all": round(float(g["y"].mean()), 4),
                "mean_pd": round(float(g["pd"].mean()), 4) if "pd" in g else None,
                "auc": None if auc is None else round(auc, 4),
            }
        )
    usable = [r["approval_rate"] for r in rows if r["n"] >= min_group_size]
    if len(usable) < 2 or max(usable) == 0:
        return rows, None, None
    return rows, round((max(usable) - min(usable)) * 100, 2), round(min(usable) / max(usable), 3)


def approval_gap(
    group: pd.Series,
    approved: np.ndarray,
    y: np.ndarray,
    pd_scores: np.ndarray | None = None,
    min_group_size: int = 100,
    n_boot: int = 0,
    seed: int = 0,
) -> dict:
    """Per-group approval rate, default rate among approved, mean PD and AUC; the gap in percentage points
    (max - min approval rate), the adverse-impact ratio (min / max), and an optional bootstrap CI of the gap.
    Groups smaller than `min_group_size` are listed but excluded from the gap."""
    data = {"group": group.astype("object").to_numpy(), "approved": np.asarray(approved, bool), "y": np.asarray(y, float)}
    if pd_scores is not None:
        data["pd"] = np.asarray(pd_scores, float)
    frame = pd.DataFrame(data).dropna(subset=["group"]).reset_index(drop=True)
    rows, gap, air = _summarise(frame, min_group_size)
    out = {"groups": rows, "gap_pts": gap, "adverse_impact_ratio": air}
    if n_boot and gap is not None:
        ci = bootstrap_ci(lambda idx: _summarise(frame.iloc[idx], min_group_size, False)[1], len(frame), n_boot, seed)
        out["gap_pts_ci95"] = [round(ci["lo"], 2), round(ci["hi"], 2)]
    return out


def proxy_test(X_train: pd.DataFrame, g_train: np.ndarray, X_test: pd.DataFrame, g_test: np.ndarray) -> dict:
    """Can the model's own inputs predict the protected attribute? AUC ~0.5 means they carry little
    information about it; a high AUC means gender is recoverable from 'neutral' features (proxy risk)."""
    clf = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, tree_method="hist", n_jobs=-1, random_state=0)
    clf.fit(X_train[FEATURE_COLUMNS], g_train)
    auc = float(roc_auc_score(g_test, clf.predict_proba(X_test[FEATURE_COLUMNS])[:, 1]))
    top = sorted(zip(FEATURE_COLUMNS, clf.feature_importances_), key=lambda kv: -kv[1])[:3]
    return {
        "gender_from_features_auc": round(auc, 4),
        "top_predictive_features": [name for name, _ in top],
        "reading": "0.5 = features carry no gender signal; the higher, the stronger the proxy risk.",
    }

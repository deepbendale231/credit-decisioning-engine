"""Layer B: calibrated XGBoost probability-of-default model + SHAP reason codes.

Design choices (see docs/DECISIONS.md):
  * plain log-loss objective, no scale_pos_weight, so probabilities stay interpretable;
  * monotone constraints: a higher external score can never raise predicted risk;
  * Platt scaling fitted on the validation split so PDs behave like probabilities;
  * a logistic-regression baseline is always trained so the added complexity is justified with numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .features import FEATURE_COLUMNS

# Plain-language reason codes for the features that push a PD up.
REASON_CODES = {
    "AGE_YEARS": "Age profile carries higher risk",
    "YEARS_EMPLOYED": "Short or no employment history",
    "AMT_INCOME_TOTAL": "Low income",
    "AMT_CREDIT": "Large loan amount",
    "AMT_ANNUITY": "High repayment amount",
    "CREDIT_INCOME_RATIO": "High credit-to-income ratio",
    "ANNUITY_INCOME_RATIO": "High repayment burden relative to income",
    "CREDIT_GOODS_RATIO": "Loan is large relative to the price of the goods financed",
    "CREDIT_TERM": "Loan term / repayment structure raises risk",
    "EXT_SOURCE_1": "Low external score (source 1)",
    "EXT_SOURCE_2": "Low external score (source 2)",
    "EXT_SOURCE_3": "Low external score (source 3)",
    "EXT_SOURCE_MEAN": "Low external score",
    "MISSING_COUNT": "Several application details are missing",
}

# +1: risk may only rise with the feature; -1: risk may only fall with it; 0: unconstrained.
MONOTONE = {
    "EXT_SOURCE_1": -1,
    "EXT_SOURCE_2": -1,
    "EXT_SOURCE_3": -1,
    "EXT_SOURCE_MEAN": -1,
    "CREDIT_INCOME_RATIO": 1,
    "ANNUITY_INCOME_RATIO": 1,
}

DEFAULT_PARAMS = dict(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    eval_metric="auc",
    early_stopping_rounds=40,
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)

SEARCH_SPACE = dict(
    max_depth=[3, 4, 5, 6],
    learning_rate=[0.03, 0.05, 0.08],
    min_child_weight=[3, 5, 10, 20],
    subsample=[0.7, 0.8, 0.9],
    colsample_bytree=[0.6, 0.8, 1.0],
    reg_lambda=[1.0, 2.0, 5.0, 10.0],
)


def _reason_text(feature: str, value) -> str:
    """A feature that is missing pushes risk up because it is missing, not because it is low."""
    if pd.isna(value) and feature != "MISSING_COUNT":
        return f"Information not available: {feature.replace('_', ' ').lower()}"
    return REASON_CODES[feature]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@dataclass
class ModelBundle:
    model: XGBClassifier
    version: str
    meta: dict = field(default_factory=dict)
    calibration: dict | None = None  # {"a": float, "b": float}: p = sigmoid(a * logit(raw) + b)
    _explainer: shap.TreeExplainer | None = field(default=None, repr=False)

    @staticmethod
    def _X(features: pd.DataFrame) -> pd.DataFrame:
        return features[FEATURE_COLUMNS].astype(float)  # None -> NaN, never object dtype

    def predict_raw(self, features: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._X(features))[:, 1]

    def predict_pd(self, features: pd.DataFrame) -> np.ndarray:
        raw = self.predict_raw(features)
        if not self.calibration:
            return raw
        z = self.calibration["a"] * _logit(raw) + self.calibration["b"]
        return 1 / (1 + np.exp(-z))

    @property
    def explainer(self) -> shap.TreeExplainer:
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.model)
        return self._explainer

    def shap_matrix(self, features: pd.DataFrame) -> np.ndarray:
        values = self.explainer.shap_values(self._X(features))
        if isinstance(values, list):  # older shap: one array per class
            values = values[1]
        values = np.asarray(values)
        return values[..., 1] if values.ndim == 3 else values

    def explain(self, features: pd.DataFrame, top_k: int = 3) -> list[list[dict]]:
        """Top-k features pushing each row's PD up, as plain-language reason codes."""
        X = self._X(features)
        reasons = []
        for r, row in enumerate(self.shap_matrix(features)):
            order = np.argsort(-row)[:top_k]
            reasons.append(
                [
                    {
                        "feature": FEATURE_COLUMNS[i],
                        "reason": _reason_text(FEATURE_COLUMNS[i], X.iloc[r, i]),
                        "shap": round(float(row[i]), 4),
                    }
                    for i in order
                    if row[i] > 0
                ]
            )
        return reasons


def fit_platt(raw_val: np.ndarray, y_val: np.ndarray) -> dict:
    """Two-parameter logistic recalibration. Monotone, so it never changes the ranking (AUC)."""
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(_logit(raw_val).reshape(-1, 1), y_val)
    return {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}


def monotone_constraints(enabled: bool = True) -> tuple:
    return tuple(MONOTONE.get(c, 0) if enabled else 0 for c in FEATURE_COLUMNS)


def train_model(X_train, y_train, X_val, y_val, monotone: bool = True, **overrides) -> XGBClassifier:
    params = {**DEFAULT_PARAMS, **overrides}
    if monotone:
        params["monotone_constraints"] = monotone_constraints(True)
    model = XGBClassifier(**params)
    model.fit(X_train[FEATURE_COLUMNS], y_train, eval_set=[(X_val[FEATURE_COLUMNS], y_val)], verbose=False)
    return model


def tune_model(X_train, y_train, X_val, y_val, n_trials: int = 12, seed: int = 42, monotone: bool = True):
    """Small seeded random search on validation AUC. Trial 0 is the default config, so tuning
    can only help. Returns (best_model, best_params, trials)."""
    rng = np.random.default_rng(seed)
    candidates = [{}] + [{k: v[rng.integers(len(v))] for k, v in SEARCH_SPACE.items()} for _ in range(max(n_trials - 1, 0))]
    best, trials = None, []
    for params in candidates:
        model = train_model(X_train, y_train, X_val, y_val, monotone=monotone, **params)
        auc = float(roc_auc_score(y_val, model.predict_proba(X_val[FEATURE_COLUMNS])[:, 1]))
        trials.append(
            {"params": {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}, "val_auc": round(auc, 4)}
        )
        if best is None or auc > best[0]:
            best = (auc, model, trials[-1]["params"])
    return best[1], best[2], trials


def train_baseline(X_train, y_train):
    """Interpretable benchmark: median-impute (+ missing indicators), scale, logistic regression."""
    pipe = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.5),
    )
    pipe.fit(X_train[FEATURE_COLUMNS], y_train)
    return pipe


def new_version(tag: str = "") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    return f"pd-xgb-{stamp}" + (f"-{tag}" if tag else "")


def save_bundle(bundle: ModelBundle, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": bundle.model, "version": bundle.version, "meta": bundle.meta, "calibration": bundle.calibration}, path)


def load_bundle(path: str | Path) -> ModelBundle:
    blob = joblib.load(path)
    return ModelBundle(model=blob["model"], version=blob["version"], meta=blob["meta"], calibration=blob.get("calibration"))

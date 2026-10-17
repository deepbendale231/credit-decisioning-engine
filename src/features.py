"""Cleaning + feature builder.

One function (`build_features`) is used by training, the pipeline and the API,
so there is no train/serve skew. CODE_GENDER is deliberately NOT a feature: it is
kept aside for the fairness audit.
"""

from __future__ import annotations

import pandas as pd

DAYS_EMPLOYED_PLACEHOLDER = 365243  # Home Credit's marker for "not employed"
DAYS_PER_YEAR = 365.25
EXT_COLUMNS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
RAW_COLUMNS = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    *EXT_COLUMNS,
]

# What the model sees (order matters: it is saved with the model).
FEATURE_COLUMNS = [
    "AGE_YEARS",
    "YEARS_EMPLOYED",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_GOODS_RATIO",
    "CREDIT_TERM",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "EXT_SOURCE_MEAN",
    "MISSING_COUNT",
]


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """num / den, NaN when the denominator is missing or not positive."""
    return num / den.where(den > 0)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw Home Credit-style columns -> cleaned columns + engineered features.

    Missing raw columns are treated as missing values (NaN), so the API can send a
    partial applicant. The output contains FEATURE_COLUMNS (model inputs) and nothing
    from the label or protected attributes.
    """
    raw = df.reindex(columns=RAW_COLUMNS).astype(float)
    out = pd.DataFrame(index=raw.index)

    out["AGE_YEARS"] = -raw["DAYS_BIRTH"] / DAYS_PER_YEAR
    employed = raw["DAYS_EMPLOYED"].where(raw["DAYS_EMPLOYED"] != DAYS_EMPLOYED_PLACEHOLDER)
    out["YEARS_EMPLOYED"] = -employed / DAYS_PER_YEAR

    for col in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"]:
        out[col] = raw[col]
    out["CREDIT_INCOME_RATIO"] = _ratio(raw["AMT_CREDIT"], raw["AMT_INCOME_TOTAL"])
    out["ANNUITY_INCOME_RATIO"] = _ratio(raw["AMT_ANNUITY"], raw["AMT_INCOME_TOTAL"])
    out["CREDIT_GOODS_RATIO"] = _ratio(raw["AMT_CREDIT"], raw["AMT_GOODS_PRICE"])
    out["CREDIT_TERM"] = _ratio(raw["AMT_CREDIT"], raw["AMT_ANNUITY"])

    for col in EXT_COLUMNS:
        out[col] = raw[col]
    out["EXT_SOURCE_MEAN"] = raw[EXT_COLUMNS].mean(axis=1)

    # Missing flags are counted on the cleaned raw inputs (placeholder -> missing).
    cleaned = raw.copy()
    cleaned["DAYS_EMPLOYED"] = employed
    out["MISSING_COUNT"] = cleaned.isna().sum(axis=1).astype(float)
    return out

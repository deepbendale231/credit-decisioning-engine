"""Fail loudly on bad input data instead of silently training on it."""

from __future__ import annotations

import pandas as pd

from .features import DAYS_EMPLOYED_PLACEHOLDER, EXT_COLUMNS, RAW_COLUMNS

REQUIRED_COLUMNS = ["TARGET", "CODE_GENDER", *RAW_COLUMNS]


class DataValidationError(ValueError):
    pass


def validate_training_frame(df: pd.DataFrame) -> dict:
    """Structural problems raise; soft problems are counted into the returned data-quality report."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"missing required columns: {missing}")
    if len(df) < 1_000:
        raise DataValidationError(f"only {len(df)} rows; need at least 1,000 to train and evaluate")
    if not set(df["TARGET"].dropna().unique()) <= {0, 1}:
        raise DataValidationError("TARGET must be 0/1")
    if df["TARGET"].isna().any():
        raise DataValidationError("TARGET has missing values")
    rate = float(df["TARGET"].mean())
    if not 0.005 <= rate <= 0.5:
        raise DataValidationError(f"implausible default rate {rate:.3f}")

    report = {
        "n_rows": len(df),
        "default_rate": round(rate, 4),
        "duplicate_ids": int(df["SK_ID_CURR"].duplicated().sum()) if "SK_ID_CURR" in df else None,
        "missing_rate": {c: round(float(df[c].isna().mean()), 4) for c in RAW_COLUMNS},
        "employed_placeholder_share": round(float((df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_PLACEHOLDER).mean()), 4),
        "range_violations": {
            "days_birth_not_negative": int((df["DAYS_BIRTH"] >= 0).sum()),
            "negative_income": int((df["AMT_INCOME_TOTAL"] < 0).sum()),
            "ext_source_outside_0_1": int(sum(((df[c] < 0) | (df[c] > 1)).sum() for c in EXT_COLUMNS)),
        },
        "gender_values": {str(k): int(v) for k, v in df["CODE_GENDER"].value_counts().items()},
    }
    if report["duplicate_ids"]:
        raise DataValidationError(f"{report['duplicate_ids']} duplicated SK_ID_CURR values")
    if report["range_violations"]["days_birth_not_negative"] > 0.01 * len(df):
        raise DataValidationError("DAYS_BIRTH should be negative days; >1% of rows are not")
    return report

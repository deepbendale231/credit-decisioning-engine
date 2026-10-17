"""Synthetic stand-in for the Home Credit tables (same column names).

ONLY for tests, CI and the demo before the real data is downloaded. Every number
produced from this data is labelled `data_source: synthetic` in metrics.json and
must never be quoted on a resume.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import DAYS_EMPLOYED_PLACEHOLDER


def make_application_train(n: int = 60_000, seed: int = 7, base_rate: float = 0.08) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gender = rng.choice(["F", "M"], size=n, p=[0.66, 0.34])
    age = rng.uniform(21, 68, n)
    income = np.exp(rng.normal(11.9, 0.55, n)) * np.where(gender == "M", 1.12, 1.0)
    credit = np.exp(rng.normal(13.0, 0.7, n))
    annuity = credit / rng.uniform(8, 35, n)
    goods = credit * rng.uniform(0.85, 1.0, n)

    pensioner = rng.random(n) < 0.18
    years_employed = np.where(pensioner, np.nan, rng.exponential(6.5, n))

    z = rng.normal(size=n)  # latent creditworthiness (higher = riskier)

    def ext(noise: float, missing: float) -> np.ndarray:
        v = np.clip(0.52 - 0.15 * z + rng.normal(0, noise, n), 0.01, 0.99)
        v[rng.random(n) < missing] = np.nan
        return v

    ext1, ext2, ext3 = ext(0.17, 0.56), ext(0.12, 0.002), ext(0.15, 0.20)

    logit_core = 0.55 * z + 0.35 * np.log(credit / income) - 0.012 * (age - 40) - 0.03 * np.nan_to_num(years_employed, nan=0.0)
    lo, hi = -12.0, 4.0  # bisection on the intercept to hit the target default rate
    for _ in range(40):
        mid = (lo + hi) / 2
        rate = (1 / (1 + np.exp(-(mid + logit_core)))).mean()
        lo, hi = (mid, hi) if rate < base_rate else (lo, mid)
    p = 1 / (1 + np.exp(-((lo + hi) / 2 + logit_core)))

    days_employed = np.where(pensioner, DAYS_EMPLOYED_PLACEHOLDER, -years_employed * 365.25)
    return pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(100_000, 100_000 + n),
            "TARGET": (rng.random(n) < p).astype(int),
            "CODE_GENDER": gender,
            "AMT_INCOME_TOTAL": income.round(2),
            "AMT_CREDIT": credit.round(2),
            "AMT_ANNUITY": annuity.round(2),
            "AMT_GOODS_PRICE": goods.round(2),
            "DAYS_BIRTH": (-age * 365.25).round(0),
            "DAYS_EMPLOYED": np.round(days_employed, 0),
            "EXT_SOURCE_1": ext1,
            "EXT_SOURCE_2": ext2,
            "EXT_SOURCE_3": ext3,
        }
    )


def make_previous_application(n: int = 150_000, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Consumer loans", "Cash loans", "Revolving loans"], n, p=[0.45, 0.45, 0.10])
    client = rng.choice(["Repeater", "New", "Refreshed"], n, p=[0.73, 0.18, 0.09])

    p_refused = 0.10 + np.where(client == "New", 0.17, 0.0) + np.where(contract == "Cash loans", 0.10, 0.0)
    p_canceled = np.full(n, 0.19)
    p_unused = np.full(n, 0.03)
    u = rng.random(n)
    status = np.where(
        u < p_refused,
        "Refused",
        np.where(
            u < p_refused + p_canceled, "Canceled", np.where(u < p_refused + p_canceled + p_unused, "Unused offer", "Approved")
        ),
    )
    reasons = rng.choice(["HC", "LIMIT", "SCO", "SCOFR", "XAP"], n, p=[0.45, 0.2, 0.15, 0.1, 0.1])
    return pd.DataFrame(
        {
            "SK_ID_PREV": np.arange(n),
            "SK_ID_CURR": rng.integers(100_000, 160_000, n),
            "NAME_CONTRACT_STATUS": status,
            "NAME_CONTRACT_TYPE": contract,
            "NAME_CLIENT_TYPE": client,
            "CODE_REJECT_REASON": np.where(status == "Refused", reasons, "XAP"),
            "NAME_PORTFOLIO": rng.choice(["POS", "Cash", "Cards", "XNA"], n, p=[0.45, 0.4, 0.1, 0.05]),
            "WEEKDAY_APPR_PROCESS_START": rng.choice(
                ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"], n
            ),
            "HOUR_APPR_PROCESS_START": rng.integers(0, 24, n),
        }
    )

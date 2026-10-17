import numpy as np
import pandas as pd

from src.features import DAYS_EMPLOYED_PLACEHOLDER, FEATURE_COLUMNS, build_features


def row(**kw):
    base = dict(
        AMT_INCOME_TOTAL=100_000.0,
        AMT_CREDIT=200_000.0,
        AMT_ANNUITY=20_000.0,
        AMT_GOODS_PRICE=180_000.0,
        DAYS_BIRTH=-365.25 * 40,
        DAYS_EMPLOYED=-365.25 * 5,
        EXT_SOURCE_1=0.4,
        EXT_SOURCE_2=0.5,
        EXT_SOURCE_3=0.6,
    )
    return pd.DataFrame([{**base, **kw}])


def test_columns_and_no_protected_or_label_fields():
    f = build_features(row())
    assert list(f[FEATURE_COLUMNS].columns) == FEATURE_COLUMNS
    assert not {"CODE_GENDER", "TARGET"} & set(f.columns)


def test_basic_values():
    f = build_features(row()).iloc[0]
    assert f["AGE_YEARS"] == 40
    assert f["YEARS_EMPLOYED"] == 5
    assert f["CREDIT_INCOME_RATIO"] == 2
    assert f["ANNUITY_INCOME_RATIO"] == 0.2
    assert np.isclose(f["EXT_SOURCE_MEAN"], 0.5)
    assert f["MISSING_COUNT"] == 0


def test_employment_placeholder_is_missing():
    f = build_features(row(DAYS_EMPLOYED=DAYS_EMPLOYED_PLACEHOLDER)).iloc[0]
    assert np.isnan(f["YEARS_EMPLOYED"])
    assert f["MISSING_COUNT"] == 1


def test_zero_income_gives_nan_ratio_not_inf():
    f = build_features(row(AMT_INCOME_TOTAL=0.0)).iloc[0]
    assert np.isnan(f["CREDIT_INCOME_RATIO"])


def test_partial_applicant_does_not_crash():
    f = build_features(pd.DataFrame([{"AMT_INCOME_TOTAL": 1.0, "DAYS_BIRTH": -365.25 * 30}]))
    assert f.shape[0] == 1 and f.iloc[0]["MISSING_COUNT"] >= 5

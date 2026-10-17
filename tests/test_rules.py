import pandas as pd
import pytest

from src.rules import RulesConfig, evaluate_rules, rules_pass_mask

GOOD = {
    "AGE_YEARS": 35,
    "AMT_INCOME_TOTAL": 600_000.0,
    "AMT_ANNUITY": 60_000.0,
    "AMT_CREDIT": 300_000.0,
    "KYC_VERIFIED": 1,
}


def failed(**overrides):
    result = evaluate_rules({**GOOD, **overrides})
    return [name for passed, name in result if not passed]


def test_good_applicant_passes_everything():
    assert failed() == []


@pytest.mark.parametrize("age,ok", [(20, False), (21, True), (60, True), (61, False)])
def test_age_boundaries(age, ok):
    assert (failed(AGE_YEARS=age) == []) is ok
    if not ok:
        assert failed(AGE_YEARS=age) == ["age_21_60"]


def test_zero_income_fails_without_crashing():
    names = failed(AMT_INCOME_TOTAL=0.0)
    assert "income_floor" in names and "obligation_ratio_cap" in names


def test_missing_values_fail_closed():
    assert "age_21_60" in failed(AGE_YEARS=float("nan"))
    assert "obligation_ratio_cap" in failed(AMT_ANNUITY=float("nan"))


def test_obligation_ratio_cap():
    assert failed(AMT_ANNUITY=0.5 * 600_000) == []  # exactly at the cap passes
    assert failed(AMT_ANNUITY=0.5 * 600_000 + 1) == ["obligation_ratio_cap"]


def test_kyc_required():
    assert failed(KYC_VERIFIED=0) == ["kyc_verified"]


def test_vectorised_matches_scalar():
    rows = [
        GOOD,
        {**GOOD, "AGE_YEARS": 20},
        {**GOOD, "AMT_INCOME_TOTAL": 0.0},
        {**GOOD, "KYC_VERIFIED": 0},
        {**GOOD, "AMT_CREDIT": 11 * 600_000},
    ]
    scalar = [not failed(**{k: v for k, v in r.items() if GOOD[k] != v}) for r in rows]
    assert list(rules_pass_mask(pd.DataFrame(rows))) == scalar


def test_custom_config():
    cfg = RulesConfig(min_age=25)
    assert [n for p, n in evaluate_rules({**GOOD, "AGE_YEARS": 24}, cfg) if not p] == ["age_21_60"]

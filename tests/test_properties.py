import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from src.policy import APPROVE, DECLINE, REFER, Policy, decide, decide_many
from src.rules import evaluate_rules

GOOD = {"AGE_YEARS": 35, "AMT_INCOME_TOTAL": 600_000.0, "AMT_ANNUITY": 60_000.0, "AMT_CREDIT": 300_000.0, "KYC_VERIFIED": 1}
ORDER = {APPROVE: 0, REFER: 1, DECLINE: 2}


@given(age=st.floats(0, 100, allow_nan=False))
def test_age_rule_is_exactly_the_stated_band(age):
    passed = dict((n, p) for p, n in evaluate_rules({**GOOD, "AGE_YEARS": age}))["age_21_60"]
    assert passed == (21 <= age <= 60)


@given(income=st.floats(0, 5e7), factor=st.floats(1, 20), annuity=st.floats(1, 1e6), credit=st.floats(1, 5e7))
def test_more_income_never_turns_a_pass_into_a_fail(income, factor, annuity, credit):
    base = {**GOOD, "AMT_INCOME_TOTAL": income, "AMT_ANNUITY": annuity, "AMT_CREDIT": credit}
    richer = {**base, "AMT_INCOME_TOTAL": income * factor}
    before = {n: p for p, n in evaluate_rules(base)}
    after = {n: p for p, n in evaluate_rules(richer)}
    assert all(after[n] for n, p in before.items() if p)


@given(income=st.floats(-10, 5e6), annuity=st.floats(-10, 1e6), credit=st.floats(-10, 5e7))
def test_rules_never_crash_or_return_non_bool(income, annuity, credit):
    out = evaluate_rules({**GOOD, "AMT_INCOME_TOTAL": income, "AMT_ANNUITY": annuity, "AMT_CREDIT": credit})
    assert len(out) == 5 and all(isinstance(p, bool) for p, _ in out)


@settings(max_examples=200)
@given(t1=st.floats(0, 1), width=st.floats(0, 1), pds=st.lists(st.floats(0, 1), min_size=2, max_size=40))
def test_policy_is_monotone_in_pd(t1, width, pds):
    policy = Policy(t1, min(t1 + width, 1.0))
    ranks = [ORDER[decide(p, policy)] for p in sorted(pds)]
    assert ranks == sorted(ranks)
    assert list(decide_many(np.array(pds), policy)) == [decide(p, policy) for p in pds]

"""Layer A: hard policy rules.

Lenders keep policy / regulatory rules separate from the statistical score, so a
rule failure is a DECLINE with a named rule no matter what the model says.

Each rule is a small pure function returning (passed, rule_name). They work on a
single applicant (dict of scalars) and, unchanged, on a whole DataFrame, so the
rules-only baseline uses exactly the logic the API uses.

Inputs are the cleaned columns produced by `features.build_features` plus
`KYC_VERIFIED` (1/0).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, replace

import numpy as np

RULE_TEXT = {
    "age_21_60": "Applicant age is outside the 21-60 eligibility range",
    "income_floor": "Income is below the minimum income requirement",
    "kyc_verified": "KYC could not be verified",
    "obligation_ratio_cap": "Annual repayment is too high relative to income",
    "credit_to_income_cap": "Loan amount is too high relative to income",
}


@dataclass(frozen=True)
class RulesConfig:
    min_age: float = 21
    max_age: float = 60
    min_income: float = 25_000.0  # in the dataset's (unspecified) currency units
    max_annuity_income: float = 0.50
    max_credit_income: float = 10.0

    @classmethod
    def from_dict(cls, d: Mapping) -> RulesConfig:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def _num(applicant: Mapping, key: str) -> np.ndarray:
    return np.asarray(applicant[key], dtype=float)


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """num / den; +inf when income is zero/negative (fails any cap), NaN stays NaN."""
    positive = den > 0
    with np.errstate(invalid="ignore", over="ignore"):
        return np.where(positive, num / np.where(positive, den, 1.0), np.inf)


def rule_age(a: Mapping, cfg: RulesConfig) -> tuple:
    age = _num(a, "AGE_YEARS")
    return (age >= cfg.min_age) & (age <= cfg.max_age), "age_21_60"


def rule_income_floor(a: Mapping, cfg: RulesConfig) -> tuple:
    return _num(a, "AMT_INCOME_TOTAL") >= cfg.min_income, "income_floor"


def rule_kyc(a: Mapping, cfg: RulesConfig) -> tuple:
    return _num(a, "KYC_VERIFIED") == 1, "kyc_verified"


def rule_obligation_ratio(a: Mapping, cfg: RulesConfig) -> tuple:
    ratio = _safe_ratio(_num(a, "AMT_ANNUITY"), _num(a, "AMT_INCOME_TOTAL"))
    return ratio <= cfg.max_annuity_income, "obligation_ratio_cap"


def rule_credit_to_income(a: Mapping, cfg: RulesConfig) -> tuple:
    ratio = _safe_ratio(_num(a, "AMT_CREDIT"), _num(a, "AMT_INCOME_TOTAL"))
    return ratio <= cfg.max_credit_income, "credit_to_income_cap"


RULES: list[Callable[[Mapping, RulesConfig], tuple]] = [
    rule_age,
    rule_income_floor,
    rule_kyc,
    rule_obligation_ratio,
    rule_credit_to_income,
]


def evaluate_rules(applicant: Mapping, cfg: RulesConfig | None = None) -> list[tuple[bool, str]]:
    """All rules for one applicant, as [(passed, rule_name), ...]. Missing data fails."""
    cfg = cfg or RulesConfig()
    return [(bool(passed), name) for passed, name in (rule(applicant, cfg) for rule in RULES)]


def rules_pass_mask(df, cfg: RulesConfig | None = None) -> np.ndarray:
    """Boolean array: True where an applicant passes every rule (vectorised)."""
    cfg = cfg or RulesConfig()
    masks = [np.asarray(passed, dtype=bool) for passed, _ in (rule(df, cfg) for rule in RULES)]
    return np.logical_and.reduce(masks)


def with_overrides(cfg: RulesConfig, **kwargs) -> RulesConfig:
    return replace(cfg, **kwargs)

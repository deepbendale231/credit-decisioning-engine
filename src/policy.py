"""Layer C: turn a probability of default (PD) into APPROVE / REFER / DECLINE.

    PD <  t1        -> APPROVE
    t1 <= PD < t2   -> REFER   (manual review)
    PD >= t2        -> DECLINE

Cutoffs are picked on a validation split, never on the test split.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass

import numpy as np

from .rules import RulesConfig, rules_pass_mask

APPROVE, REFER, DECLINE = "APPROVE", "REFER", "DECLINE"


@dataclass(frozen=True)
class Policy:
    t1: float
    t2: float

    def __post_init__(self):
        if self.t1 > self.t2:
            raise ValueError("t1 must be <= t2")


def decide(pd_value: float, policy: Policy) -> str:
    if pd_value < policy.t1:
        return APPROVE
    if pd_value < policy.t2:
        return REFER
    return DECLINE


def decide_many(pd_values: np.ndarray, policy: Policy) -> np.ndarray:
    pd_values = np.asarray(pd_values, dtype=float)
    return np.where(pd_values < policy.t1, APPROVE, np.where(pd_values < policy.t2, REFER, DECLINE))


def pick_approve_cutoff(pd_values: np.ndarray, y: np.ndarray, target_default_rate: float) -> float:
    """t1: approve the lowest-PD applicants as far as the observed default rate among
    the approved stays <= target. Returns 0.0 if even the safest slice misses the target."""
    pd_values, y = np.asarray(pd_values, float), np.asarray(y, float)
    order = np.argsort(pd_values, kind="stable")
    pds, ys = pd_values[order], y[order]
    cum_rate = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    ok = np.where(cum_rate <= target_default_rate)[0]
    if len(ok) == 0:
        return 0.0
    k = ok[-1] + 1
    if k == len(pds):
        return float(pds[-1]) + 1e-9  # everyone qualifies
    return float(0.5 * (pds[k - 1] + pds[k]))


def pick_refer_cutoff(pd_values: np.ndarray, t1: float, refer_share: float = 0.10) -> float:
    """t2: send the next `refer_share` of applicants above t1 to manual review, DECLINE the rest.

    Manual-review capacity is an operational constraint, so the REFER band is sized by capacity
    (share of scored applicants) rather than by a statistical rule. Returns t1 when `refer_share`
    is 0, and 1.0 (+eps) when the band would run past the riskiest applicant."""
    pd_values = np.sort(np.asarray(pd_values, float))
    below = float(np.mean(pd_values < t1))
    q = below + refer_share
    if q >= 1.0:
        return float(pd_values[-1]) + 1e-9
    return max(t1, float(np.quantile(pd_values, q)))


# ----------------------------------------------------------------- rules-only benchmark
EXT_FLOORS = [0.0, *[round(0.20 + 0.01 * i, 2) for i in range(41)]]  # 0 = no bureau-score floor; then 0.20 .. 0.60


@dataclass(frozen=True)
class TunedRules:
    """The hard rules plus the one extra lever a rules-only lender always has: a bureau / external-score floor.
    Applicants with no external score fail the floor when it is active."""

    rules: RulesConfig
    min_ext_score: float = 0.0

    def mask(self, df, base_mask: np.ndarray | None = None) -> np.ndarray:
        m = rules_pass_mask(df, self.rules) if base_mask is None else base_mask.copy()
        if self.min_ext_score > 0:
            m &= np.asarray(df["EXT_SOURCE_MEAN"], float) >= self.min_ext_score  # NaN compares False
        return m

    def as_dict(self) -> dict:
        return {**asdict(self.rules), "min_ext_score": self.min_ext_score}


def rules_grid(df, y, base: RulesConfig | None = None) -> list[dict]:
    """Approval and default rate of every rules-only configuration: income floor x repayment cap x
    credit-to-income cap x external-score floor."""
    base = base or RulesConfig()
    y = np.asarray(y, float)
    ext = np.asarray(df["EXT_SOURCE_MEAN"], float)
    rows = []
    for min_income, max_ann, max_cred in itertools.product(
        [25_000.0, 50_000.0, 75_000.0, 100_000.0, 150_000.0],
        [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
        [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
    ):
        rcfg = RulesConfig(base.min_age, base.max_age, min_income, max_ann, max_cred)
        base_mask = rules_pass_mask(df, rcfg)
        for floor in EXT_FLOORS:
            mask = base_mask & (ext >= floor) if floor > 0 else base_mask
            if mask.any():
                rows.append(
                    {
                        "cfg": TunedRules(rcfg, floor),
                        "approval_rate": float(mask.mean()),
                        "default_rate": float(y[mask].mean()),
                    }
                )
    return rows


def pareto_envelope(rows: list[dict]) -> list[dict]:
    """Keep configurations that no other configuration beats on both approval (higher) and default (lower)."""
    out, best_approval = [], -1.0
    for r in sorted(rows, key=lambda r: (r["default_rate"], -r["approval_rate"])):
        if r["approval_rate"] > best_approval:
            out.append(r)
            best_approval = r["approval_rate"]
    return out


def tune_rules_only(rows: list[dict], target_default_rate: float) -> dict:
    """Rules-only configuration with the largest approval rate whose default rate is <= target. If none
    qualifies it falls back to the lowest default rate and `reached` is False: callers must say so."""
    feasible = [r for r in rows if r["default_rate"] <= target_default_rate]
    best = max(feasible, key=lambda r: r["approval_rate"]) if feasible else min(rows, key=lambda r: r["default_rate"])
    return {**best, "reached": bool(feasible)}


def tune_rules_for_approval(rows: list[dict], min_approval_rate: float, tolerance: float = 0.0025) -> dict:
    """Rules-only configuration with the LOWEST default rate among those approving at least
    `min_approval_rate - tolerance`. The tolerance lets the benchmark approve slightly FEWER applicants than
    rules + ML, which can only flatter the benchmark, so any advantage reported for the model is conservative."""
    feasible = [r for r in rows if r["approval_rate"] >= min_approval_rate - tolerance]
    best = min(feasible, key=lambda r: r["default_rate"]) if feasible else max(rows, key=lambda r: r["approval_rate"])
    return {**best, "reached": bool(feasible)}

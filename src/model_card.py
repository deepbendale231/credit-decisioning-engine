"""Render reports/MODEL_CARD.md from metrics.json, so the card can never disagree with the numbers."""

from __future__ import annotations

from pathlib import Path

from .narrative import insight_sentence, pct, policy_rows, rng, sentences


def render(m: dict) -> str:
    meta, d, c = m["meta"], m["discrimination_test"], m["comparison"]
    models = m["models"]["comparison"]
    f_ml = m["fairness"]["rules_plus_ml"]
    policy_table = "\n".join(
        f"| {label} | {pct(b['approval_rate'])} ({rng(b['approval_rate_ci95'])}) | "
        f"{pct(b['default_rate_among_approved'])} ({rng(b['default_rate_ci95'])}) |"
        for label, b in policy_rows(c)
    )
    policy_text = "\n\n".join([*sentences(c), *([insight_sentence(m)] if insight_sentence(m) else [])])
    banner = (
        ""
        if meta["resume_safe"]
        else (
            "> **WARNING: generated from SYNTHETIC data. Plumbing check only; none of these numbers describe real credit risk.**\n\n"
        )
    )
    gap = lambda a: a["gap_pts"] if a["gap_pts"] is not None else "n/a"
    ci = lambda a: f"{a['gap_pts_ci95'][0]} to {a['gap_pts_ci95'][1]}" if "gap_pts_ci95" in a else "n/a"
    top = ", ".join(f"{k} ({v:.3f})" for k, v in list(m["explainability"]["mean_abs_shap"].items())[:5])
    return f"""# Model card: probability-of-default model

{banner}**Model version:** `{meta["model_version"]}` &nbsp;|&nbsp; **Data:** {meta["data_source"]} ({meta["n_rows"]:,} rows) &nbsp;|&nbsp; **Generated:** {meta["generated_at"]} &nbsp;|&nbsp; **Commit:** {meta["git_sha"] or "n/a"}

## Intended use
Rank personal-loan applicants who already pass hard eligibility rules by probability of default (PD), to drive an
APPROVE / REFER / DECLINE policy and to produce reason codes. A portfolio / demonstration project built on the public
Kaggle *Home Credit Default Risk* data. **Not validated for, and must not be used for, real lending decisions.**

## Data and split
Stratified 60 / 20 / 20 train / validation / test split, seed {meta["seed"]} ({meta["n_train"]:,} / {meta["n_val"]:,} / {meta["n_test"]:,}).
Overall default rate {pct(meta["default_rate_overall"])}. Validation is used for early stopping, hyper-parameter search, Platt
calibration and policy cutoffs. **All numbers below are on the untouched test split.**
Gender is excluded from the features and kept only for the fairness audit.

## Model
XGBoost with monotone constraints (a higher external score never raises predicted risk; higher credit- and repayment-to-income never lower it),
log-loss objective, {len(m["models"]["tuning_trials"])} seeded random-search trials on validation AUC, Platt-calibrated.

| Model | Test AUC |
|---|---|
| Logistic regression baseline | {models["logistic_regression"]["auc_test"]} |
| XGBoost, unconstrained, default params | {models["xgboost_unconstrained"]["auc_test"]} |
| **XGBoost, monotone, tuned (production)** | **{models["xgboost_monotone_tuned"]["auc_test"]}** |

## Performance (test)
AUC {d["auc"]} (95% CI {d["auc_ci95"][0]:.3f}-{d["auc_ci95"][1]:.3f}), Gini {d["gini"]}, KS {d["ks"]}, PR-AUC {d["pr_auc"]}, Brier {d["brier"]}.
Calibration error (ECE): {m["calibration"]["ece_uncalibrated"]} before, {m["calibration"]["ece_calibrated"]} after Platt scaling.

## Decision policy
PD < **{m["policy"]["t1"]}** -> APPROVE; PD < **{m["policy"]["t2"]}** -> REFER (manual review); otherwise DECLINE. Hard rules run first and always win.

| Policy (test) | Approval rate (95% CI) | Default rate among approved (95% CI) |
|---|---|---|
{policy_table}

{policy_text}

## Explainability
Top drivers by mean |SHAP|: {top}. Each decision returns up to 3 plain-language reason codes from SHAP; a missing input is reported as missing, not as "low".

## Fairness audit (rules + ML policy, test)
| Attribute | Approval gap, pts (95% CI) | Adverse-impact ratio |
|---|---|---|
| Gender (F / M) | {gap(f_ml["gender"])} ({ci(f_ml["gender"])}) | {f_ml["gender"]["adverse_impact_ratio"]} |
| Age band (21-30 / 31-45 / 46-60) | {gap(f_ml["age_band"])} ({ci(f_ml["age_band"])}) | {f_ml["age_band"]["adverse_impact_ratio"]} |

Proxy test: gender predicted from the model's own inputs with AUC {m["fairness"]["proxy_test"]["gender_from_features_auc"]} (0.5 = no signal).
This **measures** gaps; it does not show the model is fair. Age is a model input and is bounded by a hard rule, so an age-band gap is expected and should be read next to per-group default rates (in `metrics.json`).

## Limitations
- Labels exist only for loans that were originally granted: **selection bias, no reject inference**. Approval rates are not what a live portfolio would see.
- Rule thresholds are illustrative policy parameters, not a real lender's policy.
- Bureau and e-KYC are mocks; the bureau-to-`EXT_SOURCE_2` mapping used when no external score is supplied is unvalidated.
- No temporal validation (the data has no application date), so out-of-time stability is unknown.
- Monitoring is PSI-based and manual (`python -m src.monitor`); there is no automated retraining or champion-challenger.
"""


def write_model_card(metrics: dict, path: Path) -> None:
    Path(path).write_text(render(metrics))

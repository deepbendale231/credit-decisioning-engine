# Model card: probability-of-default model

**Model version:** `pd-xgb-20261017-1049-home_credit` &nbsp;|&nbsp; **Data:** home_credit (307,511 rows) &nbsp;|&nbsp; **Generated:** 2026-10-17T10:49:59+00:00 &nbsp;|&nbsp; **Commit:** n/a

## Intended use
Rank personal-loan applicants who already pass hard eligibility rules by probability of default (PD), to drive an
APPROVE / REFER / DECLINE policy and to produce reason codes. A portfolio / demonstration project built on the public
Kaggle *Home Credit Default Risk* data. **Not validated for, and must not be used for, real lending decisions.**

## Data and split
Stratified 60 / 20 / 20 train / validation / test split, seed 42 (184,506 / 61,502 / 61,503).
Overall default rate 8.1%. Validation is used for early stopping, hyper-parameter search, Platt
calibration and policy cutoffs. **All numbers below are on the untouched test split.**
Gender is excluded from the features and kept only for the fairness audit.

## Model
XGBoost with monotone constraints (a higher external score never raises predicted risk; higher credit- and repayment-to-income never lower it),
log-loss objective, 12 seeded random-search trials on validation AUC, Platt-calibrated.

| Model | Test AUC |
|---|---|
| Logistic regression baseline | 0.74 |
| XGBoost, unconstrained, default params | 0.7613 |
| **XGBoost, monotone, tuned (production)** | **0.763** |

## Performance (test)
AUC 0.763 (95% CI 0.756-0.769), Gini 0.526, KS 0.3919, PR-AUC 0.2469, Brier 0.0676.
Calibration error (ECE): 0.0029 before, 0.0021 after Platt scaling.

## Decision policy
PD < **0.1954** -> APPROVE; PD < **0.2666** -> REFER (manual review); otherwise DECLINE. Hard rules run first and always win.

| Policy (test) | Approval rate (95% CI) | Default rate among approved (95% CI) |
|---|---|---|
| Current rules only | 85.6% (85.2-85.8%) | 8.6% (8.3-8.8%) |
| Tuned rules only, same 6.4% target default rate | 73.5% (73.0-73.9%) | 6.3% (6.1-6.5%) |
| Tuned rules only, same 77.3% approval rate as rules + ML | 77.7% (77.3-78.0%) | 6.9% (6.7-7.2%) |
| Rules + calibrated XGBoost | 77.3% (77.0-77.6%) | 6.3% (6.1-6.5%) |

At a matched approval rate (rules + ML 77.3%, best tuned rules-only 77.7%; matched within 0.25 pts on validation), the best rules-only policy (income floor, repayment cap, loan-to-income cap and a bureau-score floor, all tuned on validation) has a 6.9% default rate against 6.3% for rules + ML: **0.59 pts lower** (95% CI 0.47 to 0.7).

At the same ~6.4% default rate, rules + ML approves **3.84 pts more applicants** than the best rules-only policy (95% CI 3.63 to 4.05).

Versus the current rules, rules + ML lowers the default rate among approved by 2.26 pts at a 77.3% approval rate.

**Policy insight:** the 21-60 age rule removes the safest group. Applicants over 60 are 11.5% of applications and default at 4.8%, against 8.5% inside the band, so the rule pushes portfolio default up (overall 8.1%). Worth a business and legal review.

## Explainability
Top drivers by mean |SHAP|: EXT_SOURCE_MEAN (0.533), CREDIT_TERM (0.245), EXT_SOURCE_3 (0.171), YEARS_EMPLOYED (0.131), EXT_SOURCE_1 (0.124). Each decision returns up to 3 plain-language reason codes from SHAP; a missing input is reported as missing, not as "low".

## Fairness audit (rules + ML policy, test)
| Attribute | Approval gap, pts (95% CI) | Adverse-impact ratio |
|---|---|---|
| Gender (F / M) | 3.93 (3.24 to 4.57) | 0.951 |
| Age band (21-30 / 31-45 / 46-60) | 3.73 (2.79 to 4.52) | 0.957 |

Proxy test: gender predicted from the model's own inputs with AUC 0.7364 (0.5 = no signal).
This **measures** gaps; it does not show the model is fair. Age is a model input and is bounded by a hard rule, so an age-band gap is expected and should be read next to per-group default rates (in `metrics.json`).

## Limitations
- Labels exist only for loans that were originally granted: **selection bias, no reject inference**. Approval rates are not what a live portfolio would see.
- Rule thresholds are illustrative policy parameters, not a real lender's policy.
- Bureau and e-KYC are mocks; the bureau-to-`EXT_SOURCE_2` mapping used when no external score is supplied is unvalidated.
- No temporal validation (the data has no application date), so out-of-time stability is unknown.
- Monitoring is PSI-based and manual (`python -m src.monitor`); there is no automated retraining or champion-challenger.

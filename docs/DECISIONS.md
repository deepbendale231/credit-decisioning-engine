# Design decisions

Short records of the choices a reviewer is most likely to question.

## 1. Rules and a model, not one or the other
**Decision:** hard policy rules run first and always win; the model ranks risk among eligible applicants.
**Why:** lenders keep regulatory and risk-appetite rules (age band, income floor, KYC, repayment burden) separate from a statistical score. Rules are auditable and change without retraining; a score cannot override them.
**Cost:** rules can only *remove* applicants, so "rules + ML" can never approve more than rules alone.

## 2. Compare against the best rules-only policy, not the current rules
**Decision:** the headline comparison is rules + ML vs the best rules-only policy: income floor, repayment cap, loan-to-income cap **and a bureau / external-score floor**, grid-searched on validation. Reported two ways: at a matched approval rate, and at the same target default rate.
**Why:** the layered policy is always a subset of the current rules, so comparing with them is apples to oranges. An early version searched only three thresholds; on real data no rules-only configuration could even reach the target default rate, which made the model look 60 points better. A lender always has a score cutoff, so leaving it out was a strawman. Adding it shrank the advantage to a few points. That is the number worth defending.
**How it is guarded:** the table states when a benchmark misses its target (`target_reached_on_validation`), and the matched-approval benchmark may differ by at most 0.25 pts from the model's approval rate.
**Cost:** the result depends on the levers searched. A richer rules engine would narrow the gap; the README says so.

## 3. Cutoffs on validation, reported on test
**Decision:** t1, t2, the Platt calibrator and the tuned-rules thresholds are all chosen on the validation split. Every reported number comes from the untouched test split, with bootstrap 95% CIs.
**Why:** choosing a cutoff on the data you report on inflates the headline.

## 4. Monotone constraints
**Decision:** XGBoost with constraints: risk cannot fall as credit-to-income / repayment-to-income rise, and cannot rise as external scores rise.
**Why:** a risk committee will not accept a model where a better bureau score makes an applicant riskier. Unconstrained AUC is reported next to it, so the price is visible.

## 5. Platt calibration, not isotonic
**Decision:** two-parameter logistic recalibration fitted on validation.
**Why:** monotone and smooth (no ties in PD, so thresholds behave), cannot overfit with two parameters, and leaves the ranking (AUC) untouched. Only helps when the raw model is mis-calibrated; the card reports ECE before and after, including when it does not help.

## 6. No `scale_pos_weight`
**Decision:** plain log-loss.
**Why:** re-weighting classes distorts predicted probabilities, and the cutoff logic reads PD as a probability.

## 7. Logistic-regression baseline, always
**Decision:** a simple baseline is trained and reported on every run.
**Why:** complexity should be justified with a number. If XGBoost does not beat the baseline on real data, the model card shows it.

## 8. Gender excluded, still audited
**Decision:** gender is not a feature; it is used only to measure approval gaps, plus a proxy test (can the model's inputs predict gender?).
**Why:** excluding a protected attribute does not remove proxies. Age *is* a feature (legitimate underwriting variable, also bounded by a hard rule), so the age gap is expected and reported next to per-group default rates.
**Not claimed:** that the model is fair. Only that the gaps are measured.

## 9. Drift references match the population being monitored
**Decision:** the PD reference is built from rules-passing applicants (the scored population); feature references from all applicants.
**Why:** an earlier version compared served (rules-passed) scores to an all-applicant reference and raised false alarms on age. Found by a test and fixed.

## 10. One feature function, one rules config in the model bundle
**Decision:** `build_features` is used by training, the pipeline and the API; the rules used in evaluation are stored in the model bundle and loaded by the engine.
**Why:** no train/serve skew, and the rules that were evaluated are the rules that are served.

## 11. Audit log stores inputs and features, not the PAN
**Decision:** JSONL with request inputs, engineered features, rule results, PD, decision, thresholds and model version. Only an 8-character prefix of the (already hashed) PAN.
**Why:** any decision can be replayed and reviewed; drift monitoring reads the same file. Production would use an append-only store with retention and access control.

## 12. Synthetic data is stamped
**Decision:** `--synthetic` runs write `data_source: synthetic`, `resume_safe: false`, a banner in the model card and notebooks, and `results_table --write-readme` refuses to publish them.
**Why:** a plumbing check must never be mistaken for a result.

## 13. REFER band sized by capacity
**Decision:** t2 is set so the band between t1 and t2 holds a fixed share of scored applicants (default 5%, `--refer-share`); everyone above is declined.
**Why:** manual review is a capacity constraint, not a statistical one. A first version declined the tail whose default rate was at least twice the base rate; on real data that tail was wider than the approve region, so t2 collapsed onto t1 and nobody was ever referred. A second trap: a 10% review share, when 90% of scored applicants are already approved, leaves nobody to decline, so the model never declines on its own. The default is 5%. Both were found by reading real output, not by a test.

## 14. Build configs with keywords, not position
**Decision:** `RunConfig` is built from keyword arguments only, and a test asserts CLI defaults equal dataclass defaults.
**Why:** adding a field once shifted every later positional argument, silently turning 12 tuning trials into 300 and the review share into 12. The run still "worked", only 15 times slower. Caught by tracing a stalled run.

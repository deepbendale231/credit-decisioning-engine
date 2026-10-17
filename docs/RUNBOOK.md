# Runbook

## Retrain and ship a new model
```bash
make pipeline          # real data -> reports/ + models/model.joblib (new model_version)
make results           # README block from metrics.json
make test lint
docker build -t credit-decisioning-engine . && docker run -p 8000:8000 credit-decisioning-engine
```
Check `reports/MODEL_CARD.md`: AUC CI, calibration (ECE), fairness gaps, `drift_check.status`. Do not ship a model whose test AUC CI overlaps the logistic baseline without saying so.

## Roll back
The bundle is one file (`models/model.joblib`) containing model, calibration, thresholds, rules and drift references. Redeploy the previous image or restore the previous file; `/model-info` shows the live `model_version`.

## Change a threshold or a rule
- **Policy cutoffs (t1, t2):** re-run the pipeline with a different `--target-reduction` (approval cutoff t1) or `--refer-share` (manual-review capacity, sets t2). They are chosen on validation, so do not hand-edit them in production.
- **Rules:** edit `RulesConfig` defaults (or pass overrides); the pipeline stores the rules it evaluated in the bundle, and the engine serves exactly those.

## Monitor
```bash
python -m src.monitor --log logs/decisions.jsonl     # exit code 1 on alert
```
| PSI | Meaning | Action |
|---|---|---|
| < 0.10 | stable | none |
| 0.10 - 0.25 | watch | check the input source; compare decision mix |
| > 0.25 | action | investigate upstream data change; consider retraining |

Needs at least 200 scored decisions. Also watch the decision mix (`decision_mix`) and `rule_decline_share`: a jump in rule declines usually means an upstream field changed.

## Incident: wrong decisions reported
1. Get the `request_id` (returned in the body and `X-Request-ID` header).
2. Find the record in the audit log: it holds the inputs, features, rule results, PD and `model_version`.
3. Replay: `DecisionEngine(load_bundle(path)).decide(record["request"] | {"pan_hash": "<hash>"})` must reproduce the decision.

## Configuration
| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `models/model.joblib` | model bundle |
| `AUDIT_LOG_PATH` | `logs/decisions.jsonl` | empty string disables the audit log |
| `API_KEYS` | unset | comma-separated; when set, `/decision`, `/decision/batch`, `/stats` require `X-API-Key` |
| `PORT` | `8000` | container listen port |

## Not production-ready (known gaps)
Reject inference; out-of-time validation (the data has no application date); automated retraining and champion-challenger; a real bureau and e-KYC vendor behind the adapters; rate limiting and TLS termination (put it behind a gateway); a proper append-only audit store with retention and access control.

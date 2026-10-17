"""End-to-end pipeline. Every resume number comes from this file's output (reports/metrics.json).

    python -m src.pipeline --data-dir data/            # real Home Credit CSVs
    python -m src.pipeline --synthetic                 # plumbing check only, stamped resume_safe=false

Stages: validate -> funnel -> features/split -> rules baselines -> models (logreg, XGBoost, tuned monotone
XGBoost) -> Platt calibration -> policy cutoffs (validation) -> test evaluation with bootstrap CIs ->
fairness + proxy test -> SHAP -> drift references -> figures, model card, metrics.json, model bundle.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from . import plotting, synthetic
from .evaluation import bootstrap_ci, calibration_table, decile_table, discrimination, make_reference, psi_against, psi_status
from .fairness import age_band, approval_gap, proxy_test
from .features import FEATURE_COLUMNS, build_features
from .funnel import run_funnel, save_funnel_outputs
from .model import (
    MONOTONE,
    ModelBundle,
    fit_platt,
    new_version,
    save_bundle,
    train_baseline,
    train_model,
    tune_model,
)
from .policy import (
    APPROVE,
    Policy,
    decide_many,
    pareto_envelope,
    pick_approve_cutoff,
    pick_refer_cutoff,
    rules_grid,
    tune_rules_for_approval,
    tune_rules_only,
)
from .rules import RulesConfig, rules_pass_mask
from .validation import validate_training_frame

APP_COLUMNS = [
    "SK_ID_CURR",
    "TARGET",
    "CODE_GENDER",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
]
DRIFT_FEATURES = [
    "EXT_SOURCE_MEAN",
    "AGE_YEARS",
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "AMT_INCOME_TOTAL",
    "YEARS_EMPLOYED",
]


@dataclass
class RunConfig:
    data_dir: Path = Path("data")
    synthetic: bool = False
    out_dir: Path = Path("reports")
    model_path: Path = Path("models/model.joblib")
    target_reduction: float = 0.25  # target default = rules-only default x (1 - this)
    refer_share: float = 0.05  # share of scored applicants sent to manual review (capacity)
    n_trials: int = 12
    n_boot: int = 300
    seed: int = 42
    synthetic_rows: int = 60_000
    synthetic_prev_rows: int = 150_000
    rules: RulesConfig = field(default_factory=RulesConfig)


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def _stats(approved: np.ndarray, y: np.ndarray) -> dict:
    return {
        "approval_rate": round(float(approved.mean()), 4),
        "default_rate_among_approved": round(float(y[approved].mean()), 4) if approved.any() else None,
        "n_approved": int(approved.sum()),
    }


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (Path,)):
        return str(o)
    raise TypeError(type(o))


def _ci(stat, n, cfg: RunConfig, scale: float = 1.0, digits: int = 4) -> dict:
    c = bootstrap_ci(stat, n, cfg.n_boot, cfg.seed)
    return {"point": round(c["point"] * scale, digits), "ci95": [round(c["lo"] * scale, digits), round(c["hi"] * scale, digits)]}


def _sweep(pd_test: np.ndarray, y: np.ndarray, rules_pass: np.ndarray, points: int = 160) -> pd.DataFrame:
    """Rules + ML policy at many cutoffs: approval rate (of all applicants) and default rate among approved."""
    cuts = np.unique(np.quantile(pd_test[rules_pass], np.linspace(0.01, 1.0, points)))
    rows = []
    for t in cuts:
        a = rules_pass & (pd_test <= t)
        if a.sum() >= 30:
            rows.append({"cutoff": t, "approval_rate": a.mean(), "default_rate": y[a].mean()})
    return pd.DataFrame(rows)


def run(cfg: RunConfig) -> dict:
    t0 = time.time()
    out_dir = cfg.out_dir
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    # ---- 1. data ------------------------------------------------------------------------------------
    if cfg.synthetic:
        app = synthetic.make_application_train(n=cfg.synthetic_rows, seed=cfg.seed)
        prev = synthetic.make_previous_application(n=cfg.synthetic_prev_rows, seed=cfg.seed + 4)
        source = "synthetic"
    else:
        app = pd.read_csv(cfg.data_dir / "application_train.csv", usecols=APP_COLUMNS)
        prev_path = cfg.data_dir / "previous_application.csv"
        prev = prev_path if prev_path.exists() else None
        source = "home_credit"
    data_quality = validate_training_frame(app)
    _log(f"loaded {len(app):,} rows ({source}); validation passed", t0)

    # ---- 2. funnel ----------------------------------------------------------------------------------
    funnel_summary = None
    if prev is not None:
        funnel = run_funnel(prev)
        save_funnel_outputs(funnel, out_dir)
        funnel_summary = funnel["summary"]
        _log(f"funnel: {funnel_summary['total_rows']:,} previous applications", t0)

    # ---- 3. features + split (stratified, fixed seed, 60/20/20) ------------------------------------
    feats = build_features(app)
    feats["KYC_VERIFIED"] = 1.0  # training data has no KYC field: assume verified
    y = app["TARGET"].to_numpy()
    idx = np.arange(len(app))
    i_train, i_rest = train_test_split(idx, test_size=0.4, stratify=y, random_state=cfg.seed)
    i_val, i_test = train_test_split(i_rest, test_size=0.5, stratify=y[i_rest], random_state=cfg.seed)
    X = {"train": feats.iloc[i_train], "val": feats.iloc[i_val], "test": feats.iloc[i_test]}
    Y = {"train": y[i_train], "val": y[i_val], "test": y[i_test]}
    n_test = len(i_test)

    # ---- 4. rules-only baseline (the current policy) ------------------------------------------------
    rules_cfg = cfg.rules
    pass_mask = {k: rules_pass_mask(X[k], rules_cfg) for k in X}
    current = {k: _stats(pass_mask[k], Y[k]) for k in ("val", "test")}

    # ---- 5. models ----------------------------------------------------------------------------------
    baseline = train_baseline(X["train"], Y["train"])
    xgb_free = train_model(X["train"], Y["train"], X["val"], Y["val"], monotone=False)
    model, best_params, trials = tune_model(X["train"], Y["train"], X["val"], Y["val"], cfg.n_trials, cfg.seed)
    _log(f"models trained ({len(trials)} tuning trials); best params {best_params or 'defaults'}", t0)

    def raw(m, split):
        return m.predict_proba(X[split][FEATURE_COLUMNS])[:, 1]

    comparison_models = {}
    for name, m in [("logistic_regression", baseline), ("xgboost_unconstrained", xgb_free), ("xgboost_monotone_tuned", model)]:
        comparison_models[name] = {
            "auc_val": round(float(roc_auc_score(Y["val"], raw(m, "val"))), 4),
            "auc_test": round(float(roc_auc_score(Y["test"], raw(m, "test"))), 4),
        }

    # ---- 6. calibration (Platt, fitted on validation) -----------------------------------------------
    platt = fit_platt(raw(model, "val"), Y["val"])
    bundle = ModelBundle(model, "tmp", calibration=platt)
    pd_val, pd_test = bundle.predict_pd(X["val"]), bundle.predict_pd(X["test"])
    raw_test = raw(model, "test")
    cal_raw, ece_raw = calibration_table(Y["test"], raw_test)
    cal_cal, ece_cal = calibration_table(Y["test"], pd_test)

    # ---- 7. policy: cutoffs picked on VALIDATION, evaluated on TEST ---------------------------------
    d0_val = current["val"]["default_rate_among_approved"]
    target = d0_val * (1 - cfg.target_reduction)
    v_pass = pass_mask["val"]
    t1 = pick_approve_cutoff(pd_val[v_pass], Y["val"][v_pass], target)
    t2 = pick_refer_cutoff(pd_val[v_pass], t1, cfg.refer_share)
    policy = Policy(t1, t2)

    decision_test = np.where(pass_mask["test"], decide_many(pd_test, policy), "DECLINE")
    approved_ml = decision_test == APPROVE
    yt = Y["test"]

    # Rules-only benchmarks, tuned on VALIDATION: (a) same target default rate, (b) same approval rate as rules + ML.
    # The rules get every lever a rules-only lender has, including a bureau / external-score floor.
    grid_val = rules_grid(X["val"], Y["val"], rules_cfg)
    ml_val_approval = float((v_pass & (pd_val < t1)).mean())
    pick_target = tune_rules_only(grid_val, target)
    pick_approval = tune_rules_for_approval(grid_val, ml_val_approval)
    approved_tuned = pick_target["cfg"].mask(X["test"])
    approved_same_appr = pick_approval["cfg"].mask(X["test"])

    def rate(mask, idx_):
        return mask[idx_].mean()

    def dflt(mask, idx_):
        m = mask[idx_]
        return yt[idx_][m].mean() if m.any() else np.nan

    def policy_block(mask, extra=None):
        return {
            **_stats(mask, yt),
            **(extra or {}),
            "approval_rate_ci95": _ci(lambda i: rate(mask, i), n_test, cfg)["ci95"],
            "default_rate_ci95": _ci(lambda i: dflt(mask, i), n_test, cfg)["ci95"],
        }

    comparison = {
        "target_default_rate_val": round(float(target), 4),
        "target_reduction": cfg.target_reduction,
        "current_rules_only": policy_block(pass_mask["test"]),
        "tuned_rules_only_same_target": policy_block(
            approved_tuned,
            {
                "config": pick_target["cfg"].as_dict(),
                "target_reached_on_validation": pick_target["reached"],
                "validation_default_rate": round(pick_target["default_rate"], 4),
            },
        ),
        "tuned_rules_only_same_approval": policy_block(
            approved_same_appr,
            {"config": pick_approval["cfg"].as_dict(), "matched_validation_approval_rate": round(ml_val_approval, 4)},
        ),
        "rules_plus_ml": policy_block(approved_ml),
        "approval_uplift_vs_tuned_rules_pts": _ci(lambda i: rate(approved_ml, i) - rate(approved_tuned, i), n_test, cfg, 100, 2),
        "default_cut_vs_tuned_rules_same_approval_pts": _ci(
            lambda i: dflt(approved_same_appr, i) - dflt(approved_ml, i), n_test, cfg, 100, 2
        ),
        "default_rate_cut_vs_current_pts": _ci(lambda i: dflt(pass_mask["test"], i) - dflt(approved_ml, i), n_test, cfg, 100, 2),
    }
    age_t = X["test"]["AGE_YEARS"].to_numpy()
    over_60, in_band = age_t > 60, (age_t >= 21) & (age_t <= 60)
    rules_insight = {
        "over_60_share": round(float(over_60.mean()), 4),
        "over_60_default_rate": round(float(yt[over_60].mean()), 4) if over_60.any() else None,
        "in_band_default_rate": round(float(yt[in_band].mean()), 4),
        "overall_default_rate": round(float(yt.mean()), 4),
    }
    decision_mix = {k: round(float((decision_test == k).mean()), 4) for k in ("APPROVE", "REFER", "DECLINE")}
    _log(
        f"policy: t1={t1:.4f} t2={t2:.4f}; approve {decision_mix['APPROVE']:.1%}, refer {decision_mix['REFER']:.1%}; "
        f"rules-only reached target: {pick_target['reached']}",
        t0,
    )

    # ---- 8. discrimination with CIs, deciles, frontier ---------------------------------------------
    disc = discrimination(yt, pd_test)
    disc["auc_ci95"] = _ci(lambda i: roc_auc_score(yt[i], pd_test[i]), n_test, cfg)["ci95"]
    deciles = decile_table(yt, pd_test)
    ml_curve = _sweep(pd_test, yt, pass_mask["test"])
    grid_test = rules_grid(X["test"], yt, rules_cfg)
    env = pd.DataFrame(
        [{"approval_rate": r["approval_rate"], "default_rate": r["default_rate"]} for r in pareto_envelope(grid_test)]
    )
    ml_curve.to_csv(out_dir / "frontier_ml.csv", index=False)
    env.to_csv(out_dir / "frontier_rules.csv", index=False)

    # ---- 9. fairness + proxy test --------------------------------------------------------------------
    t = X["test"].reset_index(drop=True)
    gender = app["CODE_GENDER"].iloc[i_test].reset_index(drop=True)
    ageb = age_band(t["AGE_YEARS"])
    g_mask, a_mask = gender.isin(["F", "M"]).to_numpy(), ageb.notna().to_numpy()

    def audit(approved):
        return {
            "gender": approval_gap(
                gender[g_mask].reset_index(drop=True),
                approved[g_mask],
                yt[g_mask],
                pd_test[g_mask],
                n_boot=cfg.n_boot,
                seed=cfg.seed,
            ),
            "age_band": approval_gap(
                ageb[a_mask].reset_index(drop=True),
                approved[a_mask],
                yt[a_mask],
                pd_test[a_mask],
                n_boot=cfg.n_boot,
                seed=cfg.seed,
            ),
        }

    tr_gender = app["CODE_GENDER"].iloc[i_train].to_numpy()
    tr_ok = np.isin(tr_gender, ["F", "M"])
    proxy = proxy_test(
        X["train"][tr_ok], (tr_gender[tr_ok] == "M").astype(int), t[g_mask], (gender[g_mask] == "M").astype(int).to_numpy()
    )
    fairness = {
        "rules_plus_ml": audit(approved_ml),
        "current_rules_only": audit(pass_mask["test"]),
        "proxy_test": proxy,
        "note": "Gap = max - min approval rate in percentage points. Measured, not a fairness certification. "
        "Gender is not a model feature; age is, and is also bounded by a hard rule.",
    }
    _log("fairness audit done", t0)

    # ---- 10. explainability ---------------------------------------------------------------------------
    sample = t.sample(min(3000, len(t)), random_state=cfg.seed)
    shap_values = bundle.shap_matrix(sample)
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = sorted(zip(FEATURE_COLUMNS, mean_abs.round(4).tolist()), key=lambda kv: -kv[1])
    top1 = pd.Series([FEATURE_COLUMNS[i] for i in shap_values.argmax(axis=1)]).value_counts(normalize=True).round(4)
    explain = {"mean_abs_shap": dict(importance), "top_reason_frequency": top1.to_dict()}

    # ---- 11. drift references (for src.monitor) -------------------------------------------------------
    # PD reference = the SCORED population (rules-passing applicants); feature references = all applicants.
    reference = {"pd": make_reference(pd_val[v_pass]), "features": {f: make_reference(X["train"][f]) for f in DRIFT_FEATURES}}
    drift_check = {
        "pd_psi_val_vs_test": round(psi_against(reference["pd"], pd_test[pass_mask["test"]]), 4),
        "feature_psi_train_vs_test": {f: round(psi_against(reference["features"][f], X["test"][f]), 4) for f in DRIFT_FEATURES},
    }
    drift_check["status"] = psi_status(
        max([drift_check["pd_psi_val_vs_test"], *drift_check["feature_psi_train_vs_test"].values()])
    )

    # ---- 12. artefacts ---------------------------------------------------------------------------------
    pd.DataFrame(
        {
            "pd": pd_test,
            "y": yt,
            "rules_pass": pass_mask["test"],
            "tuned_rules_pass": approved_tuned,
            "gender": gender.to_numpy(),
            "age_years": t["AGE_YEARS"].to_numpy(),
        }
    ).to_parquet(out_dir / "test_scores.parquet", index=False)
    deciles.to_csv(out_dir / "decile_table.csv", index=False)

    figs = out_dir / "figures"
    curves = {}
    for label, p in [("Logistic regression", raw(baseline, "test")), ("XGBoost (monotone, tuned)", pd_test)]:
        fpr, tpr, _ = roc_curve(yt, p)
        curves[label] = (fpr, tpr, roc_auc_score(yt, p))
    plotting.roc_plot(curves, figs / "roc.png")
    plotting.calibration_plot(cal_raw, cal_cal, figs / "calibration.png", ece_raw, ece_cal)
    plotting.decile_plot(deciles, float(yt.mean()), figs / "deciles.png")
    as_point = lambda d: {"approval_rate": d["approval_rate"], "default_rate": d["default_rate_among_approved"]}
    plotting.frontier_plot(
        ml_curve, env, as_point(comparison["current_rules_only"]), as_point(comparison["rules_plus_ml"]), figs / "frontier.png"
    )
    plotting.importance_plot([n for n, _ in importance], [v for _, v in importance], figs / "shap_importance.png")
    plotting.fairness_plot(
        {"gender": fairness["rules_plus_ml"]["gender"]["groups"], "age band": fairness["rules_plus_ml"]["age_band"]["groups"]},
        figs / "fairness.png",
    )

    version = new_version(source)
    meta = {
        "policy": {"t1": t1, "t2": t2},
        "rules": asdict(rules_cfg),
        "data_source": source,
        "feature_columns": FEATURE_COLUMNS,
        "reference": reference,
        "test_auc": disc["auc"],
    }
    bundle = ModelBundle(model, version, meta, platt)
    save_bundle(bundle, cfg.model_path)

    metrics = {
        "meta": {
            "data_source": source,
            "resume_safe": source == "home_credit",
            "n_rows": len(app),
            "n_train": len(i_train),
            "n_val": len(i_val),
            "n_test": int(n_test),
            "default_rate_overall": round(float(y.mean()), 4),
            "seed": cfg.seed,
            "model_version": version,
            "git_sha": _git_sha(),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "versions": {
                "python": sys.version.split()[0],
                "xgboost": xgboost.__version__,
                "sklearn": sklearn.__version__,
                "pandas": pd.__version__,
            },
        },
        "data_quality": data_quality,
        "funnel": funnel_summary,
        "models": {
            "comparison": comparison_models,
            "tuning_trials": trials,
            "best_params": best_params,
            "monotone_features": [c for c in FEATURE_COLUMNS if c in MONOTONE],
        },
        "discrimination_test": disc,
        "calibration": {"platt": platt, "ece_uncalibrated": round(ece_raw, 4), "ece_calibrated": round(ece_cal, 4)},
        "deciles": deciles.round(4).to_dict(orient="records"),
        "policy": {"t1": round(t1, 4), "t2": round(t2, 4), "rules": asdict(rules_cfg)},
        "comparison": comparison,
        "decision_mix_test": decision_mix,
        "rules_insight": rules_insight,
        "fairness": fairness,
        "explainability": explain,
        "drift_check": drift_check,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_jsonable))
    from .model_card import write_model_card

    write_model_card(metrics, out_dir / "MODEL_CARD.md")
    _log(f"done: metrics.json, MODEL_CARD.md, {len(list(figs.glob('*.png')))} figures, model -> {cfg.model_path}", t0)
    return metrics


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--synthetic", action="store_true", help="plumbing check only; numbers are NOT resume-safe")
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    ap.add_argument("--model-path", type=Path, default=Path("models/model.joblib"))
    ap.add_argument("--target-reduction", type=float, default=0.25, help="target default = rules-only default x (1 - this)")
    ap.add_argument("--refer-share", type=float, default=0.05, help="share of scored applicants sent to manual review")
    ap.add_argument("--n-trials", type=int, default=12)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--quick", action="store_true", help="3 tuning trials, 60 bootstrap draws")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    if not args.synthetic and not (args.data_dir / "application_train.csv").exists():
        raise SystemExit(f"{args.data_dir}/application_train.csv not found. See data/README.md or use --synthetic.")
    cfg = RunConfig(
        data_dir=args.data_dir,
        synthetic=args.synthetic,
        out_dir=args.out_dir,
        model_path=args.model_path,
        target_reduction=args.target_reduction,
        refer_share=args.refer_share,
        n_trials=3 if args.quick else args.n_trials,
        n_boot=60 if args.quick else args.n_boot,
        seed=args.seed,
    )
    m = run(cfg)
    if not m["meta"]["resume_safe"]:
        print("\n!! synthetic data: these numbers are for plumbing checks only, never for the resume !!")
    else:
        print("\nNext: python -m src.results_table --write-readme")


if __name__ == "__main__":
    main()

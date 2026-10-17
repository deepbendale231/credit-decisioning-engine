"""Generate (and optionally execute) the four analysis notebooks.

    python notebooks/build_notebooks.py             # write unexecuted notebooks
    python notebooks/build_notebooks.py --execute   # run them in place (do this AFTER `make pipeline` on real data)

Each notebook starts with a banner that says whether it is looking at real or synthetic data, so a
synthetic run can never be mistaken for a real one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent

SETUP = '''import json, sys
from pathlib import Path
ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display
pd.set_option("display.width", 140); pd.set_option("display.max_columns", 30)

METRICS = json.loads((ROOT / "reports" / "metrics.json").read_text())
SOURCE = METRICS["meta"]["data_source"]
display(Markdown(
    "> **Data source: real Home Credit data.**" if METRICS["meta"]["resume_safe"]
    else "> ⚠️ **SYNTHETIC DATA. Plumbing check only; do not read these numbers as findings.**"))

def load_application():
    """Real CSV if present and the reports came from it, otherwise the synthetic stand-in."""
    from src import synthetic
    path = ROOT / "data" / "application_train.csv"
    if METRICS["meta"]["resume_safe"] and path.exists():
        from src.pipeline import APP_COLUMNS
        return pd.read_csv(path, usecols=APP_COLUMNS)
    return synthetic.make_application_train(n=METRICS["meta"]["n_rows"], seed=METRICS["meta"]["seed"])
'''

NOTEBOOKS: dict[str, list[tuple[str, str]]] = {
    "01_funnel_analysis": [
        (
            "md",
            "# 1. Application outcome funnel\n\nWhere do past applications end up, and for whom? Source: `previous_application` "
            "(approved / refused / canceled / unused offer). **This is outcome data, not a click-by-click UX funnel.**",
        ),
        ("code", SETUP),
        (
            "code",
            """import duckdb
from src.funnel import SEGMENT_SQL, run_funnel
path = ROOT / "data" / "previous_application.csv"
if METRICS["meta"]["resume_safe"] and path.exists():
    source = path
else:
    from src import synthetic; source = synthetic.make_previous_application()
result = run_funnel(source)
con = duckdb.connect()
con.register("prev", source) if isinstance(source, pd.DataFrame) else con.execute(f"CREATE VIEW prev AS SELECT * FROM read_csv_auto('{source}')")
print("Total applications:", f"{result['summary']['total_rows']:,}")""",
        ),
        ("md", "## Overall outcomes (SQL)"),
        (
            "code",
            '''con.execute("""
SELECT NAME_CONTRACT_STATUS AS status, COUNT(*) AS n,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
FROM prev GROUP BY 1 ORDER BY n DESC""").df()''',
        ),
        ("md", "## Refusal rate by segment (SQL)"),
        (
            "code",
            """seg = result["segments"]
for dim in ["NAME_CLIENT_TYPE", "NAME_CONTRACT_TYPE", "NAME_PORTFOLIO"]:
    display(Markdown(f"**{dim}**"))
    display(seg[(seg.dimension == dim) & (seg.n >= 0.01 * result["summary"]["total_rows"])].drop(columns="dimension"))""",
        ),
        ("md", "## Does the hour or weekday matter?"),
        (
            "code",
            """hour = seg[seg.dimension == "HOUR_APPR_PROCESS_START"].assign(h=lambda d: d.segment.astype(int)).sort_values("h")
fig, ax = plt.subplots(figsize=(8, 3)); ax.bar(hour.h, hour.refused_pct, color="#2a78d6"); ax.set_xlabel("Hour"); ax.set_ylabel("% refused"); plt.show()
seg[seg.dimension == "WEEKDAY_APPR_PROCESS_START"].drop(columns="dimension")""",
        ),
        ("md", "## Findings (computed, not typed)"),
        (
            "code",
            """for line in result["summary"]["findings"]:
    print("•", line)
print(f"• {result['summary']['unused_share_of_offers_pct']}% of offers made (approved + unused) were never used.")""",
        ),
        (
            "md",
            "**Reading this honestly:** a high refusal rate for new clients says new clients are refused more often, not why. "
            "Reject reasons (`CODE_REJECT_REASON`) hint at the policy lever; proving causality would need an experiment.",
        ),
    ],
    "02_eda_features": [
        (
            "md",
            "# 2. EDA and features\n\nWhat the application table looks like and how `src/features.py` cleans it. The same function is used for training and serving.",
        ),
        ("code", SETUP),
        (
            "code",
            """from src.features import FEATURE_COLUMNS, build_features
from src.validation import validate_training_frame
app = load_application()
report = validate_training_frame(app)
print(f"{report['n_rows']:,} rows, default rate {report['default_rate']:.2%}")
pd.Series(report["missing_rate"]).sort_values(ascending=False).to_frame("missing share")""",
        ),
        (
            "md",
            "## Data quirks handled\n\n- `DAYS_BIRTH` is negative days. `DAYS_EMPLOYED == 365243` is a placeholder for *not employed* and is treated as **missing**, not as 1,000 years of work.",
        ),
        (
            "code",
            """print("DAYS_EMPLOYED placeholder share:", report["employed_placeholder_share"])
feats = build_features(app); y = app.TARGET.values
feats["TARGET"] = y
feats[["AGE_YEARS", "YEARS_EMPLOYED", "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "EXT_SOURCE_MEAN"]].describe().T""",
        ),
        ("md", "## Which features separate defaulters?"),
        (
            "code",
            """from sklearn.metrics import roc_auc_score
rows = []
for c in FEATURE_COLUMNS:
    m = feats[c].notna()
    a = roc_auc_score(y[m], feats.loc[m, c]) if m.sum() > 100 else np.nan
    rows.append({"feature": c, "single-feature AUC": round(max(a, 1 - a), 3), "missing %": round(100 * (~m).mean(), 1)})
pd.DataFrame(rows).sort_values("single-feature AUC", ascending=False)""",
        ),
        (
            "code",
            """fig, axes = plt.subplots(1, 3, figsize=(13, 3.3))
for ax, c in zip(axes, ["EXT_SOURCE_MEAN", "CREDIT_INCOME_RATIO", "AGE_YEARS"]):
    q = pd.qcut(feats[c], 10, duplicates="drop")
    (feats.groupby(q, observed=True).TARGET.mean() * 100).plot(ax=ax, marker="o", color="#2a78d6")
    ax.set_title(f"Default rate by {c} decile"); ax.set_xlabel(""); ax.set_xticklabels([]); ax.set_ylabel("% default")
plt.tight_layout(); plt.show()""",
        ),
        (
            "md",
            "## Why gender is not a feature\n\n`CODE_GENDER` stays out of the model on purpose and is used only for the fairness audit. "
            "Excluding it does not guarantee equal outcomes (other features can act as proxies), which is why notebook 04 measures the gap.",
        ),
    ],
    "03_model_and_policy": [
        (
            "md",
            "# 3. Model and policy\n\nAUC alone does not run a lending business. The policy question is: **at a given default rate, how many people do we approve?** "
            "All numbers are read from `reports/metrics.json`, produced by `python -m src.pipeline`.",
        ),
        ("code", SETUP),
        (
            "code",
            """m = METRICS
display(pd.DataFrame(m["models"]["comparison"]).T.rename(columns={"auc_val": "val AUC", "auc_test": "test AUC"}))
d = m["discrimination_test"]
print(f"Production model test AUC {d['auc']} (95% CI {d['auc_ci95']}), Gini {d['gini']}, KS {d['ks']}")
print("Tuning trials:", len(m["models"]["tuning_trials"]), "| best params:", m["models"]["best_params"] or "defaults")""",
        ),
        (
            "md",
            "## Why a monotone-constrained, calibrated model\n\nRisk may never *fall* as credit-to-income rises or *rise* as the external score rises: easier to defend to a risk committee. "
            "Platt scaling makes the PD behave like a probability so cutoffs mean something.",
        ),
        (
            "code",
            """c = m["calibration"]; print("ECE before:", c["ece_uncalibrated"], "| after Platt:", c["ece_calibrated"])
display(Image(filename=str(ROOT / "reports/figures/calibration.png"), width=420)); display(Image(filename=str(ROOT / "reports/figures/deciles.png"), width=520))""",
        ),
        (
            "md",
            "## The policy comparison\n\nRules can only *remove* applicants, so rules + ML can never approve more than the rules alone. "
            "The fair question is therefore: against the **best rules-only policy** (income floor, repayment caps, loan-to-income cap **and a bureau-score floor**, tuned on validation), how much lower is the default rate at the same approval rate, and can rules reach the model's default rate at all?",
        ),
        (
            "code",
            """c = m["comparison"]
tbl = pd.DataFrame({k: {"approval rate": v["approval_rate"], "default among approved": v["default_rate_among_approved"]}
                    for k, v in [("current rules", c["current_rules_only"]), ("tuned rules, same target", c["tuned_rules_only_same_target"]), ("tuned rules, same approval", c["tuned_rules_only_same_approval"]), ("rules + ML", c["rules_plus_ml"])]}).T
display(tbl.style.format("{:.2%}"))
print("Target reached by rules-only on validation:", c["tuned_rules_only_same_target"]["target_reached_on_validation"])
print("Default-rate cut vs tuned rules at the same approval rate (pts):", c["default_cut_vs_tuned_rules_same_approval_pts"])
print("Approval uplift vs tuned rules at the same target (pts):", c["approval_uplift_vs_tuned_rules_pts"])
display(Image(filename=str(ROOT / "reports/figures/frontier.png"), width=620))""",
        ),
        (
            "code",
            """scores = pd.read_parquet(ROOT / "reports/test_scores.parquet")
ok = scores.rules_pass.values
rows = []
for t1 in np.quantile(scores.pd[ok], [0.2, 0.4, 0.5, 0.6, 0.8, 1.0]):
    a = ok & (scores.pd.values <= t1)
    rows.append({"PD cutoff": round(t1, 4), "approval rate": a.mean(), "default among approved": scores.y.values[a].mean(),
                 "bad loans per 1,000 applicants": 1000 * (a & (scores.y.values == 1)).mean()})
pd.DataFrame(rows).style.format({"approval rate": "{:.1%}", "default among approved": "{:.2%}", "bad loans per 1,000 applicants": "{:.1f}"})""",
        ),
        (
            "md",
            f"## Caveats that matter\n\n- Training labels exist only for loans that were originally **granted** (selection bias, no reject inference).\n"
            f"- Cutoffs were picked on validation and evaluated on a separate test split; CIs are bootstrap over the test set.\n"
            f"- The policy comparison is only as good as the rules grid searched; a richer rules engine would narrow the gap.",
        ),
    ],
    "04_explain_and_fairness": [
        (
            "md",
            "# 4. Explainability and fairness\n\nEvery decision returns SHAP-based reason codes. The audit measures approval gaps by gender and age band: it **measures** them, it does not certify fairness.",
        ),
        ("code", SETUP),
        (
            "code",
            """from src.engine import DecisionEngine
from src.features import FEATURE_COLUMNS, build_features
from src.model import load_bundle
import shap
bundle = load_bundle(ROOT / "models" / "model.joblib")
app = load_application()
X = build_features(app).sample(2000, random_state=0)
sv = bundle.shap_matrix(X)
plt.figure(); shap.summary_plot(sv, X[FEATURE_COLUMNS], show=False, max_display=14); plt.tight_layout(); plt.show()""",
        ),
        ("md", "## Reason codes for three kinds of applicant"),
        (
            "code",
            """engine = DecisionEngine(bundle)
cases = {"strong": dict(pan_hash="nb-1", age_years=38, annual_income=300000, loan_amount=300000, annuity=18000, years_employed=9, ext_source_2=0.78),
         "borderline": dict(pan_hash="nb-2", age_years=29, annual_income=150000, loan_amount=500000, annuity=30000, years_employed=2, ext_source_2=0.12),
         "weak": dict(pan_hash="nb-3", age_years=22, annual_income=80000, loan_amount=650000, annuity=38000, years_employed=0.3, ext_source_2=0.10)}
for name, req in cases.items():
    out = engine.decide(req)
    print(f"{name:11s} -> {out['decision']:8s} PD={out['pd']}  reasons: {[r['reason'] for r in out['reasons']]}")""",
        ),
        ("md", "## Approval gaps (rules + ML policy, test set)"),
        (
            "code",
            """fa = METRICS["fairness"]
for attr in ["gender", "age_band"]:
    g = fa["rules_plus_ml"][attr]
    display(Markdown(f"**{attr}**: gap {g['gap_pts']} pts, 95% CI {g.get('gap_pts_ci95')}, adverse-impact ratio {g['adverse_impact_ratio']} "
                     f"(gap under rules only: {fa['current_rules_only'][attr]['gap_pts']} pts)"))
    display(pd.DataFrame(g["groups"]))""",
        ),
        (
            "code",
            """p = fa["proxy_test"]
print(f"Gender predicted from the model's own inputs: AUC {p['gender_from_features_auc']} (0.5 = no signal). Most informative: {p['top_predictive_features']}")
display(Image(filename=str(ROOT / "reports/figures/fairness.png"), width=620))""",
        ),
        (
            "md",
            "## How to read this\n\n"
            "- A gap is not automatically unfairness: compare each group's **default rate among approved** and observed default rate. If a group is approved less *and* defaults more, the gap tracks risk; if not, investigate.\n"
            "- **Age is a model input** and is also bounded by a hard rule (21-60), so an age-band gap is expected and needs a business and legal view.\n"
            "- Next steps in practice: review features correlated with the attribute, test per-group thresholds only where legally permitted, monitor the gap over time.",
        ),
    ],
}


def build(name: str, cells: list[tuple[str, str]]) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in cells]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return nb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="execute in place with outputs saved")
    ap.add_argument("--out-dir", type=Path, default=HERE)
    args = ap.parse_args()
    for name, cells in NOTEBOOKS.items():
        nb = build(name, cells)
        if args.execute:
            from nbconvert.preprocessors import ExecutePreprocessor

            ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(HERE)}})
            # Library warnings on stderr are noise in a published notebook and can leak local filesystem paths.
            for cell in nb.cells:
                if cell.cell_type == "code":
                    cell.outputs = [o for o in cell.outputs if not (o.output_type == "stream" and o.get("name") == "stderr")]
        nbf.write(nb, args.out_dir / f"{name}.ipynb")
        print(("executed " if args.execute else "wrote    ") + f"{name}.ipynb")


if __name__ == "__main__":
    main()

"""Demo app: decision card with SHAP, policy explorer, outcome funnel, model quality, fairness.

    streamlit run app/streamlit_app.py

Calls the DecisionEngine in-process (the same code the FastAPI service wraps), so it deploys as one app.
"""

import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.engine import DecisionEngine
from src.features import FEATURE_COLUMNS
from src.model import load_bundle

REPORTS = ROOT / "reports"
BLUE, ORANGE, INK = "#2a78d6", "#eb6834", "#52514e"
DECISION_COLOR = {"APPROVE": "#1baf7a", "REFER": "#eda100", "DECLINE": "#e34948"}

st.set_page_config(page_title="Credit Decisioning Engine", page_icon="🏦", layout="wide")

PRESETS = {
    "Strong applicant": dict(
        pan="DEMOSTRONG", age=38, income=300_000, loan=300_000, annuity=18_000, goods=280_000, years=9.0, ext2=0.78
    ),
    "Borderline applicant": dict(
        pan="DEMOMID", age=29, income=150_000, loan=500_000, annuity=30_000, goods=470_000, years=2.0, ext2=0.12
    ),
    "Weak applicant": dict(
        pan="DEMOWEAK", age=22, income=80_000, loan=650_000, annuity=38_000, goods=600_000, years=0.3, ext2=0.10
    ),
    "Rule hit: age 19": dict(
        pan="DEMOYOUNG", age=19, income=200_000, loan=300_000, annuity=20_000, goods=290_000, years=1.0, ext2=0.6
    ),
    "Rule hit: KYC fails": dict(
        pan="KYCFAIL-DEMO", age=40, income=250_000, loan=300_000, annuity=20_000, goods=290_000, years=8.0, ext2=0.6
    ),
}


def load_preset():
    for k, v in PRESETS[st.session_state["preset"]].items():
        st.session_state[k] = v


for k, v in PRESETS["Borderline applicant"].items():
    st.session_state.setdefault(k, v)

metrics_path = REPORTS / "metrics.json"
metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None


@st.cache_resource
def get_engine():
    return DecisionEngine(load_bundle(ROOT / "models" / "model.joblib"))


@st.cache_data
def load_scores():
    return (
        pd.read_parquet(REPORTS / "test_scores.parquet"),
        pd.read_csv(REPORTS / "frontier_ml.csv"),
        pd.read_csv(REPORTS / "frontier_rules.csv"),
    )


st.title("🏦 Credit Decisioning Engine")
st.caption(
    "Hard eligibility rules → calibrated default-risk model → policy decision with SHAP reason codes. Mock bureau and e-KYC. Not for real lending."
)
if metrics and not metrics["meta"]["resume_safe"]:
    st.warning("Running on SYNTHETIC data. Numbers are for illustration only; the real-data run is what the README reports.")
if metrics is None:
    st.error("No reports found. Run `python -m src.pipeline` first.")
    st.stop()

tab_dec, tab_pol, tab_fun, tab_q, tab_fair = st.tabs(
    ["Decision", "Policy explorer", "Outcome funnel", "Model quality", "Fairness"]
)

# ------------------------------------------------------------------------------------------- decision
with tab_dec:
    engine = get_engine()
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.selectbox("Load an example", list(PRESETS), index=1, key="preset", on_change=load_preset)
        c1, c2 = st.columns(2)
        pan = c1.text_input("PAN hash (mock; start with KYCFAIL to fail KYC)", key="pan")
        age = c2.number_input("Age", 15, 90, key="age")
        income = c1.number_input("Annual income", 0, 50_000_000, step=10_000, key="income")
        years = c2.number_input("Years employed", 0.0, 50.0, key="years")
        loan = c1.number_input("Loan amount", 1, 100_000_000, step=10_000, key="loan")
        annuity = c2.number_input("Annuity (repayment amount)", 1, 10_000_000, step=1_000, key="annuity")
        goods = c1.number_input("Goods price (0 = unknown)", 0, 100_000_000, step=10_000, key="goods")
        ext2 = c2.slider("External score (0 = use mock bureau)", 0.0, 1.0, step=0.01, key="ext2")
    out = engine.decide(
        dict(
            pan_hash=pan or "x",
            age_years=int(age),
            annual_income=float(income),
            loan_amount=float(loan),
            annuity=float(annuity),
            goods_price=float(goods) or None,
            years_employed=float(years),
            ext_source_2=ext2 or None,
        )
    )
    with right:
        color = DECISION_COLOR[out["decision"]]
        st.markdown(
            f"<div style='padding:14px 18px;border-radius:12px;background:{color}22;border-left:8px solid {color}'>"
            f"<span style='font-size:13px;color:{INK}'>Decision</span><br>"
            f"<span style='font-size:34px;font-weight:700;color:{color}'>{out['decision']}</span></div>",
            unsafe_allow_html=True,
        )
        if out["pd"] is None:
            st.markdown("**Declined by a hard rule before scoring.**")
            for r in out["reasons"]:
                st.write(f"❌ {r['reason']}")
        else:
            t1, t2 = out["thresholds"]["t1"], out["thresholds"]["t2"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Probability of default", f"{out['pd']:.1%}")
            m2.metric("Approve below", f"{t1:.1%}")
            m3.metric("Decline from", f"{t2:.1%}")
            top = max(t2 * 1.6, out["pd"] * 1.3, 0.05)
            bands = pd.DataFrame({"a": [0, t1, t2], "b": [t1, t2, top], "band": ["APPROVE", "REFER", "DECLINE"]})
            bar = (
                alt.Chart(bands)
                .mark_rect(opacity=0.85)
                .encode(
                    x=alt.X("a:Q", axis=alt.Axis(format="%", title="PD"), scale=alt.Scale(domain=[0, top])),
                    x2="b:Q",
                    y=alt.value(22),
                    y2=alt.value(46),
                    color=alt.Color(
                        "band:N", scale=alt.Scale(domain=list(DECISION_COLOR), range=list(DECISION_COLOR.values())), legend=None
                    ),
                )
            )
            xscale = alt.Scale(domain=[0, top])
            mark = (
                alt.Chart(pd.DataFrame({"pd": [out["pd"]], "label": [f"PD {out['pd']:.1%}"]}))
                .mark_rule(color="black", strokeWidth=4)
                .encode(x=alt.X("pd:Q", scale=xscale, axis=None), y=alt.value(16), y2=alt.value(52))
            )
            tag = (
                alt.Chart(pd.DataFrame({"pd": [out["pd"]], "label": [f"PD {out['pd']:.1%}"]}))
                .mark_text(fontWeight="bold", fontSize=12, baseline="bottom")
                .encode(x=alt.X("pd:Q", scale=xscale, axis=None), y=alt.value(13), text="label:N")
            )
            st.altair_chart((bar + mark + tag).properties(height=96), use_container_width=True)
            st.markdown("**Top reasons pushing risk up**")
            for r in out["reasons"] or [{"reason": "No factor pushes risk above the portfolio average."}]:
                st.write(f"• {r['reason']}")
            feats = pd.DataFrame([out["features"]])[FEATURE_COLUMNS]
            sv = engine.bundle.shap_matrix(feats)[0]
            shap_df = pd.DataFrame({"feature": FEATURE_COLUMNS, "shap": sv}).sort_values("shap")
            shap_df["effect"] = np.where(shap_df["shap"] > 0, "raises risk", "lowers risk")
            st.altair_chart(
                alt.Chart(shap_df)
                .mark_bar()
                .encode(
                    x=alt.X("shap:Q", title="SHAP contribution to log-odds of default"),
                    y=alt.Y("feature:N", sort=None, title=None, axis=alt.Axis(labelLimit=220)),
                    color=alt.Color(
                        "effect:N",
                        scale=alt.Scale(domain=["raises risk", "lowers risk"], range=[ORANGE, BLUE]),
                        legend=alt.Legend(orient="bottom", title=None),
                    ),
                    tooltip=["feature", alt.Tooltip("shap:Q", format=".3f")],
                )
                .properties(height=300),
                use_container_width=True,
            )
        with st.expander("Rule checks, bureau and KYC"):
            st.dataframe(
                pd.DataFrame(out["rule_results"]).assign(passed=lambda d: d["passed"].map({True: "✅", False: "❌"})),
                hide_index=True,
                use_container_width=True,
            )
            st.write(
                f"Mock bureau score: **{out['bureau_score']}** · KYC verified: **{out['kyc_verified']}** · model `{out['model_version']}`"
            )

# ------------------------------------------------------------------------------------ policy explorer
with tab_pol:
    scores, ml_curve, rules_curve = load_scores()
    comp = metrics["comparison"]
    st.markdown(
        "Move the approval cutoff and see the trade-off on the **held-out test set**. The orange curve is the best you can do by only tuning the rules."
    )
    pass_ = scores["rules_pass"].to_numpy()
    lo, hi = float(scores.loc[pass_, "pd"].quantile(0.02)), float(scores.loc[pass_, "pd"].quantile(0.995))
    t1 = st.slider("Approve if PD is below", lo, hi, float(metrics["policy"]["t1"]), step=0.001, format="%.3f")
    approved = pass_ & (scores["pd"].to_numpy() < t1)
    appr, dflt = approved.mean(), scores["y"].to_numpy()[approved].mean() if approved.any() else float("nan")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Approval rate", f"{appr:.1%}")
    k2.metric("Default rate among approved", f"{dflt:.2%}")
    k3.metric("Approved per 1,000 applicants", f"{appr * 1000:.0f}")
    k4.metric("Bad loans per 1,000 applicants", f"{appr * dflt * 1000:.1f}")
    cur = comp["current_rules_only"]
    ml_c = ml_curve.assign(policy="Rules + ML")
    ru_c = rules_curve.assign(policy="Tuned rules only")
    lines = (
        alt.Chart(pd.concat([ml_c, ru_c]))
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("approval_rate:Q", axis=alt.Axis(format="%", title="Approval rate (share of all applicants)")),
            y=alt.Y(
                "default_rate:Q", axis=alt.Axis(format=".1%", title="Default rate among approved"), scale=alt.Scale(zero=True)
            ),
            color=alt.Color(
                "policy:N",
                scale=alt.Scale(domain=["Rules + ML", "Tuned rules only"], range=[BLUE, ORANGE]),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
    )
    pts = (
        alt.Chart(
            pd.DataFrame(
                {
                    "approval_rate": [appr, cur["approval_rate"]],
                    "default_rate": [dflt, cur["default_rate_among_approved"]],
                    "label": ["your cutoff", "current rules"],
                }
            )
        )
        .mark_point(size=130, filled=True, color="black")
        .encode(
            x="approval_rate:Q",
            y="default_rate:Q",
            tooltip=["label", alt.Tooltip("approval_rate:Q", format=".1%"), alt.Tooltip("default_rate:Q", format=".2%")],
        )
    )
    txt = pts.mark_text(align="left", dx=10, dy=-8, fontSize=12).encode(text="label:N")
    st.altair_chart((lines + pts + txt).properties(height=380), use_container_width=True)
    st.caption(
        f"Pipeline-chosen cutoff: PD < {metrics['policy']['t1']} approves {comp['rules_plus_ml']['approval_rate']:.1%} at a {comp['rules_plus_ml']['default_rate_among_approved']:.2%} default rate. "
        "Rules can only remove applicants, so the fair benchmark is the best rules-only policy (thresholds plus a bureau-score floor) at the same default rate."
    )

# ------------------------------------------------------------------------------------------- funnel
with tab_fun:
    f = metrics.get("funnel")
    if not f:
        st.info("No previous_application.csv was provided, so there is no funnel.")
    else:
        st.metric("Previous applications analysed", f"{f['total_rows_k']:,}K")
        for line in f["findings"]:
            st.write("•", line)
        st.write(f"• {f['unused_share_of_offers_pct']}% of offers made (approved + unused) were never used.")
        st.caption("Application OUTCOME funnel (approved / refused / canceled / unused offer), not a click-by-click UX funnel.")
        a, b = st.columns(2)
        a.image(str(REPORTS / "figures/funnel_outcomes.png"))
        b.image(str(REPORTS / "figures/funnel_refusal_by_segment.png"))
        seg = pd.read_csv(REPORTS / "funnel_summary.csv")
        dim = st.selectbox("Segment by", seg["dimension"].unique())
        st.dataframe(seg[seg["dimension"] == dim].drop(columns="dimension"), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------------------------------- quality
with tab_q:
    d, mc = metrics["discrimination_test"], metrics["models"]["comparison"]
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("ROC-AUC (test)", d["auc"], help=f"95% CI {d['auc_ci95'][0]:.3f}-{d['auc_ci95'][1]:.3f}")
    q2.metric("Gini", d["gini"])
    q3.metric("KS", d["ks"])
    q4.metric(
        "Calibration error (ECE)",
        metrics["calibration"]["ece_calibrated"],
        delta=f"{metrics['calibration']['ece_calibrated'] - metrics['calibration']['ece_uncalibrated']:+.4f} vs raw",
        delta_color="inverse",
    )
    st.dataframe(
        pd.DataFrame(mc).T.rename(columns={"auc_val": "Validation AUC", "auc_test": "Test AUC"}), use_container_width=True
    )
    a, b, c = st.columns(3)
    a.image(str(REPORTS / "figures/roc.png"))
    b.image(str(REPORTS / "figures/calibration.png"))
    c.image(str(REPORTS / "figures/deciles.png"))
    st.image(str(REPORTS / "figures/shap_importance.png"), width=520)
    st.markdown("Full detail in [`reports/MODEL_CARD.md`](reports/MODEL_CARD.md).")

# ------------------------------------------------------------------------------------------ fairness
with tab_fair:
    fa = metrics["fairness"]
    st.markdown(
        "Gender is **not** a model feature; it is held aside for this audit. Age is a feature and is bounded by a hard rule. These are measured gaps, not a claim that the model is fair."
    )
    for name in ("gender", "age_band"):
        g = fa["rules_plus_ml"][name]
        ci = g.get("gap_pts_ci95")
        st.subheader(
            f"{name.replace('_', ' ').title()}: approval gap {g['gap_pts']} pts" + (f" (95% CI {ci[0]} to {ci[1]})" if ci else "")
        )
        st.dataframe(pd.DataFrame(g["groups"]), hide_index=True, use_container_width=True)
        st.caption(
            f"Adverse-impact ratio {g['adverse_impact_ratio']}. Same audit under rules only: gap {fa['current_rules_only'][name]['gap_pts']} pts."
        )
    st.image(str(REPORTS / "figures/fairness.png"))
    st.info(
        f"Proxy test: gender is predicted from the model's own inputs with AUC {fa['proxy_test']['gender_from_features_auc']} (0.5 = no signal). Most informative inputs: {', '.join(fa['proxy_test']['top_predictive_features'])}."
    )

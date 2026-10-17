"""One place that turns the policy comparison in metrics.json into table rows and sentences, so the README,
the model card and the resume line can never tell different stories (or hide that a benchmark missed its target)."""

from __future__ import annotations


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def rng(ci) -> str:
    return f"{100 * ci[0]:.1f}-{100 * ci[1]:.1f}%"


def policy_rows(c: dict) -> list[tuple[str, dict]]:
    tun = c["tuned_rules_only_same_target"]
    tun_label = f"Tuned rules only, same {pct(c['target_default_rate_val'])} target default rate"
    if not tun["target_reached_on_validation"]:
        tun_label = (
            f"Tuned rules only, lowest default rate any rules reached (target {pct(c['target_default_rate_val'])} NOT reached)"
        )
    return [
        ("Current rules only", c["current_rules_only"]),
        (tun_label, tun),
        (
            f"Tuned rules only, same {pct(c['rules_plus_ml']['approval_rate'])} approval rate as rules + ML",
            c["tuned_rules_only_same_approval"],
        ),
        ("Rules + calibrated XGBoost", c["rules_plus_ml"]),
    ]


def sentences(c: dict) -> list[str]:
    ml, sa, tun = c["rules_plus_ml"], c["tuned_rules_only_same_approval"], c["tuned_rules_only_same_target"]
    cut, up, cur = (
        c["default_cut_vs_tuned_rules_same_approval_pts"],
        c["approval_uplift_vs_tuned_rules_pts"],
        c["default_rate_cut_vs_current_pts"],
    )
    out = [
        f"At a matched approval rate (rules + ML {pct(ml['approval_rate'])}, best tuned rules-only {pct(sa['approval_rate'])}; matched within 0.25 pts "
        f"on validation), the best rules-only policy (income floor, repayment cap, loan-to-income cap and a bureau-score "
        f"floor, all tuned on validation) has a {pct(sa['default_rate_among_approved'])} default rate against "
        f"{pct(ml['default_rate_among_approved'])} for rules + ML: **{cut['point']} pts lower** (95% CI {cut['ci95'][0]} to {cut['ci95'][1]})."
    ]
    if tun["target_reached_on_validation"]:
        out.append(
            f"At the same ~{pct(c['target_default_rate_val'])} default rate, rules + ML approves **{up['point']} pts more applicants** than the best "
            f"rules-only policy (95% CI {up['ci95'][0]} to {up['ci95'][1]})."
        )
    else:
        out.append(
            f"No rules-only configuration reached the {pct(c['target_default_rate_val'])} target default rate on validation; the lowest it found was "
            f"{pct(tun['validation_default_rate'])}, at {pct(tun['approval_rate'])} approval on the test split, whereas rules + ML holds "
            f"{pct(ml['default_rate_among_approved'])} at {pct(ml['approval_rate'])}."
        )
    out.append(
        f"Versus the current rules, rules + ML lowers the default rate among approved by {cur['point']} pts at a {pct(ml['approval_rate'])} approval rate."
    )
    return out


def insight_sentence(m: dict) -> str | None:
    i = m.get("rules_insight")
    if not i or i["over_60_default_rate"] is None:
        return None
    return (
        f"**Policy insight:** the 21-60 age rule removes the safest group. Applicants over 60 are {pct(i['over_60_share'])} of applications and default at "
        f"{pct(i['over_60_default_rate'])}, against {pct(i['in_band_default_rate'])} inside the band, so the rule pushes portfolio default up "
        f"(overall {pct(i['overall_default_rate'])}). Worth a business and legal review."
    )


def resume_line(c: dict) -> str:
    ml, sa, tun = c["rules_plus_ml"], c["tuned_rules_only_same_approval"], c["tuned_rules_only_same_target"]
    if tun["target_reached_on_validation"]:
        return (
            f"approval {pct(tun['approval_rate'])} (tuned rules) -> {pct(ml['approval_rate'])} (rules + ML) "
            f"at ~{pct(c['target_default_rate_val'])} default"
        )
    return (
        f"default rate {pct(sa['default_rate_among_approved'])} (tuned rules) -> {pct(ml['default_rate_among_approved'])} (rules + ML) "
        f"at the same {pct(ml['approval_rate'])} approval rate"
    )

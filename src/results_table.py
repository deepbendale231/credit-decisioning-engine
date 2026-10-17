"""Resume numbers and the README results block, straight from metrics.json (nothing typed by hand).

    python -m src.results_table                  # print the block + resume placeholders
    python -m src.results_table --write-readme   # inject the block between the RESULTS markers in README.md

Refuses to write synthetic-data numbers into the README unless --allow-synthetic is given.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .narrative import insight_sentence, pct, policy_rows, resume_line, rng, sentences

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def readme_block(m: dict) -> str:
    meta, c, d, f = m["meta"], m["comparison"], m["discrimination_test"], m["funnel"]
    fm, fr = m["fairness"]["rules_plus_ml"], m["fairness"]["current_rules_only"]
    models = m["models"]["comparison"]
    warn = "" if meta["resume_safe"] else "> **SYNTHETIC DATA: plumbing check only. Do not quote these numbers.**\n\n"

    def gap(a):
        ci = f" (95% CI {a['gap_pts_ci95'][0]} to {a['gap_pts_ci95'][1]})" if "gap_pts_ci95" in a else ""
        return f"{a['gap_pts']} pts{ci}"

    lines = [
        warn + f"_{meta['n_rows']:,} applications, {meta['data_source']} data, model `{meta['model_version']}`. "
        f"Held-out test split of {meta['n_test']:,}. 95% bootstrap CIs in brackets._\n",
        "| Policy (test set) | Approval rate | Default rate among approved |",
        "|---|---|---|",
        *[
            f"| {'**' + label + '**' if i == 3 else label} | {pct(blk['approval_rate'])} ({rng(blk['approval_rate_ci95'])}) | "
            f"{pct(blk['default_rate_among_approved'])} ({rng(blk['default_rate_ci95'])}) |"
            for i, (label, blk) in enumerate(policy_rows(c))
        ],
        "",
        *[line + "\n" for line in sentences(c)],
        *([insight_sentence(m) + "\n"] if insight_sentence(m) else []),
        "| Model quality (test) | |",
        "|---|---|",
        f"| ROC-AUC | **{d['auc']}** (95% CI {d['auc_ci95'][0]:.3f}-{d['auc_ci95'][1]:.3f}) |",
        f"| Gini / KS / PR-AUC | {d['gini']} / {d['ks']} / {d['pr_auc']} |",
        f"| Logistic-regression baseline AUC | {models['logistic_regression']['auc_test']} |",
        f"| Calibration error (ECE) before / after Platt | {m['calibration']['ece_uncalibrated']} / {m['calibration']['ece_calibrated']} |",
        "",
        "| Fairness audit (test) | Approval gap | Adverse-impact ratio | Gap under rules only |",
        "|---|---|---|---|",
        f"| Gender (F / M) | {gap(fm['gender'])} | {fm['gender']['adverse_impact_ratio']} | {fr['gender']['gap_pts']} pts |",
        f"| Age band (21-30 / 31-45 / 46-60) | {gap(fm['age_band'])} | {fm['age_band']['adverse_impact_ratio']} | {fr['age_band']['gap_pts']} pts |",
        "",
        f"Gender is not a feature; it can be predicted from the model inputs with AUC {m['fairness']['proxy_test']['gender_from_features_auc']} (0.5 = no signal). "
        "These are measured gaps, not a claim that the model is fair.\n",
    ]
    if f:
        lines += [
            f"**Outcome funnel:** {f['total_rows_k']:,}K previous applications. "
            + " ".join(f["findings"])
            + f" {f['unused_share_of_offers_pct']}% of offers made (approved + unused) were never used.\n"
        ]
    lines += [
        "![Approval vs default](reports/figures/frontier.png)\n",
        "| | | |\n|---|---|---|\n| ![ROC](reports/figures/roc.png) | ![Calibration](reports/figures/calibration.png) | ![Deciles](reports/figures/deciles.png) |\n",
        "Full detail: [`reports/metrics.json`](reports/metrics.json), [`reports/MODEL_CARD.md`](reports/MODEL_CARD.md).",
    ]
    return "\n".join(lines)


def resume_placeholders(m: dict) -> str:
    c, f = m["comparison"], m["funnel"]
    out = ["Resume placeholders (from reports/metrics.json):"]
    if f:
        out.append(f"  [N]K applications analysed : {f['total_rows_k']:,}K")
        out += [f"  [key finding]              : {line}" for line in f["findings"]]
        out.append(f"  unused share of offers     : {f['unused_share_of_offers_pct']}%")
    out += [
        f"  [0.xx] AUC                 : {m['discrimination_test']['auc']}",
        f"  [X% to Y%] approval        : {resume_line(c)}",
        f"  [X] pts gender gap         : {m['fairness']['rules_plus_ml']['gender']['gap_pts']}",
        f"  [X] pts age-band gap       : {m['fairness']['rules_plus_ml']['age_band']['gap_pts']}",
    ]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", type=Path, default=Path("reports/metrics.json"))
    ap.add_argument("--readme", type=Path, default=Path("README.md"))
    ap.add_argument("--write-readme", action="store_true")
    ap.add_argument("--allow-synthetic", action="store_true")
    args = ap.parse_args(argv)
    m = json.loads(args.metrics.read_text())
    block = readme_block(m)
    if not m["meta"]["resume_safe"]:
        print(
            f"!! data_source={m['meta']['data_source']}: NOT resume-safe. Re-run the pipeline on the real data first.\n",
            file=sys.stderr,
        )
    if args.write_readme:
        if not m["meta"]["resume_safe"] and not args.allow_synthetic:
            print("refusing to write synthetic numbers into the README (use --allow-synthetic to override)", file=sys.stderr)
            return 1
        text = args.readme.read_text()
        if START not in text or END not in text:
            print(f"README is missing {START} / {END} markers", file=sys.stderr)
            return 1
        new = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: f"{START}\n{block}\n{END}", text, flags=re.S)
        args.readme.write_text(new)
        print(f"README results block updated from {args.metrics}")
    else:
        print(block, "\n")
    print(resume_placeholders(m))
    return 0


if __name__ == "__main__":
    sys.exit(main())

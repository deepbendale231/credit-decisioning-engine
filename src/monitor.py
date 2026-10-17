"""Production monitoring from the audit log: score and feature drift (PSI) plus decision mix.

    python -m src.monitor --log logs/decisions.jsonl --model models/model.joblib [--json]

Exit code 1 if any PSI crosses the action threshold (0.25), so it can gate a scheduled job.
Rule of thumb: PSI < 0.10 stable, 0.10-0.25 watch, > 0.25 investigate / consider retraining.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from .audit import AuditLog
from .evaluation import PSI_ACTION, psi_against, psi_status
from .model import load_bundle

MIN_RECORDS = 200  # PSI on fewer decisions is mostly noise


def build_report(records: list[dict], reference: dict, expected_mix: dict | None = None) -> dict:
    scored = [r for r in records if r["pd"] is not None]
    report = {
        "n_decisions": len(records),
        "n_scored": len(scored),
        "decision_mix": {k: round(v / len(records), 4) for k, v in Counter(r["decision"] for r in records).items()}
        if records
        else {},
        "rule_decline_share": round(sum(1 for r in records if r["failed_rules"]) / len(records), 4) if records else None,
        "sufficient_data": len(scored) >= MIN_RECORDS,
        "psi": {},
    }
    if expected_mix:
        report["expected_mix_test"] = expected_mix
    if not report["sufficient_data"]:
        return report
    # Score drift is measured on the scored population (rules-passing); feature drift on ALL logged applicants.
    report["psi"]["pd"] = round(psi_against(reference["pd"], np.array([r["pd"] for r in scored], float)), 4)
    for name, ref in reference["features"].items():
        values = np.array([np.nan if r["features"][name] is None else r["features"][name] for r in records], float)
        report["psi"][name] = round(psi_against(ref, values), 4)
    report["status"] = {k: psi_status(v) for k, v in report["psi"].items()}
    report["alert"] = any(v >= PSI_ACTION for v in report["psi"].values())
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, default=Path("logs/decisions.jsonl"))
    ap.add_argument("--model", type=Path, default=Path("models/model.joblib"))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    records = AuditLog(args.log).read()
    bundle = load_bundle(args.model)
    report = build_report(records, bundle.meta["reference"])
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"decisions: {report['n_decisions']} (scored by model: {report['n_scored']})  mix: {report['decision_mix']}")
        if not report["sufficient_data"]:
            print(f"need >= {MIN_RECORDS} scored decisions for a meaningful PSI; skipping drift.")
        for k, v in report["psi"].items():
            print(f"  PSI {k:<22} {v:>7.4f}  {report['status'][k]}")
        if report.get("alert"):
            print("ALERT: PSI above action threshold; investigate before trusting the policy cutoffs.")
    return 1 if report.get("alert") else 0


if __name__ == "__main__":
    sys.exit(main())

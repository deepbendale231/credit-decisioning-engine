"""Application-outcome funnel from previous_application, using DuckDB SQL.

This is OUTCOME data (approved / refused / canceled / unused), not a click-by-click
UX funnel. It tells you where applications end, not where users dropped inside a screen.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

STATUSES = ["Approved", "Refused", "Canceled", "Unused offer"]
# Whitelisted column names (they are interpolated into SQL).
DIMENSIONS = [
    "NAME_CONTRACT_TYPE",
    "NAME_CLIENT_TYPE",
    "CODE_REJECT_REASON",
    "NAME_PORTFOLIO",
    "WEEKDAY_APPR_PROCESS_START",
    "HOUR_APPR_PROCESS_START",
]

SEGMENT_SQL = """
SELECT '{dim}' AS dimension,
       CAST({dim} AS VARCHAR) AS segment,
       COUNT(*) AS n,
       ROUND(100.0 * AVG(CASE WHEN NAME_CONTRACT_STATUS = 'Approved'     THEN 1 ELSE 0 END), 1) AS approved_pct,
       ROUND(100.0 * AVG(CASE WHEN NAME_CONTRACT_STATUS = 'Refused'      THEN 1 ELSE 0 END), 1) AS refused_pct,
       ROUND(100.0 * AVG(CASE WHEN NAME_CONTRACT_STATUS = 'Canceled'     THEN 1 ELSE 0 END), 1) AS canceled_pct,
       ROUND(100.0 * AVG(CASE WHEN NAME_CONTRACT_STATUS = 'Unused offer' THEN 1 ELSE 0 END), 1) AS unused_pct
FROM prev
GROUP BY 1, 2
ORDER BY n DESC
"""


def _connect(source) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    if isinstance(source, pd.DataFrame):
        con.register("prev", source)
    else:
        path = str(Path(source)).replace("'", "''")
        con.execute(f"CREATE VIEW prev AS SELECT * FROM read_csv_auto('{path}')")
    return con


def run_funnel(source) -> dict:
    """`source` is a previous_application.csv path or a DataFrame. Returns
    {'status': df, 'segments': df, 'summary': dict}."""
    con = _connect(source)
    status = con.execute(
        """SELECT NAME_CONTRACT_STATUS AS status, COUNT(*) AS n,
                  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
           FROM prev GROUP BY 1 ORDER BY n DESC"""
    ).df()
    segments = pd.concat([con.execute(SEGMENT_SQL.format(dim=d)).df() for d in DIMENSIONS], ignore_index=True)
    total = int(status["n"].sum())
    count = dict(zip(status["status"], status["n"]))
    offers = count.get("Approved", 0) + count.get("Unused offer", 0)

    summary = {
        "total_rows": total,
        "total_rows_k": round(total / 1000, 1),
        "status_pct": dict(zip(status["status"], status["pct"].astype(float))),
        "unused_share_of_offers_pct": round(100 * count.get("Unused offer", 0) / offers, 1) if offers else None,
        "findings": _findings(segments, total),
    }
    return {"status": status, "segments": segments, "summary": summary}


def _findings(segments: pd.DataFrame, total: int) -> list[str]:
    """Plain-language facts computed from the query output (nothing typed by hand)."""
    out = []
    for dim, label in [("NAME_CLIENT_TYPE", "client type"), ("NAME_CONTRACT_TYPE", "contract type")]:
        seg = segments[(segments["dimension"] == dim) & (segments["n"] >= 0.01 * total)]
        seg = seg[seg["segment"] != "XNA"]
        if len(seg) < 2:
            continue
        hi, lo = seg.loc[seg["refused_pct"].idxmax()], seg.loc[seg["refused_pct"].idxmin()]
        out.append(
            f"By {label}, refusal rate ranges from {lo['refused_pct']}% ({lo['segment']}, n={int(lo['n']):,}) "
            f"to {hi['refused_pct']}% ({hi['segment']}, n={int(hi['n']):,})."
        )
    return out


def save_funnel_outputs(result: dict, reports_dir: str | Path) -> None:
    from matplotlib.figure import Figure

    from .plotting import BLUE, INK, ORANGE, _save, _style

    reports_dir = Path(reports_dir)
    result["segments"].to_csv(reports_dir / "funnel_summary.csv", index=False)

    status = result["status"]
    fig = Figure(figsize=(6, 3.4))
    ax = fig.subplots()
    _style(ax, "Application outcomes", "% of applications")
    ax.barh(status["status"][::-1], status["pct"][::-1], color=BLUE, height=0.6)
    for i, v in enumerate(status["pct"][::-1]):
        ax.text(v + 0.8, i, f"{v}%", va="center", color=INK, fontsize=9)
    ax.set_xlim(0, status["pct"].max() * 1.15)
    ax.grid(axis="y", visible=False)
    _save(fig, reports_dir / "figures" / "funnel_outcomes.png")

    seg = result["segments"]
    seg = seg[seg["dimension"].isin(["NAME_CLIENT_TYPE", "NAME_CONTRACT_TYPE"]) & (seg["segment"] != "XNA")]
    seg = seg[seg["n"] >= 0.01 * result["summary"]["total_rows"]]
    fig = Figure(figsize=(6, 3.4))
    ax = fig.subplots()
    _style(ax, "Refusal rate by segment", "% refused")
    labels = seg["dimension"].str.replace("NAME_", "").str.replace("_", " ").str.title() + ": " + seg["segment"]
    ax.barh(labels[::-1], seg["refused_pct"][::-1], color=ORANGE, height=0.6)
    for i, v in enumerate(seg["refused_pct"][::-1]):
        ax.text(v + 0.4, i, f"{v}%", va="center", color=INK, fontsize=9)
    ax.set_xlim(0, seg["refused_pct"].max() * 1.15)
    ax.grid(axis="y", visible=False)
    _save(fig, reports_dir / "figures" / "funnel_refusal_by_segment.png")

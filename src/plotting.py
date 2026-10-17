"""Report figures. Light surface, thin marks, recessive grid, at most 3 series, direct labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure  # no pyplot: importing this module must not change the global backend

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3 of the validated reference palette


def _style(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.figure.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)


def roc_plot(curves: dict[str, tuple], path: Path):
    """curves: label -> (fpr, tpr, auc)."""
    fig = Figure(figsize=(5.2, 4.4))
    ax = fig.subplots()
    _style(ax, "Discrimination on the held-out test set", "False positive rate", "True positive rate")
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.2)
    for (label, (fpr, tpr, auc)), color in zip(curves.items(), (BLUE, ORANGE, AQUA)):
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{label} (AUC {auc:.3f})")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")
    _save(fig, path)


def calibration_plot(raw: pd.DataFrame, cal: pd.DataFrame, path: Path, ece_raw: float, ece_cal: float):
    fig = Figure(figsize=(5.2, 4.4))
    ax = fig.subplots()
    _style(ax, "Predicted vs observed default rate", "Mean predicted PD", "Observed default rate")
    top = max(raw["observed"].max(), cal["observed"].max(), raw["mean_pd"].max()) * 1.1
    ax.plot([0, top], [0, top], color=GRID, linewidth=1.2)
    ax.plot(
        raw["mean_pd"],
        raw["observed"],
        "o-",
        color=ORANGE,
        linewidth=1.5,
        markersize=6,
        label=f"Uncalibrated (ECE {ece_raw:.4f})",
    )
    ax.plot(
        cal["mean_pd"],
        cal["observed"],
        "o-",
        color=BLUE,
        linewidth=2,
        markersize=6,
        label=f"Platt-calibrated (ECE {ece_cal:.4f})",
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    _save(fig, path)


def decile_plot(table: pd.DataFrame, base_rate: float, path: Path):
    fig = Figure(figsize=(6.2, 4.0))
    ax = fig.subplots()
    _style(ax, "Default rate by model-risk decile (1 = riskiest)", "Risk decile", "Observed default rate")
    ax.bar(table["decile"], table["default_rate"] * 100, color=BLUE, width=0.62)
    ax.axhline(base_rate * 100, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(10.4, base_rate * 100, f"portfolio {base_rate * 100:.1f}%", color=INK2, fontsize=8.5, va="bottom", ha="right")
    ax.set_xticks(table["decile"])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.grid(axis="x", visible=False)
    _save(fig, path)


def frontier_plot(ml: pd.DataFrame, rules: pd.DataFrame, current: dict, operating: dict, path: Path):
    """Approval vs default trade-off: ML policy curve against the best tuned rules-only envelope."""
    fig = Figure(figsize=(6.6, 4.6))
    ax = fig.subplots()
    _style(
        ax,
        "Approval vs default: rules + ML against tuned rules",
        "Approval rate (share of all applicants)",
        "Default rate among approved",
    )
    ax.plot(ml["approval_rate"] * 100, ml["default_rate"] * 100, color=BLUE, linewidth=2.2)
    ax.plot(rules["approval_rate"] * 100, rules["default_rate"] * 100, color=ORANGE, linewidth=2, marker="o", markersize=4)
    ax.scatter([current["approval_rate"] * 100], [current["default_rate"] * 100], color=INK, s=46, zorder=5)
    ax.annotate(
        "current rules",
        (current["approval_rate"] * 100, current["default_rate"] * 100),
        xytext=(8, 6),
        textcoords="offset points",
        color=INK,
        fontsize=9,
    )
    ax.scatter(
        [operating["approval_rate"] * 100],
        [operating["default_rate"] * 100],
        facecolor=SURFACE,
        edgecolor=BLUE,
        s=90,
        linewidth=2,
        zorder=6,
    )
    ax.annotate(
        "chosen cutoff",
        (operating["approval_rate"] * 100, operating["default_rate"] * 100),
        xytext=(-8, 14),
        textcoords="offset points",
        color=BLUE,
        fontsize=9,
        ha="right",
    )
    ax.text(
        ml["approval_rate"].iloc[len(ml) // 2] * 100 + 2,
        ml["default_rate"].iloc[len(ml) // 2] * 100 - 0.55,
        "rules + ML",
        color=BLUE,
        fontsize=9.5,
        fontweight="bold",
    )
    ax.text(
        rules["approval_rate"].iloc[-1] * 100,
        rules["default_rate"].iloc[-1] * 100 + 0.35,
        "tuned rules only",
        color=ORANGE,
        fontsize=9.5,
        fontweight="bold",
        ha="right",
    )
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1f}%")
    _save(fig, path)


def importance_plot(names: list[str], values: list[float], path: Path):
    order = np.argsort(values)
    fig = Figure(figsize=(6.2, 4.4))
    ax = fig.subplots()
    _style(ax, "What drives predicted risk (mean |SHAP|, test sample)", "Mean |SHAP| (log-odds)")
    ax.barh(np.array(names)[order], np.array(values)[order], color=BLUE, height=0.62)
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def fairness_plot(groups: dict[str, list[dict]], path: Path):
    """groups: attribute -> rows with group / approval_rate / default_rate_among_approved."""
    fig = Figure(figsize=(4.2 * len(groups), 3.8))
    axes = fig.subplots(1, len(groups), squeeze=False)
    for ax, (attr, rows) in zip(axes[0], groups.items()):
        _style(ax, f"Approval rate by {attr}")
        labels = [r["group"] for r in rows]
        vals = [r["approval_rate"] * 100 for r in rows]
        ax.bar(labels, vals, color=BLUE, width=0.55)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.6, f"{v:.1f}%", ha="center", color=INK, fontsize=9)
        ax.set_ylim(0, max(vals) * 1.18)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax.grid(axis="x", visible=False)
    _save(fig, path)

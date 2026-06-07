"""
Report Charts.
Generates all charts for the scientific report as base64-encoded PNG.
Embedded directly in the HTML/PDF — no external files needed.

Charts produced:
  1. Convergence history — match score and R² per round
  2. Parameter effects — coefficient plot with CI bars
  3. Time effects — temporal progression
  4. Entropy/KL trajectory (if composition-series data)
"""

from __future__ import annotations

import base64
import io

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # non-interactive backend

COLORS = {
    "accent": "#00d4aa",
    "teal": "#38bdf8",
    "amber": "#f59e0b",
    "red": "#f87171",
    "muted": "#6b7a9a",
    "grid": "#1e2840",
    "bg": "#111620",
    "text": "#e8edf5",
}


def _fig_to_base64(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64 string for HTML embedding."""
    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"


def _apply_dark_style(ax: plt.Axes, fig: plt.Figure) -> None:
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.tick_params(colors=COLORS["muted"], labelsize=9)
    ax.xaxis.label.set_color(COLORS["muted"])
    ax.yaxis.label.set_color(COLORS["muted"])
    ax.title.set_color(COLORS["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS["grid"])
    ax.grid(color=COLORS["grid"], linewidth=0.5, linestyle="--", alpha=0.6)


def convergence_chart(
    rounds: list[int],
    match_scores: list[float],
    r_squared: list[float],
    threshold: float = 0.75,
) -> str:
    """Convergence history — match score and R² per round."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    _apply_dark_style(ax, fig)

    ax.plot(
        rounds,
        match_scores,
        "o-",
        color=COLORS["accent"],
        linewidth=2,
        markersize=5,
        label="Match Score",
        zorder=3,
    )
    ax.plot(
        rounds,
        r_squared,
        "s--",
        color=COLORS["teal"],
        linewidth=1.5,
        markersize=4,
        label="Best R²",
        zorder=3,
    )

    # Threshold line
    ax.axhline(
        threshold,
        color=COLORS["amber"],
        linewidth=1,
        linestyle=":",
        alpha=0.8,
        label=f"Threshold ({threshold})",
    )

    ax.set_xlabel("Round")
    ax.set_ylabel("Score")
    ax.set_title("Convergence History", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(rounds)
    ax.legend(
        fontsize=8,
        facecolor=COLORS["bg"],
        labelcolor=COLORS["muted"],
        edgecolor=COLORS["grid"],
    )

    return _fig_to_base64(fig)


def parameter_effects_chart(
    variables: list[str],
    coefficients: list[float],
    ci_lower: list[float],
    ci_upper: list[float],
    p_values: list[float],
    output_name: str,
) -> str:
    """Horizontal coefficient plot with confidence intervals."""
    n = len(variables)
    if n == 0:
        return ""

    fig, ax = plt.subplots(figsize=(7, max(3, n * 0.5)))
    _apply_dark_style(ax, fig)

    y_pos = np.arange(n)
    colors = [COLORS["accent"] if c > 0 else COLORS["red"] for c in coefficients]
    alphas = [1.0 if p < 0.05 else 0.45 for p in p_values]

    for i, (coef, lo, hi, col, alpha) in enumerate(
        zip(coefficients, ci_lower, ci_upper, colors, alphas)
    ):
        ax.barh(i, coef, color=col, alpha=alpha, height=0.5, zorder=3)
        ax.plot(
            [lo, hi],
            [i, i],
            color=col,
            alpha=alpha,
            linewidth=2,
            solid_capstyle="round",
        )
        ax.plot([lo, hi], [i, i], "|", color=col, alpha=alpha, markersize=6)

    ax.axvline(0, color=COLORS["muted"], linewidth=1, alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(variables, fontsize=9)
    ax.set_xlabel("Coefficient (95% CI)")
    ax.set_title(f"Parameter Effects — {output_name}", fontsize=11, fontweight="bold")

    # Significance legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=COLORS["accent"], label="Positive (p<0.05)"),
        Patch(facecolor=COLORS["red"], label="Negative (p<0.05)"),
        Patch(facecolor=COLORS["muted"], alpha=0.4, label="Not significant"),
    ]
    ax.legend(
        handles=legend_elements,
        fontsize=8,
        facecolor=COLORS["bg"],
        labelcolor=COLORS["muted"],
        edgecolor=COLORS["grid"],
        loc="lower right",
    )

    fig.tight_layout()
    return _fig_to_base64(fig)


def time_effects_chart(
    time_values: list[float],
    coefficients: list[float],
    ci_lower: list[float],
    ci_upper: list[float],
    p_values: list[float],
    output_name: str,
) -> str:
    """Time indicator coefficient plot — shows temporal progression."""
    if not time_values:
        return ""

    fig, ax = plt.subplots(figsize=(7, 3.5))
    _apply_dark_style(ax, fig)

    alphas = [1.0 if p < 0.05 else 0.4 for p in p_values]

    ax.fill_between(time_values, ci_lower, ci_upper, color=COLORS["teal"], alpha=0.15)
    ax.plot(
        time_values,
        coefficients,
        "o-",
        color=COLORS["teal"],
        linewidth=2,
        markersize=5,
        zorder=3,
    )

    for t, c, a in zip(time_values, coefficients, alphas):
        ax.plot(
            t,
            c,
            "o",
            color=COLORS["teal"] if a > 0.5 else COLORS["muted"],
            markersize=6,
            zorder=4,
        )

    ax.axhline(0, color=COLORS["muted"], linewidth=1, alpha=0.4)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Time Effect (coefficient)")
    ax.set_title(f"Time Effects — {output_name}", fontsize=11, fontweight="bold")

    fig.tight_layout()
    return _fig_to_base64(fig)


def entropy_trajectory_chart(
    time_values: list[float],
    entropy_values: list[float],
    kl_values: list[float],
    inflection: float | None = None,
) -> str:
    """Shannon entropy and KL divergence trajectories."""
    if not entropy_values:
        return ""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax in (ax1, ax2):
        _apply_dark_style(ax, fig)

    ax1.plot(
        time_values,
        entropy_values,
        "o-",
        color=COLORS["accent"],
        linewidth=2,
        markersize=5,
    )
    if inflection:
        ax1.axvline(
            inflection,
            color=COLORS["amber"],
            linewidth=1,
            linestyle=":",
            alpha=0.8,
            label=f"Inflection ({inflection:.0f} min)",
        )
        ax1.legend(
            fontsize=8,
            facecolor=COLORS["bg"],
            labelcolor=COLORS["muted"],
            edgecolor=COLORS["grid"],
        )
    ax1.set_xlabel("Time (min)")
    ax1.set_ylabel("Shannon Entropy")
    ax1.set_title("Intermediate Distribution Spread", fontsize=10, fontweight="bold")

    ax2.plot(
        time_values, kl_values, "s-", color=COLORS["teal"], linewidth=2, markersize=5
    )
    if inflection:
        ax2.axvline(
            inflection, color=COLORS["amber"], linewidth=1, linestyle=":", alpha=0.8
        )
    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel("KL Divergence")
    ax2.set_title("Pathway Shift from Initial State", fontsize=10, fontweight="bold")

    fig.tight_layout()
    return _fig_to_base64(fig)

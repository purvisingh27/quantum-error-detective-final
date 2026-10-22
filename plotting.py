"""Matplotlib figures for Quantum Error Detective.

Colours follow a validated categorical palette (slots 1-3, which are safe for
colour-vision deficiency even when all pairs appear together). The aqua slot is
low-contrast on a light surface, so every chart also has a legend plus direct
labels, and all numbers are available in the CSV files.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CIRCUIT_COLORS = {"Bell": BLUE, "GHZ3": ORANGE, "Grover2": AQUA}
NOISE_COLORS = {"gate": BLUE, "readout": ORANGE, "combined": AQUA}
NOISE_LABELS = {"gate": "Gate noise", "readout": "Readout noise", "combined": "Combined"}
MIT_COLORS = {"raw": BLUE, "mitigated": ORANGE}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9, length=3)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig(*args, **kwargs):
    fig, axes = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def _legend(ax_or_fig, handles, **kw):
    leg = ax_or_fig.legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK_2, **kw)
    return leg


# ------------------------------------------------------------------ figure 1
def plot_noise_vs_success(sweep, path):
    """Success probability vs noise level, one panel per noise type."""
    circuits = list(sweep["circuit"].unique())
    types = ["gate", "readout", "combined"]
    fig, axes = _fig(1, 3, figsize=(13, 4.4), sharey=True)

    for ax, ntype in zip(axes, types):
        _style(ax)
        sub = sweep[sweep["noise_type"] == ntype]
        for name in circuits:
            d = sub[sub["circuit"] == name].sort_values("noise_level")
            color = CIRCUIT_COLORS.get(name, INK_2)
            ax.fill_between(d["noise_level"], d["ci_low"], d["ci_high"], color=color, alpha=0.15, linewidth=0)
            ax.plot(d["noise_level"], d["success_prob"], color=color, linewidth=2, marker="o",
                    markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
        ax.set_title(NOISE_LABELS[ntype], color=INK, fontsize=11, loc="left", pad=8)
        ax.set_xlabel("Noise level", color=INK_2, fontsize=9)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Success probability", color=INK_2, fontsize=9)
    handles = [plt.Line2D([0], [0], color=CIRCUIT_COLORS.get(c, INK_2), linewidth=2, marker="o",
                          markersize=6.5, label=c) for c in circuits]
    fig.legend(handles=handles, loc="lower center", ncol=len(circuits), frameon=False,
               fontsize=9, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("How noise lowers the chance of a correct answer", color=INK, fontsize=13,
                 x=0.012, ha="left", y=1.0)
    fig.text(0.012, 0.915, "Shaded band = 95% confidence interval. Noise level maps to gate / readout error rates "
                           "(see run_metadata.json).", color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------ figure 2
def plot_mitigation_comparison(mit, path):
    """Raw vs mitigated success probability, one panel per noise scenario."""
    scenarios = list(mit["scenario"].unique())
    circuits = list(mit["circuit"].unique())
    fig, axes = _fig(1, len(scenarios), figsize=(5.2 * len(scenarios) + 1, 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    width = 0.36
    x = np.arange(len(circuits))

    for ax, scenario in zip(axes, scenarios):
        _style(ax)
        sub = mit[mit["scenario"] == scenario].set_index("circuit").loc[circuits]
        for i, (key, label) in enumerate([("raw", "Raw (noisy)"), ("mitigated", "Mitigated")]):
            vals = sub[f"{key}_success"].to_numpy()
            lo = sub[f"{'raw' if key == 'raw' else 'mit'}_ci_low"].to_numpy()
            hi = sub[f"{'raw' if key == 'raw' else 'mit'}_ci_high"].to_numpy()
            pos = x + (i - 0.5) * (width + 0.02)
            ax.bar(pos, vals, width, color=MIT_COLORS[key], edgecolor=SURFACE, linewidth=1.5, label=label)
            ax.errorbar(pos, vals, yerr=[vals - lo, hi - vals], fmt="none", ecolor=INK_2,
                        elinewidth=1, capsize=3)
            for p, v in zip(pos, vals):
                ax.text(p, min(v + 0.045, 1.07), f"{v:.2f}", ha="center", va="bottom",
                        fontsize=8.5, color=INK_2)
        ideal = sub["ideal_success"].to_numpy()
        ax.axhline(ideal.max(), color=MUTED, linestyle="--", linewidth=1)
        ax.text(len(circuits) - 0.5, ideal.max() + 0.01, "ideal", color=MUTED, fontsize=8.5,
                ha="right", va="bottom")
        title = {"readout": "Readout noise only", "combined": "Gate + readout noise"}.get(scenario, scenario)
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(circuits)
        ax.set_ylim(0, 1.18)
    axes[0].set_ylabel("Success probability", color=INK_2, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=MIT_COLORS["raw"], label="Raw (noisy)"),
               plt.Rectangle((0, 0), 1, 1, color=MIT_COLORS["mitigated"], label="Mitigated")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9,
               labelcolor=INK_2, bbox_to_anchor=(0.5, -0.02))
    level = mit["noise_level"].iloc[0]
    fig.suptitle("Does readout mitigation help?", color=INK, fontsize=13, x=0.012, ha="left", y=1.0)
    fig.text(0.012, 0.915, f"Noise level {level:g}. Error bars = 95% confidence interval. "
                           "Readout mitigation cannot undo gate errors.", color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------ figure 3
def plot_noise_components(sweep, level, path):
    """Gate vs readout vs combined noise at one fixed noise level."""
    sub = sweep[np.isclose(sweep["noise_level"], level)]
    circuits = list(sub["circuit"].unique())
    types = ["gate", "readout", "combined"]
    fig, ax = _fig(figsize=(8.2, 4.8))
    _style(ax)
    width = 0.25
    x = np.arange(len(circuits))

    for i, ntype in enumerate(types):
        d = sub[sub["noise_type"] == ntype].set_index("circuit").loc[circuits]
        vals = d["success_prob"].to_numpy()
        lo, hi = d["ci_low"].to_numpy(), d["ci_high"].to_numpy()
        pos = x + (i - 1) * (width + 0.02)
        ax.bar(pos, vals, width, color=NOISE_COLORS[ntype], edgecolor=SURFACE, linewidth=1.5,
               label=NOISE_LABELS[ntype])
        ax.errorbar(pos, vals, yerr=[vals - lo, hi - vals], fmt="none", ecolor=INK_2,
                    elinewidth=1, capsize=3)
        for p, v in zip(pos, vals):
            ax.text(p, min(v + 0.045, 1.07), f"{v:.2f}", ha="center", va="bottom",
                    fontsize=8.5, color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels(circuits)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Success probability", color=INK_2, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=NOISE_COLORS[t], label=NOISE_LABELS[t]) for t in types]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9,
               labelcolor=INK_2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Which kind of noise hurts most?", color=INK, fontsize=13, x=0.015, ha="left", y=0.995)
    fig.text(0.015, 0.905, f"Noise level {level:g}. Error bars = 95% confidence interval.",
             color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)

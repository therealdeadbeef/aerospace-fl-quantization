#!/usr/bin/env python3
"""
plot_results.py
Generates publication-ready PDF figures for the paper
"""

import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import ttest_rel

matplotlib.use("Agg")

# -- Style -------------------------------------------------------------------
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         12,
    "axes.labelsize":    14,
    "axes.titlesize":    14,
    "legend.fontsize":   11,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "lines.linewidth":   2,
    "lines.markersize":  8,
    "figure.autolayout": True,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

OUT_DIR     = "figures"
RESULT_FILE = "results/all_results.json"
os.makedirs(OUT_DIR, exist_ok=True)

# -- Load data ----------------------------------------------------------------
print(f"Loading results from '{RESULT_FILE}' ...")
with open(RESULT_FILE, "r") as f:
    data = json.load(f)

CONFIGS = ["FP32", "INT8", "INT4", "INT2"]
STYLES  = {
    "FP32": {"color": "#1f77b4", "marker": "o", "label": "FP32"},
    "INT8": {"color": "#ff7f0e", "marker": "s", "label": "INT8"},
    "INT4": {"color": "#2ca02c", "marker": "^", "label": "INT4"},
    "INT2": {"color": "#d62728", "marker": "D", "label": "INT2"},
}
ROUNDS = np.arange(1, 21)


# -- Helpers ------------------------------------------------------------------
def sorted_seeds(subset, config):
    """Returns seed keys sorted numerically. REQUIRED for paired t-tests."""
    return sorted(data[subset][config].keys(), key=lambda x: int(x))


def get_stats(subset, config, metric):
    """
    Returns (mean, std) over seeds.
    Seeds are sorted numerically to ensure reproducibility.
    """
    seeds  = sorted_seeds(subset, config)
    values = [data[subset][config][s][metric] for s in seeds]

    if isinstance(values[0], (int, float)):
        arr = np.array(values, dtype=float)
        return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

    matrix = np.array(values, dtype=float)  # (N_seeds, N_rounds)
    return matrix.mean(axis=0), matrix.std(axis=0, ddof=1)


def get_raw_last_round(subset, config, metric):
    """
    Returns last-round scalar per seed in SORTED seed order.
    Sorting is mandatory: ttest_rel is a PAIRED test.
    Seed k of FP32 must align with seed k of INT4.
    Without this sort, JSON dict order may differ between configs,
    pairing wrong seeds and producing incorrect p-values.
    """
    seeds = sorted_seeds(subset, config)
    return [data[subset][config][s][metric][-1] for s in seeds]


def paired_pvalue(subset, cfg_a, cfg_b, metric):
    """Two-tailed paired t-test between cfg_a and cfg_b at last round."""
    a = get_raw_last_round(subset, cfg_a, metric)
    b = get_raw_last_round(subset, cfg_b, metric)
    _, p = ttest_rel(a, b)
    return float(p)


print("Writing figures to 'figures/' ...")


# ===== 1 -- Convergence curves (MAE and Score) ===============================
for subset in ["FD001", "FD002"]:
    for metric, ylabel, title_label in [
        ("mae",   "MAE (cycles)",                     "MAE Convergence"),
        ("score", "NASA Score $S$ (lower is better)", "NASA Score Convergence"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for cfg in CONFIGS:
            mean_val, std_val = get_stats(subset, cfg, metric)
            ax.plot(ROUNDS, mean_val,
                    color=STYLES[cfg]["color"], marker=STYLES[cfg]["marker"],
                    label=STYLES[cfg]["label"], markevery=5)
            ax.fill_between(ROUNDS, mean_val - std_val, mean_val + std_val,
                            color=STYLES[cfg]["color"], alpha=0.15)
        ax.set_xlabel("FL Round")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_label} C-MAPSS {subset}")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_xlim(1, 20)
        if metric == "score":
            ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
            # Clip y so INT2 early explosion does not dominate the scale
            stable = []
            for cfg in ["FP32", "INT8", "INT4"]:
                m, _ = get_stats(subset, cfg, "score")
                stable.extend(m.tolist())
            ax.set_ylim(top=np.percentile(stable, 95) * 2.5)
        fname = f"{OUT_DIR}/fig_{metric}_convergence_{subset}.pdf"
        fig.savefig(fname, format="pdf", dpi=300)
        plt.close(fig)
        print(f"  Saved {fname}")


# ===== 2 -- Gradient-distortion proxy ========================================
fig, ax = plt.subplots(figsize=(6, 4.5))
for cfg in ["INT8", "INT4", "INT2"]:   # FP32 omitted (L_priv = 0 by definition)
    mean_val, std_val = get_stats("FD001", cfg, "leakage")
    mean_val = np.clip(mean_val, 1e-15, None)
    ax.plot(ROUNDS, mean_val,
            color=STYLES[cfg]["color"], marker=STYLES[cfg]["marker"],
            label=STYLES[cfg]["label"], markevery=5, linestyle="-.")
    ax.fill_between(ROUNDS, mean_val - std_val, mean_val + std_val,
                    color=STYLES[cfg]["color"], alpha=0.15)
ax.set_yscale("log")
ax.set_xlabel("FL Round")
ax.set_ylabel(r"$\mathcal{L}_{\mathrm{priv}}$ (log scale)")
ax.set_title("Gradient-Distortion Proxy FD001")
ax.grid(True, which="both", linestyle="--", alpha=0.6)
ax.legend()
ax.set_xticks([1, 5, 10, 15, 20])
ax.set_xlim(1, 20)
fname = f"{OUT_DIR}/fig_leakage_FD001.pdf"
fig.savefig(fname, format="pdf", dpi=300)
plt.close(fig)
print(f"  Saved {fname}")


# ===== 3 -- Pareto accuracy-communication trade-off ==========================
for subset in ["FD001", "FD002"]:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fp32_comm = fp32_score = int4_comm = int4_score = 0.0

    for cfg in CONFIGS:
        mean_score, std_score = get_stats(subset, cfg, "score")
        final_mean = float(mean_score[-1])
        final_std  = float(std_score[-1])
        comm_cost, _ = get_stats(subset, cfg, "comm_kib")

        if cfg == "FP32":
            fp32_comm, fp32_score = comm_cost, final_mean
        elif cfg == "INT4":
            int4_comm, int4_score = comm_cost, final_mean

        ax.errorbar(comm_cost, final_mean, yerr=final_std,
                    fmt=STYLES[cfg]["marker"],
                    color=STYLES[cfg]["color"],
                    label=cfg, markersize=10, capsize=5, elinewidth=2)

        offsets = {"INT4": (1.0, 25000), "INT8": (1.0, -25000),
                   "INT2": (1.0, 15000), "FP32": (1.0, -15000)}
        x_off, y_off = offsets[cfg]
        # monospace ensures "INT2" renders with capital I (fixes iNT2 artifact)
        ax.text(comm_cost + x_off, final_mean + y_off, cfg,
                fontsize=11, color=STYLES[cfg]["color"],
                fontfamily="monospace", fontweight="bold",
                ha="left", va="center")

    # Arrow FP32 -> INT4
    ax.annotate("", xy=(int4_comm + 1, int4_score),
                xytext=(fp32_comm - 1.5, fp32_score),
                arrowprops=dict(arrowstyle="->", color="gray", linestyle="dashed"))

    # p-value using SORTED paired t-test -- now matches Table III exactly
    p_val = paired_pvalue(subset, "FP32", "INT4", "score")
    mid_x = (fp32_comm + int4_comm) / 2
    mid_y = (fp32_score + int4_score) / 2
    ax.text(mid_x + 1.0, mid_y + 0.05 * fp32_score,
            f"$p={p_val:.3f}$", fontsize=10, color="0.35", fontweight="bold")

    ax.set_xlabel("Communication Cost per Round (KiB)")
    ax.set_ylabel("NASA Score $S$ (lower is better)")
    ax.set_title(f"Pareto Trade-off {subset}")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.set_xlim(0, max(45, fp32_comm + 5))

    fname = f"{OUT_DIR}/fig_pareto_{subset}.pdf"
    fig.savefig(fname, format="pdf", dpi=300)
    plt.close(fig)
    print(f"  Saved {fname}")


# ===== 4 -- Bar chart: MAE by subset and quantization level ==================
fig, ax = plt.subplots(figsize=(7, 4.5))
x     = np.arange(len(CONFIGS))
width = 0.35

for i, subset in enumerate(["FD001", "FD002"]):
    offset = -width / 2 if i == 0 else width / 2
    hatch  = "" if i == 0 else "//"

    means = [float(get_stats(subset, cfg, "mae")[0][-1]) for cfg in CONFIGS]
    stds  = [float(get_stats(subset, cfg, "mae")[1][-1]) for cfg in CONFIGS]

    ax.bar(x + offset, means, width, yerr=stds, capsize=5,
           color=[STYLES[c]["color"] for c in CONFIGS],
           alpha=0.8, hatch=hatch)

    for j, cfg in enumerate(CONFIGS):
        if cfg == "FP32":
            continue
        p_val = paired_pvalue(subset, "FP32", cfg, "mae")
        if p_val < 0.05:
            ax.text(x[j] + offset, means[j] + stds[j] + 0.8,
                    "*", ha="center", va="bottom", fontsize=16, fontweight="bold")

ax.set_ylabel("MAE (cycles)")
ax.set_title("MAE by Subset and Quantization Level")
ax.set_xticks(x)
ax.set_xticklabels(CONFIGS)
ax.grid(axis="y", linestyle="--", alpha=0.6)

fd001_patch = mpatches.Patch(facecolor="white", edgecolor="black",
                              label="FD001 (1 Op. Cond.)")
fd002_patch = mpatches.Patch(facecolor="white", edgecolor="black",
                              hatch="//", label="FD002 (6 Op. Cond.)")
ax.legend(handles=[fd001_patch, fd002_patch], loc="upper left")

ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin, ymax + 2)

fname = f"{OUT_DIR}/fig_bar_mae_subsets.pdf"
fig.savefig(fname, format="pdf", dpi=300)
plt.close(fig)
print(f"  Saved {fname}")


print()
print("Done. Figures saved in 'figures/'")

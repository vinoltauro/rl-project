"""
Forgetting Curves — All Conditions
=====================================
Plots Pong reward, dead neurons, and CKA drift across all sequential and
interleaved conditions on the same axes for direct comparison.
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "results/plots"

# Each run specifies its directory, log file, and which columns to read
RUNS = {
    "No freeze (sequential)": {
        "dir":       "results/logs",
        "file":      "dqn_sequential_sequential_full_seed42_scalemedium_forgetting.csv",
        "step_col":  "breakout_step",
        "pong_col":  "pong_reward",
        "dead_col":  "dead_neurons",
        "cka_col":   "cka_drift",
        "color":     "#F44336",
        "style":     "-",
    },
    "Freeze conv + fc_repr": {
        "dir":       "results/logs",
        "file":      "dqn_fixed_sequential_freeze_all_seed42_scalemedium_forgetting.csv",
        "step_col":  "breakout_step",
        "pong_col":  "pong_reward",
        "dead_col":  "dead_neurons",
        "cka_col":   "cka_drift",
        "color":     "#4CAF50",
        "style":     "-",
    },
    "Freeze conv only": {
        "dir":       "results/logs",
        "file":      "dqn_fixed_sequential_freeze_conv_fixed_seed42_scalemedium_forgetting.csv",
        "step_col":  "breakout_step",
        "pong_col":  "pong_reward",
        "dead_col":  "dead_neurons",
        "cka_col":   "cka_drift",
        "color":     "#2196F3",
        "style":     "--",
    },
    "Sequential + dead reinit": {
        "dir":       "results/sequential_reinit/logs",
        "file":      "dqn_sequential_reinit_seed42_scalemedium_forgetting.csv",
        "step_col":  "breakout_step",
        "pong_col":  "pong_reward",
        "dead_col":  "dead_neurons",
        "cka_col":   "cka_drift",
        "color":     "#9C27B0",
        "style":     "-.",
    },
    "Interleaved v3 (episode-level)": {
        "dir":       "results/interleaved_v3/logs",
        "file":      "dqn_interleaved_v3_seed42_scalemedium_metrics.csv",
        "step_col":  "total_step",
        "pong_col":  "pong_reward_mean10",
        "dead_col":  "dead_neurons",
        "cka_col":   "cka_drift_from_pong",
        "color":     "#FF9800",
        "style":     "-.",
    },
    "Interleaved v2 (step-level)": {
        "dir":       "results/logs",
        "file":      "dqn_interleaved_v2_seed42_scalemedium_metrics.csv",
        "step_col":  "total_step",
        "pong_col":  "pong_reward_mean10",
        "dead_col":  "dead_neurons",
        "cka_col":   "cka_drift_from_pong",
        "color":     "#FF5722",
        "style":     ":",
    },
}


def load_run(path, step_col, pong_col, dead_col, cka_col):
    steps, pong_r, dead, cka = [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(int(float(row[step_col])))
            pong_r.append(float(row[pong_col]))
            dead.append(float(row[dead_col]))
            cka.append(float(row[cka_col]))
    # deduplicate on step
    seen, out = set(), []
    for s, p, d, c in zip(steps, pong_r, dead, cka):
        if s not in seen:
            seen.add(s)
            out.append((s, p, d, c))
    s, p, d, c = zip(*out)
    return np.array(s) / 1e6, list(p), list(d), list(c)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    titles  = [
        "Pong Reward\n(catastrophic forgetting)",
        "Dead Neuron Fraction\n(capacity loss)",
        "CKA Drift from Original Pong\n(representational shift)",
    ]
    ylabels = ["Mean Pong Reward", "Dead Neuron Fraction", "CKA"]

    for label, cfg in RUNS.items():
        path = os.path.join(cfg["dir"], cfg["file"])
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue

        steps, pong_r, dead, cka = load_run(
            path, cfg["step_col"], cfg["pong_col"],
            cfg["dead_col"], cfg["cka_col"]
        )

        for ax, values in zip(axes, [pong_r, dead, cka]):
            ax.plot(steps, values, label=label,
                    color=cfg["color"], linestyle=cfg["style"],
                    linewidth=2, marker="o", markersize=4)

    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Training Steps (millions)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

    # Reference lines on Pong reward panel
    axes[0].axhline(-21, color="gray", linestyle=":", alpha=0.4, linewidth=1)
    axes[0].axhline(0,   color="gray", linestyle=":", alpha=0.25, linewidth=1)

    fig.suptitle(
        "Forgetting and Capacity Loss Across All Conditions",
        fontsize=12, y=1.01
    )

    out = os.path.join(OUTPUT_DIR, "forgetting_curves.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")

    # Summary table
    print("\nSummary (step 0 → final):")
    for label, cfg in RUNS.items():
        path = os.path.join(cfg["dir"], cfg["file"])
        if not os.path.exists(path):
            continue
        steps, pong_r, dead, cka = load_run(
            path, cfg["step_col"], cfg["pong_col"],
            cfg["dead_col"], cfg["cka_col"]
        )
        print(f"  {label}")
        print(f"    Pong:  {pong_r[0]:.1f} → {pong_r[-1]:.1f}")
        print(f"    Dead:  {dead[0]:.3f} → {dead[-1]:.3f}")
        print(f"    CKA:   {cka[0]:.3f} → {cka[-1]:.3f}")


if __name__ == "__main__":
    main()

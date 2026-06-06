"""
Sequential Training Forgetting Curves
=======================================
Plots three forgetting metrics for both sequential conditions:
  - Pong reward over Breakout training steps
  - Dead neuron fraction over Breakout training steps
  - CKA drift from original Pong representations

Two conditions:
  - sequential_full  (--freeze none): all layers adapt to Breakout
  - freeze_conv      (--freeze conv):  conv layers frozen

Produces: results/plots/forgetting_curves.png
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_DIR    = "results/logs"
OUTPUT_DIR = "results/plots"

RUNS = {
    "No freeze (full sequential)": {
        "file":  "dqn_sequential_sequential_full_seed42_scalemedium_forgetting.csv",
        "color": "#F44336",
        "style": "-",
    },
    "Freeze conv + fc\\_repr (all)": {
        "file":  "dqn_fixed_sequential_freeze_all_seed42_scalemedium_forgetting.csv",
        "color": "#4CAF50",
        "style": "-",
    },
    "Freeze conv only": {
        "file":  "dqn_fixed_sequential_freeze_conv_fixed_seed42_scalemedium_forgetting.csv",
        "color": "#2196F3",
        "style": "--",
    },
}


def load_metrics(path):
    steps, pong_r, dead, cka = [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["breakout_step"]))
            pong_r.append(float(row["pong_reward"]))
            dead.append(float(row["dead_neurons"]))
            cka.append(float(row["cka_drift"]))
    # deduplicate final step if logged twice
    seen = set()
    out = []
    for s, p, d, c in zip(steps, pong_r, dead, cka):
        if s not in seen:
            seen.add(s)
            out.append((s, p, d, c))
    return zip(*out)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    titles = [
        "Pong Reward During Breakout Training\n(measures catastrophic forgetting)",
        "Dead Neuron Fraction During Breakout Training\n(measures capacity loss)",
        "CKA Drift from Original Pong Representations\n(measures representational shift)",
    ]
    ylabels = ["Mean Pong Reward", "Dead Neuron Fraction", "CKA (vs original Pong)"]

    for label, cfg in RUNS.items():
        path = os.path.join(LOG_DIR, cfg["file"])
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue

        steps, pong_r, dead, cka = load_metrics(path)
        steps  = np.array(list(steps))  / 1e6
        pong_r = list(pong_r)
        dead   = list(dead)
        cka    = list(cka)

        for ax, values in zip(axes, [pong_r, dead, cka]):
            ax.plot(steps, values, label=label,
                    color=cfg["color"], linestyle=cfg["style"],
                    linewidth=2, marker="o", markersize=5)

    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Breakout Training Steps (millions)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 2)

    # Add Pong chance level reference
    axes[0].axhline(-21, color="gray", linestyle=":", alpha=0.5,
                    linewidth=1, label="_nolegend_")
    axes[0].axhline(0, color="gray", linestyle=":", alpha=0.3,
                    linewidth=1, label="_nolegend_")

    fig.suptitle(
        "Sequential Training: Catastrophic Forgetting of Pong During Breakout Training",
        fontsize=12, y=1.02
    )

    out = os.path.join(OUTPUT_DIR, "forgetting_curves.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")

    # Print summary table
    print("\nSummary:")
    for label, cfg in RUNS.items():
        path = os.path.join(LOG_DIR, cfg["file"])
        if not os.path.exists(path):
            continue
        steps, pong_r, dead, cka = load_metrics(path)
        steps, pong_r, dead, cka = list(steps), list(pong_r), list(dead), list(cka)
        print(f"\n  {label}")
        print(f"  Pong reward: {pong_r[0]:.1f} → {pong_r[-1]:.1f}")
        print(f"  Dead neurons: {dead[0]:.3f} → {dead[-1]:.3f}")
        print(f"  CKA drift: {cka[0]:.3f} → {cka[-1]:.3f}")


if __name__ == "__main__":
    main()

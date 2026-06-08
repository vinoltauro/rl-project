"""
Interleaved Training Curves
=============================
Plots Pong and Breakout reward over training for the interleaved v2
(step-level alternation) experiment, compared against baselines.
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_DIR    = "results/logs"
OUTPUT_DIR = "results/plots"


def load_interleaved(path):
    total, pong_r, breakout_r, dead = [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            total.append(int(float(row["total_step"])))
            pong_r.append(float(row["pong_reward_mean10"]))
            breakout_r.append(float(row["breakout_reward_mean10"]))
            dead.append(float(row["dead_neurons"]))
    seen, out = set(), []
    for s, p, b, d in zip(total, pong_r, breakout_r, dead):
        if s not in seen:
            seen.add(s)
            out.append((s, p, b, d))
    s, p, b, d = zip(*out)
    return np.array(s)/1e6, list(p), list(b), list(d)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    path = os.path.join(LOG_DIR, "dqn_interleaved_v2_seed42_scalemedium_metrics.csv")
    if not os.path.exists(path):
        print(f"[skip] {path} not found")
        return

    steps, pong_r, breakout_r, dead = load_interleaved(path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Pong reward
    axes[0].plot(steps, pong_r, color="#2196F3", linewidth=2,
                 marker="o", markersize=4, label="Interleaved Pong")
    axes[0].axhline(-21, color="gray", linestyle=":", alpha=0.4, linewidth=1,
                    label="Worst possible (-21)")
    axes[0].axhline(7.5, color="#4CAF50", linestyle="--", alpha=0.5, linewidth=1,
                    label="Standalone Pong baseline (~7.5)")
    axes[0].set_title("Pong Reward During Interleaved Training", fontsize=10)
    axes[0].set_xlabel("Total Training Steps (millions)", fontsize=9)
    axes[0].set_ylabel("Mean Pong Reward (last 10 eps)", fontsize=9)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Breakout reward
    axes[1].plot(steps, breakout_r, color="#F44336", linewidth=2,
                 marker="o", markersize=4, label="Interleaved Breakout")
    axes[1].axhline(3.86, color="#4CAF50", linestyle="--", alpha=0.5, linewidth=1,
                    label="Standalone Breakout baseline (3.86)")
    axes[1].set_title("Breakout Reward During Interleaved Training", fontsize=10)
    axes[1].set_xlabel("Total Training Steps (millions)", fontsize=9)
    axes[1].set_ylabel("Mean Breakout Reward (last 10 eps)", fontsize=9)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Dead neurons
    axes[2].plot(steps, dead, color="#9C27B0", linewidth=2,
                 marker="o", markersize=4, label="Dead neurons (interleaved)")
    axes[2].set_title("Dead Neuron Fraction During Interleaved Training", fontsize=10)
    axes[2].set_xlabel("Total Training Steps (millions)", fontsize=9)
    axes[2].set_ylabel("Dead Neuron Fraction", fontsize=9)
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 1)

    fig.suptitle(
        "Interleaved Training (Step-Level Alternation, 1M steps per game)\n"
        "Dashed lines show standalone baselines for comparison",
        fontsize=11, y=1.02
    )

    out = os.path.join(OUTPUT_DIR, "interleaved_curves.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")

    print(f"\nFinal values:")
    print(f"  Pong reward:    {pong_r[-1]:.2f}  (baseline: ~7.5)")
    print(f"  Breakout reward: {breakout_r[-1]:.2f}  (baseline: 3.86)")
    print(f"  Dead neurons:   {dead[-1]:.3f}")


if __name__ == "__main__":
    main()

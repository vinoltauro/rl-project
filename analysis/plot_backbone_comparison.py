"""
Backbone Transfer Learning Comparison
======================================
Plots learning curves for three Breakout training conditions:
  1. Scratch          — standard DQN trained from random init
  2. Frozen conv      — Pong conv layers frozen, fc layers trained from scratch
  3. Full fine-tune   — all layers initialised from Pong, everything trains

Produces two figures:
  - backbone_comparison_reward.png  : reward over training steps
  - backbone_comparison_qvalue.png  : mean Q-value over training steps
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
    "Scratch (DQN/Breakout)": {
        "file":  "dqn_breakout_seed42_scalemedium_lr0001_buf100k.csv",
        "color": "#9E9E9E",
        "style": "-",
    },
    "Frozen conv backbone": {
        "file":  "dqn_breakout_freeze_conv_seed42_scalemedium_lr0001_buf100k.csv",
        "color": "#2196F3",
        "style": "-",
    },
    "Full fine-tune from Pong": {
        "file":  "dqn_breakout_finetune_full_seed42_scalemedium_lr0001_buf100k.csv",
        "color": "#F44336",
        "style": "-",
    },
}

SMOOTH_WINDOW = 500


def smooth(values, window):
    if len(values) < window:
        window = max(1, len(values) // 10)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def load_log(path):
    steps, rewards, qvals = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["total_steps"]))
            rewards.append(float(row["reward"]))
            qvals.append(float(row["mean_q"]))
    return np.array(steps), np.array(rewards), np.array(qvals)


def plot_metric(ax, label, steps, values, color, style, smooth_window):
    smoothed = smooth(values, smooth_window)
    # Align x axis after smoothing
    x = steps[smooth_window - 1:]
    if len(x) > len(smoothed):
        x = x[:len(smoothed)]
    elif len(smoothed) > len(x):
        smoothed = smoothed[:len(x)]
    ax.plot(x / 1e6, smoothed, label=label, color=color,
            linestyle=style, linewidth=1.6, alpha=0.9)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = {}
    for label, cfg in RUNS.items():
        path = os.path.join(LOG_DIR, cfg["file"])
        steps, rewards, qvals = load_log(path)
        data[label] = {"steps": steps, "rewards": rewards,
                       "qvals": qvals, **cfg}
        final_r = np.mean(rewards[-50:])
        best_r  = np.max(smooth(rewards, SMOOTH_WINDOW))
        print(f"{label}: final_reward={final_r:.2f}, best_smoothed={best_r:.2f}, "
              f"total_steps={steps[-1]:,}")

    # ── Reward comparison ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    for label, d in data.items():
        plot_metric(ax, label, d["steps"], d["rewards"],
                    d["color"], d["style"], SMOOTH_WINDOW)

    ax.set_xlabel("Training Steps (millions)", fontsize=12)
    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title(
        "Breakout Learning Curves: Scratch vs Frozen Backbone vs Full Fine-tune\n"
        f"({SMOOTH_WINDOW}-episode rolling average)",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5)

    out = os.path.join(OUTPUT_DIR, "backbone_comparison_reward.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n[saved] {out}")

    # ── Q-value comparison ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    for label, d in data.items():
        # Only plot where Q > 0 (before learning starts Q=0)
        mask = d["qvals"] > 0
        steps_q   = d["steps"][mask]
        qvals_pos = d["qvals"][mask]
        plot_metric(ax, label, steps_q, qvals_pos,
                    d["color"], d["style"], SMOOTH_WINDOW)

    ax.set_xlabel("Training Steps (millions)", fontsize=12)
    ax.set_ylabel("Mean Q-Value", fontsize=12)
    ax.set_title(
        "Mean Q-Value During Training: Scratch vs Frozen Backbone vs Full Fine-tune\n"
        f"({SMOOTH_WINDOW}-episode rolling average)",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5)

    out = os.path.join(OUTPUT_DIR, "backbone_comparison_qvalue.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[saved] {out}")

    print("\n[done]")


if __name__ == "__main__":
    main()

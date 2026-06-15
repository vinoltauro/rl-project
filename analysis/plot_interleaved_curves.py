"""
Interleaved Training Curves — v2 vs v3
========================================
Compares step-level (v2) and episode-level (v3) interleaved training.
Three panels: Pong reward, Breakout reward, dead neuron fraction.
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "results/plots"

RUNS = {
    "v2 — step-level (1k steps)": {
        "path":         "results/logs/dqn_interleaved_v2_seed42_scalemedium_metrics.csv",
        "pong_col":     "pong_reward_mean10",
        "breakout_col": "breakout_reward_mean10",
        "dead_col":     "dead_neurons",
        "pong_color":   "#1565C0",
        "br_color":     "#B71C1C",
        "dead_color":   "#6A1B9A",
        "style":        "--",
    },
    "v3 — episode-level (joint buffer)": {
        "path":         "results/interleaved_v3/logs/dqn_interleaved_v3_seed42_scalemedium_metrics.csv",
        "pong_col":     "pong_reward_mean10",
        "breakout_col": "breakout_reward_mean10",
        "dead_col":     "dead_neurons",
        "pong_color":   "#2196F3",
        "br_color":     "#F44336",
        "dead_color":   "#9C27B0",
        "style":        "-",
    },
}


def load_metrics(path, pong_col, breakout_col, dead_col):
    steps, pong_r, br_r, dead = [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(int(float(row["total_step"])))
            pong_r.append(float(row[pong_col]))
            br_r.append(float(row[breakout_col]))
            dead.append(float(row[dead_col]))
    seen, out = set(), []
    for s, p, b, d in zip(steps, pong_r, br_r, dead):
        if s not in seen:
            seen.add(s)
            out.append((s, p, b, d))
    s, p, b, d = zip(*out)
    return np.array(s) / 1e6, list(p), list(b), list(d)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for label, cfg in RUNS.items():
        if not os.path.exists(cfg["path"]):
            print(f"[skip] {cfg['path']} not found")
            continue
        steps, pong_r, br_r, dead = load_metrics(
            cfg["path"], cfg["pong_col"], cfg["breakout_col"], cfg["dead_col"]
        )
        axes[0].plot(steps, pong_r,  color=cfg["pong_color"],
                     linestyle=cfg["style"], linewidth=2,
                     marker="o", markersize=4, label=label)
        axes[1].plot(steps, br_r,    color=cfg["br_color"],
                     linestyle=cfg["style"], linewidth=2,
                     marker="o", markersize=4, label=label)
        axes[2].plot(steps, dead,    color=cfg["dead_color"],
                     linestyle=cfg["style"], linewidth=2,
                     marker="o", markersize=4, label=label)

    # Reference lines
    axes[0].axhline(-21, color="gray", linestyle=":", alpha=0.4, linewidth=1,
                    label="Worst possible (−21)")
    axes[0].axhline(7.5, color="#4CAF50", linestyle="--", alpha=0.4, linewidth=1,
                    label="Standalone Pong (~7.5)")
    axes[1].axhline(3.86, color="#4CAF50", linestyle="--", alpha=0.4, linewidth=1,
                    label="Standalone Breakout (3.86)")

    axes[0].set_title("Pong Reward\n(retention under joint training)", fontsize=10)
    axes[1].set_title("Breakout Reward\n(learning under joint training)", fontsize=10)
    axes[2].set_title("Dead Neuron Fraction\n(capacity loss in fc_repr)", fontsize=10)

    for ax in axes:
        ax.set_xlabel("Total Training Steps (millions)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

    axes[0].set_ylabel("Mean Pong Reward (last 10 eps)", fontsize=9)
    axes[1].set_ylabel("Mean Breakout Reward (last 10 eps)", fontsize=9)
    axes[2].set_ylabel("Dead Neuron Fraction", fontsize=9)
    axes[2].set_ylim(0, 1)

    fig.suptitle(
        "Interleaved Training: Step-Level (v2) vs Episode-Level Joint Buffer (v3)\n"
        "1M steps per game each",
        fontsize=11, y=1.02
    )

    out = os.path.join(OUTPUT_DIR, "interleaved_curves.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")

    print("\nFinal values (last eval point):")
    for label, cfg in RUNS.items():
        if not os.path.exists(cfg["path"]):
            continue
        steps, pong_r, br_r, dead = load_metrics(
            cfg["path"], cfg["pong_col"], cfg["breakout_col"], cfg["dead_col"]
        )
        print(f"  {label}")
        print(f"    Pong:     {pong_r[0]:.1f} → {pong_r[-1]:.1f}")
        print(f"    Breakout: {br_r[0]:.1f} → {br_r[-1]:.1f}")
        print(f"    Dead:     {dead[0]:.3f} → {dead[-1]:.3f}")


if __name__ == "__main__":
    main()

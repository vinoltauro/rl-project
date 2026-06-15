"""
Sequential Reinit Deep Dive
============================
Key finding: Pong forgetting happens at reinit time, not during Breakout training.
Three panels:
  1. Pong reward over Breakout steps — reinit vs no-freeze vs freeze-all
  2. Dead neuron re-accumulation — reinit vs no-freeze (with reinit event annotated)
  3. Breakout reward — reinit vs no-freeze (does reinit help or hurt acquisition?)
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUTPUT_DIR = "results/plots"

REINIT_PATH    = "results/sequential_reinit/logs/dqn_sequential_reinit_seed42_scalemedium_forgetting.csv"
NOFREE_PATH    = "results/logs/dqn_sequential_sequential_full_seed42_scalemedium_forgetting.csv"
FREEZEALL_PATH = "results/logs/dqn_fixed_sequential_freeze_all_seed42_scalemedium_forgetting.csv"


def load_forgetting(path, has_breakout_col=False):
    steps, pong_r, dead, breakout_r = [], [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(float(row["breakout_step"])))
            pong_r.append(float(row["pong_reward"]))
            dead.append(float(row["dead_neurons"]))
            if has_breakout_col and "breakout_reward_mean10" in row:
                breakout_r.append(float(row["breakout_reward_mean10"]))
            else:
                breakout_r.append(None)
    # deduplicate
    seen, out = set(), []
    for s, p, d, b in zip(steps, pong_r, dead, breakout_r):
        if s not in seen:
            seen.add(s)
            out.append((s, p, d, b))
    s, p, d, b = zip(*out)
    return np.array(s)/1e6, list(p), list(d), list(b)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    steps_ri, pong_ri, dead_ri, br_ri = load_forgetting(REINIT_PATH, has_breakout_col=True)
    steps_nf, pong_nf, dead_nf, _     = load_forgetting(NOFREE_PATH)
    steps_fa, pong_fa, dead_fa, _     = load_forgetting(FREEZEALL_PATH)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Sequential + Dead Neuron Reinitialisation — Deep Analysis\n"
        r"388/512 neurons reinitialised (orthogonal, bias=0) before Breakout training",
        fontsize=11, y=1.03
    )

    # ── Panel 1: Pong reward ──────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(steps_nf, pong_nf, color="#F44336", lw=2, marker="o",
            markersize=4, label="No freeze (baseline)")
    ax.plot(steps_fa, pong_fa, color="#4CAF50", lw=2, marker="s",
            markersize=4, label="Freeze all (baseline)")
    ax.plot(steps_ri, pong_ri, color="#9C27B0", lw=2.5, marker="^",
            markersize=5, label="Sequential + reinit (new)", zorder=5)

    # Annotate that step 0 for reinit is AFTER reinit, before Breakout
    ax.annotate(
        "Reinit event:\nPong already at −21\nbefore Breakout starts",
        xy=(steps_ri[0], pong_ri[0]),
        xytext=(0.35, -14),
        fontsize=7.5, color="#9C27B0",
        arrowprops=dict(arrowstyle="->", color="#9C27B0", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#9C27B0", alpha=0.8)
    )

    ax.axhline(-21, color="gray", ls=":", lw=1, alpha=0.5, label="Floor (−21)")
    ax.axhline(9.1, color="gray", ls="--", lw=1, alpha=0.4, label="Pong pretrain (+9.1)")
    ax.set_title("Pong Reward (chimera eval)\nForgetting happens at reinit, not during Breakout", fontsize=9)
    ax.set_xlabel("Breakout Training Steps (millions)", fontsize=9)
    ax.set_ylabel("Mean Pong Reward", fontsize=9)
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    # ── Panel 2: Dead neuron re-accumulation ──────────────────────────────────
    ax = axes[1]
    ax.plot(steps_nf, dead_nf, color="#F44336", lw=2, marker="o",
            markersize=4, label="No freeze (baseline)")
    ax.plot(steps_ri, dead_ri, color="#9C27B0", lw=2.5, marker="^",
            markersize=5, label="Sequential + reinit", zorder=5)

    # Annotate the starting point after reinit
    ax.annotate(
        "After reinit:\n33% dead\n(388 neurons reset)",
        xy=(steps_ri[0], dead_ri[0]),
        xytext=(0.5, 0.40),
        fontsize=7.5, color="#9C27B0",
        arrowprops=dict(arrowstyle="->", color="#9C27B0", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#9C27B0", alpha=0.8)
    )
    # Annotate final
    ax.annotate(
        f"Final: {dead_ri[-1]*100:.0f}%\n(higher than before reinit)",
        xy=(steps_ri[-1], dead_ri[-1]),
        xytext=(1.2, 0.68),
        fontsize=7.5, color="#9C27B0",
        arrowprops=dict(arrowstyle="->", color="#9C27B0", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#9C27B0", alpha=0.8)
    )

    # Shade the re-accumulation region
    ax.axhspan(0.758, 0.775, alpha=0.08, color="#9C27B0")
    ax.axhline(0.758, color="#9C27B0", ls=":", lw=1, alpha=0.4,
               label="Pre-reinit dead fraction (76%)")

    ax.set_title("Dead Neuron Re-accumulation\nBreakout training re-kills neurons at the same rate", fontsize=9)
    ax.set_xlabel("Breakout Training Steps (millions)", fontsize=9)
    ax.set_ylabel("Dead Neuron Fraction (fc_repr)", fontsize=9)
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)

    # ── Panel 3: Breakout reward ───────────────────────────────────────────────
    ax = axes[2]

    # Load no-freeze breakout reward from sequential forgetting CSV
    # It doesn't have breakout_reward — load from training episode CSV instead
    # Use a rolling mean of the episode rewards
    br_steps_nf, br_r_nf = [], []
    with open("results/logs/dqn_sequential_sequential_full_seed42_scalemedium.csv") as f:
        for row in csv.DictReader(f):
            br_steps_nf.append(int(float(row["total_steps"])))
            br_r_nf.append(float(row["reward"]))
    br_steps_nf = np.array(br_steps_nf)
    br_r_nf     = np.array(br_r_nf)

    # Bin into 200k-step windows
    bins = np.arange(0, 2.1e6, 2e5)
    def bin_rewards(steps, rewards, bins):
        xs, ys = [], []
        for i in range(len(bins)-1):
            mask = (steps >= bins[i]) & (steps < bins[i+1])
            if mask.sum() > 0:
                xs.append((bins[i] + bins[i+1]) / 2 / 1e6)
                ys.append(rewards[mask].mean())
        return xs, ys

    xs_nf, ys_nf = bin_rewards(br_steps_nf, br_r_nf, bins)

    ax.plot(xs_nf, ys_nf, color="#F44336", lw=2, marker="o",
            markersize=4, label="No freeze (baseline)", ls="--")

    # Plot reinit breakout reward from forgetting CSV
    br_ri_valid = [(s, b) for s, b in zip(steps_ri, br_ri) if b is not None]
    if br_ri_valid:
        s_ri, b_ri = zip(*br_ri_valid)
        ax.plot(list(s_ri), list(b_ri), color="#9C27B0", lw=2.5, marker="^",
                markersize=5, label="Sequential + reinit", zorder=5)

    ax.axhline(3.86, color="#4CAF50", ls="--", lw=1.2, alpha=0.6,
               label="Standalone Breakout (3.86)")
    ax.set_title("Breakout Reward\nReinit neither helps nor hurts acquisition", fontsize=9)
    ax.set_xlabel("Breakout Training Steps (millions)", fontsize=9)
    ax.set_ylabel("Mean Breakout Reward", fontsize=9)
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "plot_reinit_analysis.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()

"""
Interleaved v3 Deep Dive
=========================
Three key findings from the v3 experiment:
  1. Non-monotonic dead neuron trajectory — joint training initially RECOVERS plasticity
  2. Positive Breakout transfer — v3 reaches standalone Breakout performance ~8x faster
  3. Stable multitask plateau — Pong stabilises around -7 after 700k steps
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "results/plots"

V3_PATH       = "results/interleaved_v3/logs/dqn_interleaved_v3_seed42_scalemedium_metrics.csv"
V2_PATH       = "results/logs/dqn_interleaved_v2_seed42_scalemedium_metrics.csv"
BR_SOLO_PATH  = "results/logs/dqn_breakout_seed42_scalemedium_lr0001_buf100k.csv"


def load_metrics(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({
                "total":    int(float(row["total_step"])),
                "pong_s":   int(float(row.get("pong_steps", 0))),
                "break_s":  int(float(row.get("breakout_steps", 0))),
                "pong_r":   float(row["pong_reward_mean10"]),
                "break_r":  float(row["breakout_reward_mean10"]),
                "dead":     float(row["dead_neurons"]),
            })
    # deduplicate on total_step
    seen, out = set(), []
    for r in rows:
        if r["total"] not in seen:
            seen.add(r["total"])
            out.append(r)
    return out


def load_solo_breakout(path, max_steps=1_000_000):
    """Load standalone Breakout episode data, bin into 50k-step windows."""
    steps, rewards = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            s = int(float(row["total_steps"]))
            if s <= max_steps:
                steps.append(s)
                rewards.append(float(row["reward"]))
    steps   = np.array(steps)
    rewards = np.array(rewards)
    bins = np.arange(0, max_steps + 1, 50_000)
    xs, ys = [], []
    for i in range(len(bins) - 1):
        mask = (steps >= bins[i]) & (steps < bins[i+1])
        if mask.sum() > 0:
            xs.append((bins[i] + bins[i+1]) / 2 / 1e6)
            ys.append(rewards[mask].mean())
    return xs, ys


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    v3 = load_metrics(V3_PATH)
    v2 = load_metrics(V2_PATH) if os.path.exists(V2_PATH) else []

    v3_total  = np.array([r["total"]   for r in v3]) / 1e6
    v3_break  = np.array([r["break_s"] for r in v3]) / 1e6
    v3_pong_r = [r["pong_r"]  for r in v3]
    v3_br_r   = [r["break_r"] for r in v3]
    v3_dead   = [r["dead"]    for r in v3]

    v2_total  = np.array([r["total"] for r in v2]) / 1e6
    v2_dead   = [r["dead"] for r in v2]

    solo_xs, solo_ys = load_solo_breakout(BR_SOLO_PATH)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Interleaved v3 (Episode-Level, Joint Buffer) — Deep Analysis",
        fontsize=12, y=1.03
    )

    # ── Panel 1: Non-monotonic dead neuron trajectory ─────────────────────────
    ax = axes[0]

    if v2:
        ax.plot(v2_total, v2_dead, color="#FF5722", lw=2, ls="--",
                marker="o", markersize=3, alpha=0.7, label="v2 (step-level) — monotonic rise")

    ax.plot(v3_total, v3_dead, color="#9C27B0", lw=2.5, marker="o",
            markersize=5, label="v3 (episode-level) — non-monotonic", zorder=5)

    # Find minimum of v3 dead neurons
    min_idx  = int(np.argmin(v3_dead))
    min_x    = v3_total[min_idx]
    min_y    = v3_dead[min_idx]

    ax.annotate(
        f"Minimum: {min_y*100:.0f}% dead\nat {min_x:.1f}M steps\n← joint training rescues neurons",
        xy=(min_x, min_y),
        xytext=(min_x + 0.3, min_y + 0.12),
        fontsize=7.5, color="#9C27B0",
        arrowprops=dict(arrowstyle="->", color="#9C27B0", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#9C27B0", alpha=0.85)
    )
    ax.annotate(
        f"Final: {v3_dead[-1]*100:.0f}% dead\n(vs 92% for v2)",
        xy=(v3_total[-1], v3_dead[-1]),
        xytext=(1.4, v3_dead[-1] + 0.1),
        fontsize=7.5, color="#9C27B0",
        arrowprops=dict(arrowstyle="->", color="#9C27B0", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#9C27B0", alpha=0.85)
    )

    # Shade the "recovery" phase
    recovery_mask = v3_total <= min_x
    ax.fill_between(v3_total[recovery_mask],
                    np.array(v3_dead)[recovery_mask],
                    min_y,
                    alpha=0.10, color="#4CAF50",
                    label="Recovery phase (neurons rescued)")

    ax.set_title("Dead Neuron Fraction\nNon-monotonic: joint training rescues neurons early", fontsize=9)
    ax.set_xlabel("Total Training Steps (millions)", fontsize=9)
    ax.set_ylabel("Dead Neuron Fraction (fc_repr)", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)

    # ── Panel 2: Positive Breakout transfer ───────────────────────────────────
    ax = axes[1]

    ax.plot(solo_xs, solo_ys, color="#795548", lw=2, ls="--",
            marker="", alpha=0.7, label="Standalone Breakout DQN\n(dedicated 1M steps)")
    ax.plot(v3_break, v3_br_r, color="#F44336", lw=2.5, marker="o",
            markersize=5, label="v3 Breakout (joint training)", zorder=5)

    # Find where v3 first hits ~3.5
    threshold = 3.0
    above = [(x, y) for x, y in zip(v3_break, v3_br_r) if y >= threshold]
    if above:
        first_x, first_y = above[0]
        ax.annotate(
            f"v3 reaches {threshold:.0f} reward\nat {first_x:.2f}M Breakout steps",
            xy=(first_x, first_y),
            xytext=(first_x + 0.1, first_y - 1.5),
            fontsize=7.5, color="#F44336",
            arrowprops=dict(arrowstyle="->", color="#F44336", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#F44336", alpha=0.85)
        )

    ax.axhline(3.86, color="#4CAF50", ls=":", lw=1.5, alpha=0.7,
               label="Standalone Breakout final (3.86)")

    ax.set_title("Breakout Reward vs Breakout-Specific Steps\nJoint training matches standalone Breakout performance", fontsize=9)
    ax.set_xlabel("Breakout Steps (millions)", fontsize=9)
    ax.set_ylabel("Mean Breakout Reward (last 10 eps)", fontsize=9)
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    # ── Panel 3: Pong plateau ─────────────────────────────────────────────────
    ax = axes[2]

    ax.plot(v3_total, v3_pong_r, color="#2196F3", lw=2.5, marker="o",
            markersize=5, label="v3 Pong reward (trained from scratch)", zorder=5)

    # Find plateau start (~700k steps where Pong stabilises)
    plateau_step = 0.7
    plateau_vals = [y for x, y in zip(v3_total, v3_pong_r) if x >= plateau_step]
    plateau_mean = np.mean(plateau_vals) if plateau_vals else -7

    ax.axhspan(plateau_mean - 2, plateau_mean + 2, alpha=0.08, color="#2196F3")
    ax.axvline(plateau_step, color="#2196F3", ls=":", lw=1.5, alpha=0.6)
    ax.annotate(
        f"Stable plateau from ~{plateau_step:.1f}M steps\nmean ≈ {plateau_mean:.1f}",
        xy=(plateau_step, plateau_mean),
        xytext=(plateau_step + 0.2, plateau_mean + 4),
        fontsize=7.5, color="#2196F3",
        arrowprops=dict(arrowstyle="->", color="#2196F3", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2196F3", alpha=0.85)
    )

    ax.axhline(7.5,  color="#4CAF50", ls="--", lw=1.2, alpha=0.5,
               label="Standalone Pong DQN (~7.5)")
    ax.axhline(-21,  color="gray",    ls=":", lw=1, alpha=0.4,
               label="Floor (−21)")
    ax.axhline(-14.0, color="#1565C0", ls="--", lw=1.2, alpha=0.5,
               label="Freeze-conv sequential (−14.0)")

    ax.set_title("Pong Reward Over Training\nStable multitask plateau after 700k steps", fontsize=9)
    ax.set_xlabel("Total Training Steps (millions)", fontsize=9)
    ax.set_ylabel("Mean Pong Reward (last 10 eps)", fontsize=9)
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "plot_v3_analysis.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()

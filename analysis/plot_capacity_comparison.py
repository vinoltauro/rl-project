"""
Capacity Loss Across All Conditions
=====================================
Single focused plot: dead neuron fraction over time for all conditions.
Highlights the non-monotonic v3 curve and the re-accumulation in reinit.
Groups conditions by colour family: sequential (reds), interleaved (oranges/purples).
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "results/plots"

CONDITIONS = [
    {
        "label":    "No freeze (sequential)",
        "path":     "results/logs/dqn_sequential_sequential_full_seed42_scalemedium_forgetting.csv",
        "step_col": "breakout_step",
        "dead_col": "dead_neurons",
        "color":    "#F44336",
        "style":    "-",
        "lw":       1.8,
        "marker":   "o",
        "ms":       3,
        "group":    "sequential",
    },
    {
        "label":    "Freeze conv only (sequential)",
        "path":     "results/logs/dqn_fixed_sequential_freeze_conv_fixed_seed42_scalemedium_forgetting.csv",
        "step_col": "breakout_step",
        "dead_col": "dead_neurons",
        "color":    "#EF9A9A",
        "style":    "--",
        "lw":       1.5,
        "marker":   "s",
        "ms":       3,
        "group":    "sequential",
    },
    {
        "label":    "Freeze all (sequential)",
        "path":     "results/logs/dqn_fixed_sequential_freeze_all_seed42_scalemedium_forgetting.csv",
        "step_col": "breakout_step",
        "dead_col": "dead_neurons",
        "color":    "#FFCDD2",
        "style":    ":",
        "lw":       1.5,
        "marker":   "^",
        "ms":       3,
        "group":    "sequential",
    },
    {
        "label":    "Sequential + dead reinit (new)",
        "path":     "results/sequential_reinit/logs/dqn_sequential_reinit_seed42_scalemedium_forgetting.csv",
        "step_col": "breakout_step",
        "dead_col": "dead_neurons",
        "color":    "#9C27B0",
        "style":    "-.",
        "lw":       2.2,
        "marker":   "D",
        "ms":       4,
        "group":    "sequential",
    },
    {
        "label":    "Interleaved v2 (step-level)",
        "path":     "results/logs/dqn_interleaved_v2_seed42_scalemedium_metrics.csv",
        "step_col": "total_step",
        "dead_col": "dead_neurons",
        "color":    "#FF5722",
        "style":    "--",
        "lw":       1.8,
        "marker":   "o",
        "ms":       3,
        "group":    "interleaved",
    },
    {
        "label":    "Interleaved v3 — joint buffer (new)",
        "path":     "results/interleaved_v3/logs/dqn_interleaved_v3_seed42_scalemedium_metrics.csv",
        "step_col": "total_step",
        "dead_col": "dead_neurons",
        "color":    "#FF9800",
        "style":    "-",
        "lw":       2.5,
        "marker":   "o",
        "ms":       5,
        "group":    "interleaved",
    },
]


def load(path, step_col, dead_col):
    steps, dead = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(int(float(row[step_col])))
            dead.append(float(row[dead_col]))
    seen, out = set(), []
    for s, d in zip(steps, dead):
        if s not in seen:
            seen.add(s)
            out.append((s, d))
    s, d = zip(*out)
    return np.array(s)/1e6, list(d)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Capacity Loss (Dead Neuron Fraction in fc_repr) — All Conditions\n"
        "Threshold: neuron dead if post-ReLU activation = 0 in >95% of sampled frames",
        fontsize=10, y=1.03
    )

    # ── Left: all conditions on one axis ─────────────────────────────────────
    ax = axes[0]
    v3_steps, v3_dead = None, None

    for cond in CONDITIONS:
        if not os.path.exists(cond["path"]):
            print(f"[skip] {cond['path']}")
            continue
        steps, dead = load(cond["path"], cond["step_col"], cond["dead_col"])
        ax.plot(steps, dead, color=cond["color"], ls=cond["style"],
                lw=cond["lw"], marker=cond["marker"], markersize=cond["ms"],
                label=cond["label"], zorder=5 if "v3" in cond["label"] else 3)
        if "v3" in cond["label"]:
            v3_steps, v3_dead = steps, dead

    ax.set_title("All Conditions", fontsize=10)
    ax.set_xlabel("Training Steps (millions)", fontsize=9)
    ax.set_ylabel("Dead Neuron Fraction", fontsize=9)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.0)

    # ── Right: v3 non-monotonic curve zoomed + annotated ─────────────────────
    ax = axes[1]

    # v2 for comparison
    v2_cond = next(c for c in CONDITIONS if "v2" in c["label"])
    if os.path.exists(v2_cond["path"]):
        s2, d2 = load(v2_cond["path"], v2_cond["step_col"], v2_cond["dead_col"])
        ax.plot(s2, d2, color=v2_cond["color"], ls="--", lw=1.8,
                marker="o", markersize=3, alpha=0.6, label="v2 (step-level) — monotonic")

    if v3_steps is not None:
        ax.plot(v3_steps, v3_dead, color="#FF9800", lw=2.5, marker="o",
                markersize=5, label="v3 (episode-level) — non-monotonic", zorder=5)

        min_idx = int(np.argmin(v3_dead))
        min_x, min_y = v3_steps[min_idx], v3_dead[min_idx]

        # Shade recovery phase
        ax.fill_between(
            v3_steps[:min_idx+1], v3_dead[:min_idx+1], min_y,
            alpha=0.15, color="#4CAF50",
        )
        ax.annotate(
            "Recovery phase:\njoint gradients keep\nneurons alive",
            xy=(v3_steps[min_idx//2], (v3_dead[0] + min_y)/2),
            xytext=(0.05, 0.52),
            fontsize=7.5, color="#2E7D32",
            arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=1.1),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2E7D32", alpha=0.85)
        )
        ax.annotate(
            f"Minimum at {min_x:.1f}M steps\n{min_y*100:.0f}% dead",
            xy=(min_x, min_y),
            xytext=(min_x + 0.2, min_y + 0.08),
            fontsize=7.5, color="#FF9800",
            arrowprops=dict(arrowstyle="->", color="#FF9800", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#FF9800", alpha=0.85)
        )

        # Shade re-specialisation phase
        ax.fill_between(
            v3_steps[min_idx:], v3_dead[min_idx:], min_y,
            alpha=0.10, color="#F44336",
        )
        ax.annotate(
            "Re-specialisation:\nneurons die as both\ngames converge",
            xy=(v3_steps[min_idx + (len(v3_steps)-min_idx)//2],
                (v3_dead[min_idx] + v3_dead[-1])/2 + 0.02),
            xytext=(1.3, 0.46),
            fontsize=7.5, color="#B71C1C",
            arrowprops=dict(arrowstyle="->", color="#B71C1C", lw=1.1),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#B71C1C", alpha=0.85)
        )

    ax.set_title("v3 Non-Monotonic Dead Neuron Curve\n(early recovery → late re-specialisation)", fontsize=9)
    ax.set_xlabel("Total Training Steps (millions)", fontsize=9)
    ax.set_ylabel("Dead Neuron Fraction", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.0)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "plot_capacity_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()

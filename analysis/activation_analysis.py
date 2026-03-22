"""
Activation Analysis
===================
Three analyses beyond t-SNE:

  1. Dead Neuron Analysis (Fig 13)
     — Tracks what fraction of the 512-dim layer neurons are chronically
       inactive (output 0 > 95% of the time). Dead neurons = wasted capacity.
     — Expected: DQN has more dead neurons than DDQN.

  2. Q-Value Overestimation Plot (Fig 3)
     — Compares mean max Q-value over training: DQN drifts up, DDQN stays
       grounded. Direct empirical evidence of the bias DDQN was designed to fix.
     — Reads from the CSV logs produced by the Logger.

  3. Cosine Similarity Between Game Representations (Fig 14)
     — Measures how similar the mean representation vector is for Pong vs
       Breakout states, separately for DQN and DDQN.
     — Tests whether structurally similar games produce similar internal
       representations, and whether the algorithm modulates this.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────
# 1. Dead Neuron Analysis
# ─────────────────────────────────────────────────────────────────────────
def fig_dead_neurons(repr_dir: str, output_dir: str, threshold: float = 0.95):
    """
    Fig 13: Dead neuron fraction per agent at each training checkpoint.

    Args:
        repr_dir:   Directory with .npz representation files
        output_dir: Directory to save figures
        threshold:  Fraction of zero activations above which neuron = dead
    """
    import glob

    keys = {
        "dqn_pong":     ("DQN / Pong",     "#2196F3"),
        "ddqn_pong":    ("DDQN / Pong",    "#4CAF50"),
        "dqn_breakout": ("DQN / Breakout", "#FF9800"),
        "ddqn_breakout":("DDQN / Breakout","#9C27B0"),
    }

    fig, ax = plt.subplots(figsize=(9, 5))

    for key, (label, colour) in keys.items():
        pattern = os.path.join(repr_dir, f"repr_{key}*.npz")
        files   = sorted(glob.glob(pattern))

        if not files:
            print(f"  [SKIP] No files for {key}")
            continue

        steps        = []
        dead_fractions = []

        for f in files:
            data = np.load(f, allow_pickle=True)
            reprs = data["representations"]          # (N, 512)
            step  = int(data["step_at_ckpt"][0])

            # Fraction of neurons with > threshold zero activations
            zero_frac = (reprs == 0.0).mean(axis=0)  # (512,)
            dead_frac = float((zero_frac > threshold).mean())

            steps.append(step)
            dead_fractions.append(dead_frac * 100)   # Convert to %

        ax.plot(steps, dead_fractions, marker="o", label=label, color=colour, linewidth=2)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Dead Neurons (%)")
    ax.set_title("Dead Neuron Analysis: Fraction of Chronically Inactive Neurons\n"
                 f"(inactive > {threshold*100:.0f}% of the time)")
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, "dead_neurons.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[dead neurons] Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────
# 2. Q-Value Overestimation
# ─────────────────────────────────────────────────────────────────────────
def fig_qvalue_overestimation(log_dir: str, output_dir: str):
    """
    Fig 3: Mean max Q-value over training for all 4 runs.

    DQN should show upward drift (overestimation).
    DDQN should stay relatively stable.

    Reads from CSV logs produced by utils/logger.py.
    """
    run_info = {
        "dqn_pong":     ("DQN / Pong",      "#2196F3", "--"),
        "ddqn_pong":    ("DDQN / Pong",     "#4CAF50", "-"),
        "dqn_breakout": ("DQN / Breakout",  "#FF9800", "--"),
        "ddqn_breakout":("DDQN / Breakout", "#9C27B0", "-"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for game_idx, game in enumerate(["pong", "breakout"]):
        ax = axes[game_idx]

        for algo in ["dqn", "ddqn"]:
            key = f"{algo}_{game}"
            label, colour, ls = run_info[key]

            # Find CSV for this run
            pattern = os.path.join(log_dir, f"{key}*.csv")
            files   = glob.glob(pattern)
            if not files:
                print(f"  [SKIP] No log CSV for {key}")
                continue

            df = pd.read_csv(files[0])
            if "mean_q" not in df.columns:
                continue

            # Smooth with rolling window
            smoothed = df["mean_q"].rolling(20, min_periods=1).mean()
            ax.plot(
                df["total_steps"], smoothed,
                label=label, color=colour, linestyle=ls, linewidth=2
            )

        ax.set_title(f"Q-Value Overestimation: {game.capitalize()}")
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Mean Max Q-Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Q-Value Overestimation: DQN vs Double DQN\n"
                 "DQN overestimates; DDQN is more stable", fontsize=13)
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, "qvalue_overestimation.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[Q-value] Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────
# 3. Cosine Similarity Between Game Representations
# ─────────────────────────────────────────────────────────────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D vectors."""
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def fig_cosine_similarity(repr_dir: str, output_dir: str):
    """
    Fig 14: Cosine similarity between Pong and Breakout representations
    at each checkpoint, separately for DQN and DDQN.

    If the games share visual features (ball, paddle), we expect non-trivial
    similarity. If representations are game-specific, similarity will be low.
    """
    import glob

    fig, ax = plt.subplots(figsize=(9, 5))

    for algo, colour, label in [
        ("dqn",  "#2196F3", "DQN"),
        ("ddqn", "#4CAF50", "Double DQN"),
    ]:
        pong_files     = sorted(glob.glob(os.path.join(repr_dir, f"repr_{algo}_pong*.npz")))
        breakout_files = sorted(glob.glob(os.path.join(repr_dir, f"repr_{algo}_breakout*.npz")))

        if not pong_files or not breakout_files:
            print(f"  [SKIP] Missing files for {algo}")
            continue

        # Match by checkpoint index (may not have same steps if games differ)
        n = min(len(pong_files), len(breakout_files))
        similarities = []
        steps = []

        for i in range(n):
            data_p = np.load(pong_files[i],     allow_pickle=True)
            data_b = np.load(breakout_files[i], allow_pickle=True)

            mean_pong     = data_p["representations"].mean(axis=0)
            mean_breakout = data_b["representations"].mean(axis=0)
            sim = cosine_similarity(mean_pong, mean_breakout)

            step = int(data_p["step_at_ckpt"][0])
            similarities.append(sim)
            steps.append(step)

        ax.plot(steps, similarities, marker="o", label=label,
                color=colour, linewidth=2)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Cosine Similarity: Pong vs Breakout Mean Representations\n"
                 "Higher = more shared representational structure")
    ax.legend()
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, "cosine_similarity.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[cosine] Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────
# Training curves (Figs 1–2)
# ─────────────────────────────────────────────────────────────────────────
def fig_training_curves(log_dir: str, output_dir: str):
    """
    Figs 1 & 2: Episode reward over training steps.
    DQN vs DDQN on the same plot, one figure per game.
    Includes smoothed curve + raw data shading.
    """
    for game in ["pong", "breakout"]:
        fig, ax = plt.subplots(figsize=(9, 5))

        for algo, colour, ls, label in [
            ("dqn",  "#2196F3", "--", "DQN"),
            ("ddqn", "#4CAF50", "-",  "Double DQN"),
        ]:
            pattern = os.path.join(log_dir, f"{algo}_{game}*.csv")
            files   = glob.glob(pattern)
            if not files:
                print(f"  [SKIP] {algo}_{game}")
                continue

            df = pd.read_csv(files[0])
            raw      = df["reward"]
            smoothed = raw.rolling(20, min_periods=1).mean()
            steps    = df["total_steps"]

            ax.plot(steps, smoothed, color=colour, linestyle=ls,
                    linewidth=2, label=label)
            ax.fill_between(steps,
                            raw.rolling(20, min_periods=1).quantile(0.25),
                            raw.rolling(20, min_periods=1).quantile(0.75),
                            alpha=0.15, color=colour)

        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Episode Reward")
        ax.set_title(f"Training Curves: {game.capitalize()}\nDQN vs Double DQN")
        ax.legend()
        ax.grid(True, alpha=0.3)

        os.makedirs(output_dir, exist_ok=True)
        out = os.path.join(output_dir, f"training_curves_{game}.png")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[training curves] Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────
def run_all(log_dir: str, repr_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*55}")
    print("  Generating activation analysis figures")
    print(f"{'='*55}\n")

    print("── Training curves (Figs 1–2) ────────────────────────")
    fig_training_curves(log_dir, output_dir)

    print("── Q-value overestimation (Fig 3) ────────────────────")
    fig_qvalue_overestimation(log_dir, output_dir)

    print("── Dead neuron analysis (Fig 13) ─────────────────────")
    fig_dead_neurons(repr_dir, output_dir)

    print("── Cosine similarity (Fig 14) ────────────────────────")
    fig_cosine_similarity(repr_dir, output_dir)

    print(f"\n✓ All activation figures saved to: {output_dir}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir",    default="results/logs")
    parser.add_argument("--repr_dir",   default="results/representations")
    parser.add_argument("--output_dir", default="results/plots")
    args = parser.parse_args()
    run_all(args.log_dir, args.repr_dir, args.output_dir)

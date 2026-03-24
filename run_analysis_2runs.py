"""
Analysis for Runs 1 & 2 only — DQN/Pong and DQN/Breakout.
Produces clean figures with only the completed DQN runs.

Usage:
    python run_analysis_2runs.py

Figures produced:
    training_curves_pong.png         - DQN/Pong learning curve
    training_curves_breakout.png     - DQN/Breakout learning curve
    qvalue_overestimation.png        - DQN Q-value drift (both games)
    tsne_game_effect_dqn.png         - DQN/Pong vs DQN/Breakout clusters
    tsne_by_reward_dqn_pong.png      - DQN/Pong coloured by reward
    tsne_by_reward_dqn_breakout.png  - DQN/Breakout coloured by reward
    tsne_temporal_dqn_pong.png       - DQN/Pong evolution over training
    tsne_temporal_dqn_breakout.png   - DQN/Breakout evolution over training
    dead_neurons.png                 - Dead neuron fraction (DQN runs)
    cosine_similarity.png            - Cross-game similarity (DQN only)
"""

import os
import sys
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis.extract_representations import collect_representations

# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR   = "results/checkpoints"
REPR_DIR         = "results/representations/dqn_only"
OUTPUT_DIR       = "results/plots/dqn_only"
LOG_DIR          = "results/logs"

RUNS = [
    {"run_name": "dqn_pong_seed42_scalemedium_lr0001_buf100k",
     "env_id":   "ALE/Pong-v5",
     "n_actions": 6,
     "label":    "DQN / Pong",
     "colour":   "#2196F3"},

    {"run_name": "dqn_breakout_seed42_scalemedium_lr0001_buf100k",
     "env_id":   "ALE/Breakout-v5",
     "n_actions": 4,
     "label":    "DQN / Breakout",
     "colour":   "#FF9800"},
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(REPR_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Device : {DEVICE}")
print(f"Repr   : {REPR_DIR}")
print(f"Plots  : {OUTPUT_DIR}\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Extract representations
# ─────────────────────────────────────────────────────────────────────────────
def extract():
    print("=" * 55)
    print("  STEP 1 — Extracting representations")
    print("=" * 55)

    for run in RUNS:
        run_name    = run["run_name"]
        pattern     = os.path.join(CHECKPOINT_DIR, f"{run_name}_step*.pt")
        checkpoints = sorted(glob.glob(pattern))

        if not checkpoints:
            print(f"[ERROR] No checkpoints found for {run_name}")
            sys.exit(1)

        print(f"\n[{run['label']}] — {len(checkpoints)} checkpoint(s)")

        for ckpt_path in checkpoints:
            step     = int(ckpt_path.split("_step")[-1].replace(".pt", ""))
            out_path = os.path.join(REPR_DIR, f"repr_{run_name}_step{step:08d}.npz")

            if os.path.exists(out_path):
                print(f"  [skip] step {step:,} — already extracted")
                continue

            print(f"  Extracting step {step:,} ...")
            data = collect_representations(
                checkpoint_path = ckpt_path,
                env_id          = run["env_id"],
                n_actions       = run["n_actions"],
                net_scale       = "medium",
                n_steps         = 5000,
                seed            = 99,
                device          = DEVICE,
            )
            np.savez_compressed(
                out_path,
                representations = data["representations"],
                actions         = data["actions"],
                rewards         = data["rewards"],
                cumulative_r    = data["cumulative_r"],
                done_flags      = data["done_flags"],
                step_at_ckpt    = np.array([data["step_at_ckpt"]]),
            )
            print(f"    Saved → {os.path.basename(out_path)}")

    print("\n[extract] Done.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Helper — run t-SNE
# ─────────────────────────────────────────────────────────────────────────────
def run_tsne(reprs):
    X = StandardScaler().fit_transform(reprs)
    return TSNE(n_components=2, perplexity=30, max_iter=1000,
                random_state=42, verbose=0).fit_transform(X)


def load_final(run_name, max_points=3000):
    files = sorted(glob.glob(os.path.join(REPR_DIR, f"repr_{run_name}*.npz")))
    if not files:
        raise FileNotFoundError(f"No repr files for {run_name}")
    data = np.load(files[-1], allow_pickle=True)
    reprs  = data["representations"]
    cum_r  = data["cumulative_r"]
    if len(reprs) > max_points:
        idx = np.random.choice(len(reprs), max_points, replace=False)
        idx.sort()
        reprs = reprs[idx]
        cum_r = cum_r[idx]
    return reprs, cum_r


def load_all_checkpoints(run_name):
    files = sorted(glob.glob(os.path.join(REPR_DIR, f"repr_{run_name}*.npz")))
    out = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        out.append((int(d["step_at_ckpt"][0]), d["representations"]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Training curves + Q-value (reads CSV logs)
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_and_qvalue():
    import pandas as pd
    print("=" * 55)
    print("  STEP 2 — Training curves + Q-value")
    print("=" * 55)

    for run in RUNS:
        key   = run["run_name"].split("_seed")[0]   # dqn_pong or dqn_breakout
        game  = "pong" if "pong" in key else "breakout"
        files = glob.glob(os.path.join(LOG_DIR, f"{key}*.csv"))
        if not files:
            print(f"  [SKIP] No CSV for {key}")
            continue

        df       = pd.read_csv(files[0])
        raw      = df["reward"]
        smoothed = raw.rolling(20, min_periods=1).mean()
        steps    = df["total_steps"]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(steps, smoothed, color=run["colour"], linewidth=2, label=run["label"])
        ax.fill_between(steps,
                        raw.rolling(20, min_periods=1).quantile(0.25),
                        raw.rolling(20, min_periods=1).quantile(0.75),
                        alpha=0.15, color=run["colour"])
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Episode Reward")
        ax.set_title(f"Training Curve: {game.capitalize()} — DQN")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = os.path.join(OUTPUT_DIR, f"training_curves_{game}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved → {out}")

    # Q-value overestimation — both games side by side
    import pandas as pd
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for i, run in enumerate(RUNS):
        key   = run["run_name"].split("_seed")[0]
        game  = "pong" if "pong" in key else "breakout"
        files = glob.glob(os.path.join(LOG_DIR, f"{key}*.csv"))
        if not files:
            continue
        df = pd.read_csv(files[0])
        if "mean_q" not in df.columns:
            continue
        smoothed = df["mean_q"].rolling(20, min_periods=1).mean()
        axes[i].plot(df["total_steps"], smoothed,
                     color=run["colour"], linewidth=2, label=run["label"])
        axes[i].set_title(f"Q-Value: {game.capitalize()}")
        axes[i].set_xlabel("Training Steps")
        axes[i].set_ylabel("Mean Max Q-Value")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    fig.suptitle("Q-Value over Training — DQN", fontsize=13)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "qvalue_overestimation.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — t-SNE: game effect (Pong vs Breakout)
# ─────────────────────────────────────────────────────────────────────────────
def plot_game_effect():
    print("=" * 55)
    print("  STEP 3 — t-SNE: Game Effect (DQN)")
    print("=" * 55)

    r_pong,     _ = load_final(RUNS[0]["run_name"])
    r_breakout, _ = load_final(RUNS[1]["run_name"])

    combined = np.vstack([r_pong, r_breakout])
    labels   = np.array(["Pong"] * len(r_pong) + ["Breakout"] * len(r_breakout))

    print(f"  Running t-SNE on {len(combined)} points ...")
    emb = run_tsne(combined)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, colour, run in [("Pong", RUNS[0]["colour"], RUNS[0]),
                               ("Breakout", RUNS[1]["colour"], RUNS[1])]:
        mask = labels == name
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=colour, alpha=0.5, s=8, label=run["label"])

    ax.set_title("t-SNE: Game Effect (DQN)\nSame algorithm, different games")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(markerscale=3)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "tsne_game_effect_dqn.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — t-SNE: coloured by reward (one plot per run)
# ─────────────────────────────────────────────────────────────────────────────
def plot_reward():
    print("=" * 55)
    print("  STEP 4 — t-SNE: Coloured by Reward")
    print("=" * 55)

    for run in RUNS:
        reprs, cum_r = load_final(run["run_name"], max_points=2000)
        print(f"  Running t-SNE for {run['label']} ...")
        emb = run_tsne(reprs)

        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=cum_r,
                        cmap="RdYlGn", alpha=0.6, s=8)
        plt.colorbar(sc, ax=ax, label="Cumulative Reward")
        ax.set_title(f"t-SNE: Coloured by Reward\n{run['label']}")
        ax.set_xlabel("t-SNE Dim 1"); ax.set_ylabel("t-SNE Dim 2")
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        game = "pong" if "pong" in run["run_name"] else "breakout"
        out  = os.path.join(OUTPUT_DIR, f"tsne_by_reward_dqn_{game}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved → {out}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — t-SNE: temporal evolution (one plot per run)
# ─────────────────────────────────────────────────────────────────────────────
def plot_temporal():
    print("=" * 55)
    print("  STEP 5 — t-SNE: Temporal Evolution")
    print("=" * 55)

    for run in RUNS:
        ckpts = load_all_checkpoints(run["run_name"])
        n     = len(ckpts)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]
        cmap = plt.cm.plasma

        for i, (step, reprs) in enumerate(ckpts):
            if len(reprs) > 1500:
                idx    = np.random.choice(len(reprs), 1500, replace=False)
                reprs  = reprs[idx]
            print(f"  t-SNE {run['label']} step {step:,} ...")
            emb = run_tsne(reprs)
            axes[i].scatter(emb[:, 0], emb[:, 1],
                            c=[cmap(i / n)] * len(emb),
                            alpha=0.5, s=5)
            axes[i].set_title(f"Step {step:,}", fontsize=10)
            axes[i].set_xticks([]); axes[i].set_yticks([])

        fig.suptitle(f"t-SNE: Representation Evolution — {run['label']}", fontsize=13)
        fig.tight_layout()
        game = "pong" if "pong" in run["run_name"] else "breakout"
        out  = os.path.join(OUTPUT_DIR, f"tsne_temporal_dqn_{game}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved → {out}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Dead neurons
# ─────────────────────────────────────────────────────────────────────────────
def plot_dead_neurons(threshold=0.95):
    print("=" * 55)
    print("  STEP 6 — Dead Neuron Analysis")
    print("=" * 55)

    fig, ax = plt.subplots(figsize=(9, 5))

    for run in RUNS:
        files = sorted(glob.glob(os.path.join(REPR_DIR, f"repr_{run['run_name']}*.npz")))
        steps, fracs = [], []
        for f in files:
            d     = np.load(f, allow_pickle=True)
            reprs = d["representations"]
            step  = int(d["step_at_ckpt"][0])
            zero_frac = (reprs == 0.0).mean(axis=0)
            dead_frac = float((zero_frac > threshold).mean()) * 100
            steps.append(step)
            fracs.append(dead_frac)
        ax.plot(steps, fracs, marker="o", label=run["label"],
                color=run["colour"], linewidth=2)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Dead Neurons (%)")
    ax.set_title(f"Dead Neuron Analysis — DQN\n(inactive > {threshold*100:.0f}% of the time)")
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "dead_neurons.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Cosine similarity
# ─────────────────────────────────────────────────────────────────────────────
def plot_cosine_similarity():
    print("=" * 55)
    print("  STEP 7 — Cosine Similarity")
    print("=" * 55)

    pong_files     = sorted(glob.glob(os.path.join(REPR_DIR, f"repr_{RUNS[0]['run_name']}*.npz")))
    breakout_files = sorted(glob.glob(os.path.join(REPR_DIR, f"repr_{RUNS[1]['run_name']}*.npz")))
    n = min(len(pong_files), len(breakout_files))

    steps, sims = [], []
    for i in range(n):
        dp = np.load(pong_files[i],     allow_pickle=True)
        db = np.load(breakout_files[i], allow_pickle=True)
        mp = dp["representations"].mean(axis=0)
        mb = db["representations"].mean(axis=0)
        mp /= (np.linalg.norm(mp) + 1e-8)
        mb /= (np.linalg.norm(mb) + 1e-8)
        sims.append(float(np.dot(mp, mb)))
        steps.append(int(dp["step_at_ckpt"][0]))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, sims, marker="o", color="#2196F3", linewidth=2, label="DQN")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Cosine Similarity: Pong vs Breakout Mean Representations\nDQN only")
    ax.legend()
    ax.set_ylim(-0.1, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "cosine_similarity.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    extract()
    plot_training_and_qvalue()
    plot_game_effect()
    plot_reward()
    plot_temporal()
    plot_dead_neurons()
    plot_cosine_similarity()

    print("=" * 55)
    print(f"  All done. Figures saved to: {OUTPUT_DIR}")
    print("=" * 55)
    print("\nFiles produced:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  {f}")

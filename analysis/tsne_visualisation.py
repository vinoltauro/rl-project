"""
t-SNE Visualisation Pipeline
=============================
Produces all t-SNE figures from collected representation .npz files.

Figures produced:
  Fig 4: Game effect — DQN/Pong vs DQN/Breakout
  Fig 5: Game effect — DDQN/Pong vs DDQN/Breakout
  Fig 6: Algorithm effect — DQN vs DDQN on Pong
  Fig 7: Algorithm effect — DQN vs DDQN on Breakout
  Fig 8: All 4 agents together
  Fig 9: Coloured by cumulative reward
  Fig 10: Temporal evolution over training checkpoints

Usage:
    python analysis/tsne_visualisation.py --repr_dir results/representations \
                                          --output_dir results/plots

Notes:
  - t-SNE is stochastic; use random_state for reproducibility
  - Perplexity should be between 5 and 50; 30 is a good default
  - We subsample to max 3000 points per run for speed
"""

import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend for server/Colab
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import argparse

# ── Plot style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":       150,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.facecolor": "white",
})

# ── Colour palette ────────────────────────────────────────────────────────
COLOURS = {
    "dqn_pong":     "#2196F3",   # Blue
    "ddqn_pong":    "#4CAF50",   # Green
    "dqn_breakout": "#FF9800",   # Orange
    "ddqn_breakout":"#9C27B0",   # Purple
}

LABELS = {
    "dqn_pong":     "DQN / Pong",
    "ddqn_pong":    "DDQN / Pong",
    "dqn_breakout": "DQN / Breakout",
    "ddqn_breakout":"DDQN / Breakout",
}


# ─────────────────────────────────────────────────────────────────────────
# Loading utilities
# ─────────────────────────────────────────────────────────────────────────
def load_repr_file(path: str) -> dict:
    """Load a single .npz representation file."""
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def load_run_final(repr_dir: str, key: str, max_points: int = 3000) -> dict:
    """
    Load the final checkpoint representations for a given run key.
    key is one of: dqn_pong, ddqn_pong, dqn_breakout, ddqn_breakout

    Subsamples to max_points for t-SNE speed.
    """
    pattern = os.path.join(repr_dir, f"repr_{key}*")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No representation files found matching: {pattern}\n"
            f"Run analysis/extract_representations.py first."
        )
    # Use the last file (final checkpoint)
    data = load_repr_file(files[-1])

    repr_arr = data["representations"]
    if len(repr_arr) > max_points:
        idx = np.random.choice(len(repr_arr), max_points, replace=False)
        idx.sort()
        for k in ["representations", "actions", "rewards", "cumulative_r", "done_flags"]:
            if k in data:
                data[k] = data[k][idx]

    return data


def load_all_checkpoints(repr_dir: str, key: str) -> list:
    """Load ALL checkpoint files for a run (for temporal evolution plot)."""
    pattern = os.path.join(repr_dir, f"repr_{key}*")
    files = sorted(glob.glob(pattern))
    return [load_repr_file(f) for f in files]


# ─────────────────────────────────────────────────────────────────────────
# t-SNE computation
# ─────────────────────────────────────────────────────────────────────────
def run_tsne(
    representations: np.ndarray,
    perplexity:      float = 30.0,
    random_state:    int   = 42,
    n_iter:          int   = 1000,
) -> np.ndarray:
    """
    Run t-SNE on representation vectors.

    Args:
        representations: (N, D) float array
        perplexity:      t-SNE perplexity (5–50, default 30)
        random_state:    For reproducibility
        n_iter:          Number of t-SNE iterations

    Returns:
        (N, 2) embedded coordinates
    """
    # Standardise before t-SNE
    scaler = StandardScaler()
    X = scaler.fit_transform(representations)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        verbose=0,
    )
    return tsne.fit_transform(X)


# ─────────────────────────────────────────────────────────────────────────
# Individual figure generators
# ─────────────────────────────────────────────────────────────────────────
def fig_game_effect(repr_dir: str, output_dir: str, algo: str = "dqn"):
    """
    Figs 4 & 5: Game effect — same algorithm, different games.
    Shows whether similar games produce similar representations.
    """
    key_game1 = f"{algo}_pong"
    key_game2 = f"{algo}_breakout"

    data1 = load_run_final(repr_dir, key_game1)
    data2 = load_run_final(repr_dir, key_game2)

    r1 = data1["representations"]
    r2 = data2["representations"]

    # Combine and run t-SNE jointly (important: must be embedded together)
    combined = np.vstack([r1, r2])
    labels = np.array(["Pong"] * len(r1) + ["Breakout"] * len(r2))

    print(f"[t-SNE] Game effect ({algo.upper()}): {len(combined)} points...")
    emb = run_tsne(combined)

    fig, ax = plt.subplots(figsize=(8, 6))
    c1 = COLOURS[key_game1]
    c2 = COLOURS[key_game2]

    mask1 = labels == "Pong"
    ax.scatter(emb[mask1, 0], emb[mask1, 1],
               c=c1, alpha=0.5, s=8, label=f"{algo.upper()} / Pong")
    ax.scatter(emb[~mask1, 0], emb[~mask1, 1],
               c=c2, alpha=0.5, s=8, label=f"{algo.upper()} / Breakout")

    ax.set_title(f"t-SNE: Game Effect ({algo.upper()})\n"
                 f"Same algorithm, different games")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(loc="best", markerscale=3)
    ax.set_xticks([])
    ax.set_yticks([])

    out = os.path.join(output_dir, f"tsne_game_effect_{algo}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def fig_algorithm_effect(repr_dir: str, output_dir: str, game: str = "pong"):
    """
    Figs 6 & 7: Algorithm effect — same game, different algorithms.
    Shows how DQN vs DDQN affects representation geometry.
    """
    key_dqn  = f"dqn_{game}"
    key_ddqn = f"ddqn_{game}"

    data_dqn  = load_run_final(repr_dir, key_dqn)
    data_ddqn = load_run_final(repr_dir, key_ddqn)

    r_dqn  = data_dqn["representations"]
    r_ddqn = data_ddqn["representations"]

    combined = np.vstack([r_dqn, r_ddqn])
    labels   = np.array(["DQN"] * len(r_dqn) + ["DDQN"] * len(r_ddqn))

    print(f"[t-SNE] Algorithm effect ({game}): {len(combined)} points...")
    emb = run_tsne(combined)

    fig, ax = plt.subplots(figsize=(8, 6))

    mask_dqn = labels == "DQN"
    ax.scatter(emb[mask_dqn, 0], emb[mask_dqn, 1],
               c=COLOURS[key_dqn], alpha=0.5, s=8, label="DQN")
    ax.scatter(emb[~mask_dqn, 0], emb[~mask_dqn, 1],
               c=COLOURS[key_ddqn], alpha=0.5, s=8, label="Double DQN")

    ax.set_title(f"t-SNE: Algorithm Effect ({game.capitalize()})\n"
                 f"Same game, different algorithms")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(loc="best", markerscale=3)
    ax.set_xticks([])
    ax.set_yticks([])

    out = os.path.join(output_dir, f"tsne_algo_effect_{game}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def fig_all_four_agents(repr_dir: str, output_dir: str, max_per_run: int = 1500):
    """
    Fig 8: All 4 agents in one t-SNE plot.
    Reveals whether game or algorithm is the dominant clustering factor.
    """
    keys = ["dqn_pong", "dqn_breakout", "ddqn_pong", "ddqn_breakout"]
    all_repr = []
    all_keys = []

    for key in keys:
        try:
            data = load_run_final(repr_dir, key, max_points=max_per_run)
            all_repr.append(data["representations"])
            all_keys.extend([key] * len(data["representations"]))
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")

    if len(all_repr) < 2:
        print("[WARNING] Need at least 2 runs for all-agents plot")
        return

    combined  = np.vstack(all_repr)
    all_keys  = np.array(all_keys)

    print(f"[t-SNE] All 4 agents: {len(combined)} points...")
    emb = run_tsne(combined)

    fig, ax = plt.subplots(figsize=(9, 7))
    for key in keys:
        mask = all_keys == key
        if mask.sum() == 0:
            continue
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=COLOURS[key], alpha=0.5, s=6,
                   label=LABELS[key])

    ax.set_title("t-SNE: All 4 Agents\n"
                 "Do game or algorithm differences dominate representation structure?")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(loc="best", markerscale=4, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    out = os.path.join(output_dir, "tsne_all_agents.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def fig_coloured_by_reward(repr_dir: str, output_dir: str):
    """
    Fig 9: t-SNE coloured by cumulative episode reward.
    Tests whether the representation encodes performance / game state value.
    """
    keys = ["dqn_pong", "dqn_breakout", "ddqn_pong", "ddqn_breakout"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, key in enumerate(keys):
        try:
            data = load_run_final(repr_dir, key, max_points=2000)
        except FileNotFoundError:
            axes[idx].set_visible(False)
            continue

        r = data["representations"]
        cum_r = data["cumulative_r"]

        emb = run_tsne(r)

        sc = axes[idx].scatter(
            emb[:, 0], emb[:, 1],
            c=cum_r, cmap="RdYlGn",
            alpha=0.6, s=6,
        )
        plt.colorbar(sc, ax=axes[idx], label="Cumulative Reward")
        axes[idx].set_title(LABELS[key])
        axes[idx].set_xlabel("t-SNE Dim 1")
        axes[idx].set_ylabel("t-SNE Dim 2")
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
        print(f"  [t-SNE reward] {key} done")

    fig.suptitle("t-SNE: Representations Coloured by Cumulative Reward\n"
                 "Do high-reward states cluster together?", fontsize=13)
    fig.tight_layout()
    out = os.path.join(output_dir, "tsne_by_reward.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def fig_temporal_evolution(repr_dir: str, output_dir: str, key: str = "dqn_pong"):
    """
    Fig 10: How representations evolve over training checkpoints.
    Runs t-SNE at each saved checkpoint to show learning dynamics.
    """
    all_ckpts = load_all_checkpoints(repr_dir, key)
    if not all_ckpts:
        print(f"[WARNING] No checkpoints found for {key}")
        return

    n_ckpts = len(all_ckpts)
    cols = min(n_ckpts, 4)
    rows = (n_ckpts + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    cmap = plt.cm.get_cmap("plasma", n_ckpts)

    for i, (ckpt_data, ax) in enumerate(zip(all_ckpts, axes)):
        r = ckpt_data["representations"]
        step = int(ckpt_data.get("step_at_ckpt", np.array([i]))[0])

        # Subsample
        if len(r) > 1500:
            idx = np.random.choice(len(r), 1500, replace=False)
            r = r[idx]

        emb = run_tsne(r)
        ax.scatter(emb[:, 0], emb[:, 1],
                   c=[cmap(i / n_ckpts)] * len(emb),
                   alpha=0.5, s=5)
        ax.set_title(f"Step {step:,}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        print(f"  [temporal] {key} step {step:,} done")

    # Hide unused axes
    for ax in axes[n_ckpts:]:
        ax.set_visible(False)

    fig.suptitle(f"t-SNE: Representation Evolution over Training\n{LABELS.get(key, key)}",
                 fontsize=13)
    fig.tight_layout()
    out = os.path.join(output_dir, f"tsne_temporal_{key}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────
# Run all figures
# ─────────────────────────────────────────────────────────────────────────
def run_all(repr_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*55}")
    print(f"  Generating all t-SNE figures")
    print(f"  Source: {repr_dir}")
    print(f"  Output: {output_dir}")
    print(f"{'='*55}\n")

    print("── Fig 4: Game effect (DQN) ──────────────────────────")
    fig_game_effect(repr_dir, output_dir, algo="dqn")

    print("── Fig 5: Game effect (DDQN) ─────────────────────────")
    fig_game_effect(repr_dir, output_dir, algo="ddqn")

    print("── Fig 6: Algorithm effect (Pong) ────────────────────")
    fig_algorithm_effect(repr_dir, output_dir, game="pong")

    print("── Fig 7: Algorithm effect (Breakout) ────────────────")
    fig_algorithm_effect(repr_dir, output_dir, game="breakout")

    print("── Fig 8: All 4 agents ───────────────────────────────")
    fig_all_four_agents(repr_dir, output_dir)

    print("── Fig 9: Coloured by reward ─────────────────────────")
    fig_coloured_by_reward(repr_dir, output_dir)

    print("── Fig 10: Temporal evolution ────────────────────────")
    for key in ["dqn_pong", "ddqn_breakout"]:
        fig_temporal_evolution(repr_dir, output_dir, key=key)

    print(f"\n✓ All t-SNE figures saved to: {output_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repr_dir",   default="results/representations")
    parser.add_argument("--output_dir", default="results/plots")
    parser.add_argument("--figure",     default="all",
                        help="all | game_dqn | game_ddqn | algo_pong | algo_breakout | all4 | reward | temporal")
    args = parser.parse_args()

    if args.figure == "all":
        run_all(args.repr_dir, args.output_dir)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        dispatch = {
            "game_dqn":    lambda: fig_game_effect(args.repr_dir, args.output_dir, "dqn"),
            "game_ddqn":   lambda: fig_game_effect(args.repr_dir, args.output_dir, "ddqn"),
            "algo_pong":   lambda: fig_algorithm_effect(args.repr_dir, args.output_dir, "pong"),
            "algo_breakout":lambda: fig_algorithm_effect(args.repr_dir, args.output_dir, "breakout"),
            "all4":        lambda: fig_all_four_agents(args.repr_dir, args.output_dir),
            "reward":      lambda: fig_coloured_by_reward(args.repr_dir, args.output_dir),
            "temporal":    lambda: fig_temporal_evolution(args.repr_dir, args.output_dir),
        }
        dispatch.get(args.figure, lambda: print(f"Unknown figure: {args.figure}"))()

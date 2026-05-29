"""
Layer-wise Representation Similarity (CKA)
==========================================
Measures Centred Kernel Alignment (CKA) between agent representations
at each layer: Conv1, Conv2, Conv3, and the 512-dim FC representation layer.

Tests the claim that early convolutional layers generalise across games
while the representation layer specialises.

Four comparisons:
  - Game effect (DQN):      DQN/Pong  vs DQN/Breakout
  - Game effect (DDQN):     DDQN/Pong vs DDQN/Breakout
  - Algorithm effect (Pong):     DQN/Pong  vs DDQN/Pong
  - Algorithm effect (Breakout): DQN/Breakout vs DDQN/Breakout

Usage:
    python analysis/layer_similarity.py
    python analysis/layer_similarity.py --n_steps 2000  # more samples = more stable CKA
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn import AtariCNN
from envs.wrappers import make_atari_env
from utils.checkpoint import load_model_for_analysis


# ─────────────────────────────────────────────────────────────────────────────
# CKA
# ─────────────────────────────────────────────────────────────────────────────
def _center_gram(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    ones = np.ones((n, n)) / n
    return K - ones @ K - K @ ones + ones @ K @ ones


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear CKA between two activation matrices.

    Args:
        X: (N, D1) float array
        Y: (N, D2) float array — must have same N

    Returns:
        Scalar in [0, 1]. 1.0 = identical representational geometry.
    """
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    K = X @ X.T   # (N, N)
    L = Y @ Y.T   # (N, N)

    Kc = _center_gram(K)
    Lc = _center_gram(L)

    hsic_xy = np.sum(Kc * Lc)
    denom   = np.sqrt(np.sum(Kc * Kc) * np.sum(Lc * Lc))

    return float(hsic_xy / denom) if denom > 1e-10 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Activation collection
# ─────────────────────────────────────────────────────────────────────────────
def collect_layer_activations(
    checkpoint_path: str,
    env_id:          str,
    n_actions:       int,
    n_steps:         int          = 1000,
    device:          torch.device = torch.device("cpu"),
    seed:            int          = 99,
) -> dict:
    """
    Load a checkpoint and collect activations at all four layers.

    Conv layers are flattened to (N, C*H*W) for CKA computation.

    Returns:
        dict: {"conv1": (N,D), "conv2": (N,D), "conv3": (N,D), "fc_repr": (N,512)}
    """
    model = AtariCNN(n_actions=n_actions, net_scale="medium").to(device)
    load_model_for_analysis(checkpoint_path, model, device)
    model.eval()

    storage = {"conv1": [], "conv2": [], "conv3": [], "fc_repr": []}
    handles = []

    def make_hook(name):
        def hook(module, input, output):
            act = output.detach().cpu().numpy()
            if act.ndim == 4:                        # conv: (B, C, H, W)
                act = act.reshape(act.shape[0], -1)  # → (B, C*H*W)
            storage[name].append(act)
        return hook

    # Conv Sequential layout: [Conv2d, ReLU, Conv2d, ReLU, Conv2d, ReLU]
    #                indices:     0       1     2       3     4       5
    handles.append(model.conv[1].register_forward_hook(make_hook("conv1")))
    handles.append(model.conv[3].register_forward_hook(make_hook("conv2")))
    handles.append(model.conv[5].register_forward_hook(make_hook("conv3")))
    handles.append(model.fc_repr.register_forward_hook(make_hook("fc_repr")))

    env = make_atari_env(env_id, seed=seed)
    obs, _ = env.reset()

    with torch.no_grad():
        for _ in range(n_steps):
            state_t = (
                torch.from_numpy(obs).float().div(255.0)
                .unsqueeze(0).to(device)
            )
            q_vals = model(state_t)
            action = int(q_vals.argmax(dim=1).item())
            next_obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
            else:
                obs = next_obs

    env.close()
    for h in handles:
        h.remove()

    return {name: np.concatenate(acts, axis=0) for name, acts in storage.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint discovery
# ─────────────────────────────────────────────────────────────────────────────
def find_final_checkpoint(checkpoint_dir: str, run_prefix: str) -> str:
    pattern = os.path.join(checkpoint_dir, f"{run_prefix}*.pt")
    files   = sorted(f for f in glob.glob(pattern) if ".tmp" not in f)
    if not files:
        raise FileNotFoundError(
            f"No checkpoint found matching: {pattern}\n"
            "Make sure all 4 training runs are complete."
        )
    return files[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def run(checkpoint_dir: str, output_dir: str, n_steps: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    RUNS = {
        "dqn_pong":      ("dqn_pong_seed42_scalemedium_lr0001_buf100k",      "ALE/Pong-v5",     6),
        "dqn_breakout":  ("dqn_breakout_seed42_scalemedium_lr0001_buf100k",  "ALE/Breakout-v5", 4),
        "ddqn_pong":     ("ddqn_pong_seed42_scalemedium_lr0001_buf100k",     "ALE/Pong-v5",     6),
        "ddqn_breakout": ("ddqn_breakout_seed42_scalemedium_lr0001_buf100k", "ALE/Breakout-v5", 4),
    }

    # ── Collect activations ───────────────────────────────────────────────────
    all_acts = {}
    for key, (prefix, env_id, n_actions) in RUNS.items():
        print(f"\n[collect] {key} ...")
        try:
            ckpt = find_final_checkpoint(checkpoint_dir, prefix)
            print(f"  checkpoint: {os.path.basename(ckpt)}")
            all_acts[key] = collect_layer_activations(
                ckpt, env_id, n_actions, n_steps=n_steps, device=device
            )
            for layer, arr in all_acts[key].items():
                print(f"  {layer}: {arr.shape}")
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")

    if len(all_acts) < 2:
        print("[ERROR] Need at least 2 runs to compute similarity. Exiting.")
        return

    # ── Compute CKA ───────────────────────────────────────────────────────────
    LAYERS       = ["conv1",  "conv2",  "conv3",  "fc_repr"]
    LAYER_LABELS = ["Conv 1", "Conv 2", "Conv 3", "FC Repr\n(512-dim)"]

    COMPARISONS = {
        "Game effect (DQN)":          ("dqn_pong",     "dqn_breakout"),
        "Game effect (DDQN)":         ("ddqn_pong",    "ddqn_breakout"),
        "Algorithm effect (Pong)":    ("dqn_pong",     "ddqn_pong"),
        "Algorithm effect (Breakout)":("dqn_breakout", "ddqn_breakout"),
    }

    COLOURS = {
        "Game effect (DQN)":           "#2196F3",
        "Game effect (DDQN)":          "#4CAF50",
        "Algorithm effect (Pong)":     "#FF9800",
        "Algorithm effect (Breakout)": "#9C27B0",
    }

    print(f"\n[CKA] Computing (subsample N=1000 per comparison) ...")
    results = {}

    for comp_name, (key_a, key_b) in COMPARISONS.items():
        if key_a not in all_acts or key_b not in all_acts:
            print(f"  [SKIP] {comp_name} — missing run data")
            continue

        results[comp_name] = []
        for layer in LAYERS:
            A = all_acts[key_a][layer]
            B = all_acts[key_b][layer]

            n = min(len(A), len(B), 1000)
            rng = np.random.default_rng(42)
            A_sub = A[rng.choice(len(A), n, replace=False)]
            B_sub = B[rng.choice(len(B), n, replace=False)]

            cka = linear_cka(A_sub, B_sub)
            results[comp_name].append(cka)
            print(f"  {comp_name:<35} {layer:<10} CKA = {cka:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(LAYERS))

    for comp_name, cka_values in results.items():
        ax.plot(
            x, cka_values,
            marker="o", linewidth=2, markersize=7,
            label=comp_name, color=COLOURS.get(comp_name, "gray"),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(LAYER_LABELS, fontsize=10)
    ax.set_ylabel("Linear CKA (similarity)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        "Layer-wise Representational Similarity (CKA)\n"
        "High early, low late = early layers generalise, representation layer specialises"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4, linewidth=1,
               label="_nolegend_")

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, "layer_similarity_cka.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n[done] Saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="results/checkpoints")
    parser.add_argument("--output_dir",     default="results/plots")
    parser.add_argument("--n_steps",        type=int, default=1000,
                        help="Env steps to collect per agent (more = more stable CKA)")
    args = parser.parse_args()
    run(args.checkpoint_dir, args.output_dir, args.n_steps)

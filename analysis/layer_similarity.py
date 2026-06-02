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
def collect_frames(env_id: str, n_frames: int, seed: int = 99) -> list:
    """
    Collect n_frames observations from env_id using random actions.
    Returns a list of (4, 84, 84) uint8 numpy arrays.

    Using random actions ensures diverse coverage of the state space
    and avoids any dependence on a specific agent's policy.
    """
    env = make_atari_env(env_id, seed=seed)
    obs, _ = env.reset()
    frames = []
    while len(frames) < n_frames:
        frames.append(obs.copy())
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    return frames[:n_frames]


def collect_activations_on_frames(
    checkpoint_path: str,
    n_actions:       int,
    frames:          list,
    device:          torch.device = torch.device("cpu"),
) -> dict:
    """
    Run a checkpoint on a FIXED set of frames and collect activations.

    Both networks in a comparison must use the same frames — this is the
    key requirement for CKA to be meaningful.

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
            if act.ndim == 4:
                act = act.reshape(act.shape[0], -1)
            storage[name].append(act)
        return hook

    # Conv Sequential layout: [Conv2d, ReLU, Conv2d, ReLU, Conv2d, ReLU]
    handles.append(model.conv[1].register_forward_hook(make_hook("conv1")))
    handles.append(model.conv[3].register_forward_hook(make_hook("conv2")))
    handles.append(model.conv[5].register_forward_hook(make_hook("conv3")))
    handles.append(model.fc_repr.register_forward_hook(make_hook("fc_repr")))

    with torch.no_grad():
        for obs in frames:
            state_t = (
                torch.from_numpy(obs).float().div(255.0)
                .unsqueeze(0).to(device)
            )
            _ = model(state_t)

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

    CKPTS = {
        "dqn_pong":      ("dqn_pong_seed42_scalemedium_lr0001_buf100k",      6),
        "dqn_breakout":  ("dqn_breakout_seed42_scalemedium_lr0001_buf100k",  4),
        "ddqn_pong":     ("ddqn_pong_seed42_scalemedium_lr0001_buf100k",     6),
        "ddqn_breakout": ("ddqn_breakout_seed42_scalemedium_lr0001_buf100k", 4),
    }

    # ── Collect shared frame sets ─────────────────────────────────────────────
    # CKA requires BOTH networks to process the SAME inputs.
    # We collect frames independently of any agent policy (random actions)
    # so the frame set is neutral and not biased toward either agent.
    print(f"\n[frames] Collecting {n_steps} shared Pong frames ...")
    pong_frames     = collect_frames("ALE/Pong-v5",     n_steps, seed=42)
    print(f"[frames] Collecting {n_steps} shared Breakout frames ...")
    breakout_frames = collect_frames("ALE/Breakout-v5", n_steps, seed=42)

    # Game-effect comparisons use Pong frames (neutral probe of both networks
    # on the same visual input — what did each network learn to do with Pong pixels?)
    FRAME_SETS = {
        "dqn_pong":      pong_frames,
        "ddqn_pong":     pong_frames,
        "dqn_breakout":  pong_frames,     # run Breakout agent on Pong frames
        "ddqn_breakout": pong_frames,
    }

    # ── Collect activations on shared frames ──────────────────────────────────
    all_acts = {}
    for key, (prefix, n_actions) in CKPTS.items():
        print(f"\n[collect] {key} on shared frames ...")
        try:
            ckpt = find_final_checkpoint(checkpoint_dir, prefix)
            print(f"  checkpoint: {os.path.basename(ckpt)}")
            all_acts[key] = collect_activations_on_frames(
                ckpt, n_actions, FRAME_SETS[key], device=device
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

    print(f"\n[CKA] Computing on shared frames (N={n_steps}) ...")
    results = {}

    for comp_name, (key_a, key_b) in COMPARISONS.items():
        if key_a not in all_acts or key_b not in all_acts:
            print(f"  [SKIP] {comp_name} — missing run data")
            continue

        results[comp_name] = []
        for layer in LAYERS:
            A = all_acts[key_a][layer]
            B = all_acts[key_b][layer]
            cka = linear_cka(A, B)
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

"""
Representation Extractor
========================
Collects 512-dim activation vectors from the penultimate layer of
trained agents — the raw material for t-SNE analysis.

For each agent checkpoint:
  1. Load model weights (no optimiser needed)
  2. Run agent in environment for N steps (eval mode, ε=0.05)
  3. At each step, the forward hook on fc_repr populates model.representation
  4. Collect representation + metadata: game, algorithm, step, action, reward

Output: saved as numpy .npz files, one per (run, checkpoint) combination.
These are then loaded by tsne_visualisation.py.
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.wrappers import make_atari_env
from models.cnn import AtariCNN
from utils.checkpoint import load_model_for_analysis, list_checkpoints


def collect_representations(
    checkpoint_path: str,
    env_id:          str,
    n_actions:       int,
    net_scale:       str  = "medium",
    n_steps:         int  = 5000,
    seed:            int  = 0,
    device:          torch.device = torch.device("cpu"),
    epsilon_eval:    float = 0.05,
) -> dict:
    """
    Load a checkpoint and collect representations from a live rollout.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        env_id:          Gymnasium env ID
        n_actions:       Number of discrete actions
        net_scale:       CNN scale matching the training config
        n_steps:         How many env steps to collect representations for
        seed:            Env seed (use different from training seed)
        device:          Torch device
        epsilon_eval:    Small epsilon for near-greedy rollout

    Returns:
        dict with keys:
          "representations": (N, hidden_size) float32 array
          "actions":         (N,) int array
          "rewards":         (N,) float array
          "cumulative_r":    (N,) float array
          "done_flags":      (N,) bool array
          "step_at_ckpt":    int — training step of this checkpoint
    """
    # ── Load model ────────────────────────────────────────────────────────
    model = AtariCNN(n_actions=n_actions, net_scale=net_scale).to(device)
    training_step = load_model_for_analysis(checkpoint_path, model, device)
    model.eval()

    # ── Run environment ───────────────────────────────────────────────────
    env = make_atari_env(env_id, seed=seed)
    obs, _ = env.reset()

    representations = []
    actions         = []
    rewards         = []
    cumulative_r    = []
    done_flags      = []

    cum_r = 0.0

    with torch.no_grad():
        for _ in range(n_steps):
            # Select action (near-greedy)
            if np.random.random() < epsilon_eval:
                action = env.action_space.sample()
            else:
                state_t = (
                    torch.from_numpy(obs)
                    .float().div(255.0)
                    .unsqueeze(0).to(device)
                )
                q_vals = model(state_t)
                action = int(q_vals.argmax(dim=1).item())

            # The forward hook populates model.representation
            repr_vec = model.representation.numpy().squeeze(0)  # (hidden_size,)

            # Step environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            cum_r += reward

            representations.append(repr_vec.copy())
            actions.append(action)
            rewards.append(float(reward))
            cumulative_r.append(cum_r)
            done_flags.append(done)

            if done:
                obs, _ = env.reset()
                cum_r = 0.0
            else:
                obs = next_obs

    env.close()

    return {
        "representations": np.array(representations, dtype=np.float32),
        "actions":         np.array(actions,         dtype=np.int32),
        "rewards":         np.array(rewards,         dtype=np.float32),
        "cumulative_r":    np.array(cumulative_r,    dtype=np.float32),
        "done_flags":      np.array(done_flags,      dtype=bool),
        "step_at_ckpt":    training_step,
    }


def extract_all_runs(
    checkpoint_dir:  str,
    output_dir:      str,
    run_configs:     list,
    n_steps_per_run: int = 5000,
    device:          torch.device = torch.device("cpu"),
):
    """
    Extract representations for all runs and all their checkpoints.

    Args:
        checkpoint_dir: Directory containing .pt checkpoint files
        output_dir:     Directory to save .npz representation files
        run_configs:    List of dicts, each with keys:
                          run_name, env_id, n_actions, algorithm, net_scale
        n_steps_per_run: Steps to collect per checkpoint
        device:         Torch device

    Example run_configs:
        [
          {"run_name": "dqn_pong_seed42_...",   "env_id": "ALE/Pong-v5",     "n_actions": 6,  "algorithm": "dqn",  "net_scale": "medium"},
          {"run_name": "ddqn_pong_seed42_...",  "env_id": "ALE/Pong-v5",     "n_actions": 6,  "algorithm": "ddqn", "net_scale": "medium"},
          {"run_name": "dqn_breakout_seed42_...","env_id": "ALE/Breakout-v5", "n_actions": 4,  "algorithm": "dqn",  "net_scale": "medium"},
          {"run_name": "ddqn_breakout_seed42_...","env_id": "ALE/Breakout-v5","n_actions": 4,  "algorithm": "ddqn", "net_scale": "medium"},
        ]
    """
    os.makedirs(output_dir, exist_ok=True)

    for run_cfg in run_configs:
        run_name = run_cfg["run_name"]
        checkpoints = list_checkpoints(checkpoint_dir, run_name)

        if not checkpoints:
            print(f"[WARNING] No checkpoints found for {run_name}")
            continue

        print(f"\n[extract] {run_name} — {len(checkpoints)} checkpoints")

        for ckpt_path in checkpoints:
            print(f"  Processing: {os.path.basename(ckpt_path)}")
            data = collect_representations(
                checkpoint_path=ckpt_path,
                env_id=run_cfg["env_id"],
                n_actions=run_cfg["n_actions"],
                net_scale=run_cfg.get("net_scale", "medium"),
                n_steps=n_steps_per_run,
                device=device,
            )

            # Save with metadata embedded in filename
            step = data["step_at_ckpt"]
            save_path = os.path.join(
                output_dir,
                f"repr_{run_name}_step{step:08d}.npz"
            )
            np.savez_compressed(
                save_path,
                representations = data["representations"],
                actions         = data["actions"],
                rewards         = data["rewards"],
                cumulative_r    = data["cumulative_r"],
                done_flags      = data["done_flags"],
                step_at_ckpt    = np.array([step]),
                run_name        = np.array([run_name]),
                env_id          = np.array([run_cfg["env_id"]]),
                algorithm       = np.array([run_cfg["algorithm"]]),
            )
            print(f"    Saved {data['representations'].shape} → {save_path}")

    print(f"\n[extract] Done. Files in: {output_dir}")


if __name__ == "__main__":
    # Example: extract representations from a single checkpoint
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env",        required=True)
    parser.add_argument("--n_actions",  type=int, required=True)
    parser.add_argument("--output",     default="results/representations")
    parser.add_argument("--n_steps",    type=int, default=5000)
    args = parser.parse_args()

    data = collect_representations(
        checkpoint_path=args.checkpoint,
        env_id=args.env,
        n_actions=args.n_actions,
        n_steps=args.n_steps,
    )
    os.makedirs(args.output, exist_ok=True)
    out = os.path.join(args.output, "repr_single.npz")
    np.savez_compressed(out, **data)
    print(f"Saved: {out} | shape: {data['representations'].shape}")

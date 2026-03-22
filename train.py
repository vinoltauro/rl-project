"""
Main Training Script
====================
Runs a single training experiment defined by a YAML config file.

Usage:
    python train.py --config experiments/configs/run1_dqn_pong.yaml
    python train.py --config experiments/configs/run2_dqn_breakout.yaml
    python train.py --config experiments/configs/run3_ddqn_pong.yaml
    python train.py --config experiments/configs/run4_ddqn_breakout.yaml

    # Ablations:
    python train.py --config experiments/configs/ablation_net_small.yaml

    # Resume from checkpoint:
    python train.py --config experiments/configs/run1_dqn_pong.yaml \
                    --resume results/checkpoints/dqn_pong_seed42_step01000000.pt

    # Quick smoke test (100k steps):
    python train.py --config experiments/configs/run1_dqn_pong.yaml --steps 100000
"""

import os
import sys
import argparse
import random
import time

import numpy as np
import torch
import yaml

# ── Path setup (run from project root) ───────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.wrappers import make_atari_env
from agents.dqn import DQNAgent
from agents.ddqn import DoubleDQNAgent
from utils.replay_buffer import ReplayBuffer
from utils.logger import Logger
from utils.checkpoint import save_checkpoint, load_checkpoint


# ─────────────────────────────────────────────────────────────────────────
def set_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_run_name(cfg: dict) -> str:
    """Generate a descriptive run name from config."""
    algo = cfg["algorithm"]
    game = cfg["env_id"].split("/")[-1].lower().replace("-v5", "")
    seed = cfg["seed"]
    scale = cfg.get("net_scale", "medium")
    lr_str = str(cfg.get("lr", 1e-4)).replace("0.", "").replace("-", "")
    buf = cfg.get("buffer_capacity", 100_000) // 1000
    return f"{algo}_{game}_seed{seed}_scale{scale}_lr{lr_str}_buf{buf}k"


def get_device(force_cpu: bool = False) -> torch.device:
    """
    Auto-detect the best available device.
    Priority: CUDA (NVIDIA) → MPS (Apple Silicon) → CPU
    """
    if force_cpu:
        print("[device] Forced CPU mode")
        return torch.device("cpu")

    # Check for .device file written by setup_env.py
    device_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".device")
    if os.path.exists(device_file):
        with open(device_file) as f:
            stored = f.read().strip()
        if stored == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            _print_device_info(device)
            return device
        elif stored == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            _print_device_info(device)
            return device

    # Auto-detect fallback
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    _print_device_info(device)
    return device


def _print_device_info(device: torch.device):
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[device] NVIDIA GPU: {name} ({mem:.1f} GB)")
    elif device.type == "mps":
        print(f"[device] Apple Silicon GPU (MPS)")
    else:
        print(f"[device] CPU — training will be slow; see README for faster alternatives")


# ─────────────────────────────────────────────────────────────────────────
def apply_cpu_optimisations(cfg: dict) -> dict:
    """
    When running on CPU, automatically scale down settings to keep
    training feasible. All changes are printed so the user is aware.

    Key changes:
      - Smaller replay buffer       (less RAM, faster sampling)
      - Larger batch size           (better CPU utilisation)
      - More frequent target update (compensates for fewer steps)
      - Checkpoints more frequently (shorter recovery window)
    """
    print("\n[cpu-mode] Applying CPU-optimised settings:")
    overrides = {
        "buffer_capacity":    50_000,   # 100k → 50k   (saves ~3 GB RAM)
        "batch_size":         64,        # 32 → 64      (better CPU parallelism)
        "target_update_freq": 500,       # 1000 → 500
        "checkpoint_freq":    250_000,   # 500k → 250k  (more frequent saves)
    }
    for k, v in overrides.items():
        old = cfg.get(k, "—")
        cfg[k] = v
        print(f"  {k:<25} {old} → {v}")
    print()
    return cfg


def train(cfg: dict, resume_path: str = None, steps_override: int = None,
          force_cpu: bool = False):
    """
    Main training loop.

    Args:
        cfg:            Config dictionary loaded from YAML
        resume_path:    Optional path to a checkpoint to resume from
        steps_override: Override total_steps (useful for smoke tests)
        force_cpu:      Force CPU even if GPU is available
    """

    # ── Setup ─────────────────────────────────────────────────────────────
    total_steps = steps_override or cfg["total_steps"]
    set_seeds(cfg["seed"])
    device   = get_device(force_cpu=force_cpu)
    run_name = make_run_name(cfg)

    # Apply CPU optimisations if needed
    if device.type == "cpu":
        cfg = apply_cpu_optimisations(cfg)

    print(f"\n{'='*60}")
    print(f"  Run: {run_name}")
    print(f"  Algorithm: {cfg['algorithm'].upper()}")
    print(f"  Game:      {cfg['env_id']}")
    print(f"  Steps:     {total_steps:,}")
    print(f"{'='*60}\n")

    # ── Environment ───────────────────────────────────────────────────────
    env = make_atari_env(cfg["env_id"], seed=cfg["seed"])
    n_actions = env.action_space.n
    print(f"[env] {cfg['env_id']} | actions={n_actions} | "
          f"obs={env.observation_space.shape}")

    # ── Agent ─────────────────────────────────────────────────────────────
    AgentClass = DoubleDQNAgent if cfg["algorithm"] == "ddqn" else DQNAgent
    agent = AgentClass(
        n_actions=n_actions,
        device=device,
        lr=cfg["lr"],
        gamma=cfg["gamma"],
        epsilon_start=cfg["epsilon_start"],
        epsilon_end=cfg["epsilon_end"],
        epsilon_decay_steps=cfg["epsilon_decay_steps"],
        target_update_freq=cfg["target_update_freq"],
        net_scale=cfg.get("net_scale", "medium"),
        grad_clip=cfg.get("grad_clip", 10.0),
    )

    # ── Replay Buffer ─────────────────────────────────────────────────────
    buffer = ReplayBuffer(
        capacity=cfg["buffer_capacity"],
        obs_shape=(4, 84, 84),
        device=device,
    )
    print(f"[buffer] capacity={cfg['buffer_capacity']:,} | "
          f"memory≈{buffer.memory_usage_mb():.0f}MB")

    # ── Logger ────────────────────────────────────────────────────────────
    logger = Logger(
        log_dir=cfg["log_dir"],
        run_name=run_name,
        use_tb=cfg.get("tb_logging", True),
    )

    # ── Resume from checkpoint ────────────────────────────────────────────
    start_step = 0
    start_episode = 0
    if resume_path:
        state = load_checkpoint(
            resume_path, agent.online_net, agent.target_net,
            agent.optimizer, device
        )
        start_step    = state["step"]
        start_episode = state["episode"]
        agent.epsilon = state["epsilon"]
        agent._step   = start_step
        print(f"[resume] Continuing from step {start_step:,}")

    # ── Training Loop ─────────────────────────────────────────────────────
    global_step  = start_step
    episode      = start_episode
    best_mean_r  = -float("inf")

    obs, _ = env.reset()
    ep_reward = 0
    ep_length = 0
    ep_start  = time.time()

    print(f"\n[training] Starting ... filling buffer for {cfg['learning_starts']:,} steps\n")

    while global_step < total_steps:

        # ── Select and execute action ──────────────────────────────────────
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # ── Store transition ───────────────────────────────────────────────
        buffer.push(obs, action, float(reward), next_obs, done)
        obs        = next_obs
        ep_reward += reward
        ep_length += 1
        global_step += 1

        # ── Learn (once buffer is ready) ───────────────────────────────────
        loss, mean_q = None, None
        if global_step >= cfg["learning_starts"] and len(buffer) >= cfg["batch_size"]:
            batch = buffer.sample(cfg["batch_size"])
            loss, mean_q = agent.learn(batch)
            logger.log_step(loss, mean_q)

        # ── Episode end ────────────────────────────────────────────────────
        if done:
            episode += 1
            mean_r = logger.log_episode(ep_reward, ep_length, agent.epsilon)

            if episode % cfg.get("print_freq", 10) == 0:
                logger.print_progress(ep_reward, agent.epsilon)

            # Track best performance
            if mean_r > best_mean_r:
                best_mean_r = mean_r

            # Reset episode
            obs, _ = env.reset()
            ep_reward = 0
            ep_length = 0

        # ── Checkpoint ─────────────────────────────────────────────────────
        checkpoint_freq = cfg.get("checkpoint_freq", 500_000)
        if global_step % checkpoint_freq == 0:
            save_checkpoint(
                checkpoint_dir=cfg["checkpoint_dir"],
                run_name=run_name,
                step=global_step,
                model=agent.online_net,
                target_model=agent.target_net,
                optimizer=agent.optimizer,
                episode=episode,
                epsilon=agent.epsilon,
                config=cfg,
            )

    # ── Final save ────────────────────────────────────────────────────────
    save_checkpoint(
        checkpoint_dir=cfg["checkpoint_dir"],
        run_name=run_name,
        step=global_step,
        model=agent.online_net,
        target_model=agent.target_net,
        optimizer=agent.optimizer,
        episode=episode,
        epsilon=agent.epsilon,
        config=cfg,
    )

    env.close()
    logger.close()

    print(f"\n{'='*60}")
    print(f"  Training complete: {run_name}")
    print(f"  Total steps: {global_step:,} | Episodes: {episode}")
    print(f"  Best mean reward (10-ep): {best_mean_r:.2f}")
    print(f"{'='*60}\n")

    return run_name


# ─────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train DQN/DDQN on Atari — runs on NVIDIA GPU, Apple MPS, or CPU"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--resume", default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Override total_steps (e.g. --steps 50000 for smoke test)"
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU training even if GPU is available"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Make output directories
    os.makedirs(cfg.get("log_dir", "results/logs"), exist_ok=True)
    os.makedirs(cfg.get("checkpoint_dir", "results/checkpoints"), exist_ok=True)

    train(cfg, resume_path=args.resume, steps_override=args.steps,
          force_cpu=args.cpu)


if __name__ == "__main__":
    main()

"""
Fixed Backbone Training — Pong → Breakout Transfer
====================================================
Trains DQN on Breakout starting from a pre-trained DQN/Pong checkpoint,
with optional layer freezing. Two conditions:

  --freeze conv   Fixed backbone: conv layers loaded from Pong and frozen.
                  fc_repr + fc_out randomly initialised and trained from scratch.
                  Tests: do Pong's frozen conv features support learning Breakout?

  --freeze none   Full fine-tune: conv + fc_repr loaded from Pong, all layers train.
                  fc_out randomly initialised (Pong=6 actions, Breakout=4 — can't transfer).
                  Tests: does a Pong initialisation help when everything can adapt?

Compare both against the existing scratch baseline:
  results/checkpoints/dqn_breakout_seed42_scalemedium_lr0001_buf100k_step*.pt

Usage:
    # Frozen conv backbone (~9 hours on GPU):
    python experiments/train_backbone.py --freeze conv

    # Full fine-tune (~9 hours on GPU):
    python experiments/train_backbone.py --freeze none

    # Resume a run:
    python experiments/train_backbone.py --freeze conv --resume results/checkpoints/<name>.pt

    # Smoke test (50k steps):
    python experiments/train_backbone.py --freeze conv --steps 50000
"""

import os
import sys
import argparse
import random
import time

import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.wrappers import make_atari_env
from agents.dqn import DQNAgent
from utils.replay_buffer import ReplayBuffer
from utils.logger import Logger
from utils.checkpoint import save_checkpoint, load_checkpoint
from train import set_seeds, get_device


# ── Source checkpoint (DQN/Pong final) ───────────────────────────────────────
SOURCE_CHECKPOINT = (
    "results/checkpoints/"
    "dqn_pong_seed42_scalemedium_lr0001_buf100k_step02000000.pt"
)

# ── Breakout training config (matches run2 but with backbone init) ────────────
BASE_CFG = {
    "env_id":              "ALE/Breakout-v5",
    "algorithm":           "dqn",
    "total_steps":         5_000_000,
    "seed":                42,
    "net_scale":           "medium",
    "buffer_capacity":     100_000,
    "batch_size":          32,
    "learning_starts":     10_000,
    "lr":                  1e-4,
    "grad_clip":           10.0,
    "gamma":               0.99,
    "target_update_freq":  1_000,
    "epsilon_start":       1.0,
    "epsilon_end":         0.01,
    "epsilon_decay_steps": 100_000,
    "log_dir":             "results/logs",
    "checkpoint_dir":      "results/checkpoints",
    "checkpoint_freq":     500_000,
    "print_freq":          10,
}


def make_run_name(freeze: str) -> str:
    tag = "freeze_conv" if freeze == "conv" else "finetune_full"
    return f"dqn_breakout_{tag}_seed42_scalemedium_lr0001_buf100k"


def apply_backbone(agent: DQNAgent, source_path: str, freeze: str, device: torch.device):
    """
    Load weights from a Pong checkpoint into the Breakout agent and
    optionally freeze layers.

    freeze="conv":
        - Load conv weights from Pong (frozen — no gradient updates)
        - fc_repr + fc_out stay at random init and train freely
        Tests: can fixed Pong conv features support learning Breakout?

    freeze="none":
        - Load conv + fc_repr weights from Pong (all trainable)
        - fc_out stays random (Pong has 6 actions, Breakout has 4 — shape mismatch)
        Tests: does Pong initialisation help when everything can adapt?
    """
    print(f"\n[backbone] Loading source weights from: {os.path.basename(source_path)}")
    ckpt = torch.load(source_path, map_location=device, weights_only=True)
    source_state = ckpt["model_state"]

    current = agent.online_net.state_dict()

    # Always load conv weights from Pong (shape is independent of n_actions)
    conv_keys = [k for k in source_state if k.startswith("conv.")]
    for k in conv_keys:
        current[k] = source_state[k]
    print(f"[backbone] Loaded {len(conv_keys)} conv weight tensors from Pong")

    if freeze == "none":
        # Full fine-tune: also load fc_repr (fc_out can't transfer — shape mismatch)
        repr_keys = [k for k in source_state if k.startswith("fc_repr.")]
        for k in repr_keys:
            current[k] = source_state[k]
        print(f"[backbone] Loaded {len(repr_keys)} fc_repr weight tensors from Pong")
        print(f"[backbone] fc_out randomly initialised (Pong=6 actions ≠ Breakout=4 actions)")

    agent.online_net.load_state_dict(current)
    agent.target_net.load_state_dict(current)

    # ── Freeze specified layers ───────────────────────────────────────────────
    if freeze == "conv":
        frozen_count = 0
        for name, param in agent.online_net.named_parameters():
            if name.startswith("conv."):
                param.requires_grad = False
                frozen_count += param.numel()
        print(f"[backbone] Frozen conv layers — {frozen_count:,} params locked")

    # ── Rebuild optimizer with only trainable params ──────────────────────────
    trainable = [p for p in agent.online_net.parameters() if p.requires_grad]
    frozen    = [p for p in agent.online_net.parameters() if not p.requires_grad]
    agent.optimizer = optim.Adam(trainable, lr=BASE_CFG["lr"])

    print(f"[backbone] Trainable: {sum(p.numel() for p in trainable):,} params")
    print(f"[backbone] Frozen:    {sum(p.numel() for p in frozen):,} params")
    print(f"[backbone] Freeze mode: '{freeze}'\n")


def train(freeze: str, resume_path: str = None, steps_override: int = None):
    cfg      = BASE_CFG.copy()
    run_name = make_run_name(freeze)

    total_steps = steps_override or cfg["total_steps"]
    set_seeds(cfg["seed"])
    device = get_device()

    print(f"\n{'='*60}")
    print(f"  Fixed Backbone Training — {run_name}")
    print(f"  Freeze mode: {freeze}")
    print(f"  Steps:       {total_steps:,}")
    print(f"{'='*60}\n")

    env       = make_atari_env(cfg["env_id"], seed=cfg["seed"])
    n_actions = env.action_space.n

    agent = DQNAgent(
        n_actions            = n_actions,
        device               = device,
        lr                   = cfg["lr"],
        gamma                = cfg["gamma"],
        epsilon_start        = cfg["epsilon_start"],
        epsilon_end          = cfg["epsilon_end"],
        epsilon_decay_steps  = cfg["epsilon_decay_steps"],
        target_update_freq   = cfg["target_update_freq"],
        net_scale            = cfg["net_scale"],
        grad_clip            = cfg["grad_clip"],
    )

    # ── Apply backbone weights + freezing ─────────────────────────────────────
    apply_backbone(agent, SOURCE_CHECKPOINT, freeze, device)

    buffer = ReplayBuffer(
        capacity  = cfg["buffer_capacity"],
        obs_shape = (4, 84, 84),
        device    = device,
    )

    logger = Logger(
        log_dir  = cfg["log_dir"],
        run_name = run_name,
        use_tb   = True,
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step    = 0
    start_episode = 0
    if resume_path:
        state         = load_checkpoint(resume_path, agent.online_net,
                                        agent.target_net, agent.optimizer, device)
        start_step    = state["step"]
        start_episode = state["episode"]
        agent.epsilon = state["epsilon"]
        agent._step   = start_step
        # Re-apply freezing after loading (requires_grad gets reset by load_state_dict)
        if freeze == "conv":
            for name, param in agent.online_net.named_parameters():
                if name.startswith("conv."):
                    param.requires_grad = False
        print(f"[resume] Continuing from step {start_step:,}")

    # ── Training loop ─────────────────────────────────────────────────────────
    obs, _       = env.reset()
    ep_reward    = 0
    ep_length    = 0
    episode      = start_episode
    global_step  = start_step
    best_mean_r  = -float("inf")
    checkpoint_freq = cfg["checkpoint_freq"]

    print(f"[training] Filling buffer for {cfg['learning_starts']:,} steps ...\n")

    while global_step < total_steps:
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.push(obs, action, float(reward), next_obs, done)
        obs        = next_obs
        ep_reward += reward
        ep_length += 1
        global_step += 1

        if global_step >= cfg["learning_starts"] and len(buffer) >= cfg["batch_size"]:
            batch        = buffer.sample(cfg["batch_size"])
            loss, mean_q = agent.learn(batch)
            logger.log_step(loss, mean_q)

        if done:
            episode += 1
            mean_r = logger.log_episode(ep_reward, ep_length, agent.epsilon)
            if episode % cfg["print_freq"] == 0:
                logger.print_progress(ep_reward, agent.epsilon)
            if mean_r > best_mean_r:
                best_mean_r = mean_r
            obs, _ = env.reset()
            ep_reward = 0
            ep_length = 0

        if global_step % checkpoint_freq == 0:
            save_checkpoint(
                checkpoint_dir = cfg["checkpoint_dir"],
                run_name       = run_name,
                step           = global_step,
                model          = agent.online_net,
                target_model   = agent.target_net,
                optimizer      = agent.optimizer,
                episode        = episode,
                epsilon        = agent.epsilon,
                config         = cfg,
            )

    # Final save
    save_checkpoint(
        checkpoint_dir = cfg["checkpoint_dir"],
        run_name       = run_name,
        step           = global_step,
        model          = agent.online_net,
        target_model   = agent.target_net,
        optimizer      = agent.optimizer,
        episode        = episode,
        epsilon        = agent.epsilon,
        config         = cfg,
    )

    env.close()
    logger.close()

    print(f"\n{'='*60}")
    print(f"  Done: {run_name}")
    print(f"  Steps: {global_step:,} | Episodes: {episode}")
    print(f"  Best mean reward (10-ep): {best_mean_r:.2f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--freeze", required=True, choices=["conv", "none"],
        help="conv = freeze conv layers (fixed backbone) | none = full fine-tune"
    )
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--steps",  type=int, default=None, help="Override total_steps")
    parser.add_argument("--source", default=SOURCE_CHECKPOINT,
                        help="Source checkpoint to load Pong weights from")
    args = parser.parse_args()

    if args.source != SOURCE_CHECKPOINT:
        SOURCE_CHECKPOINT = args.source

    os.makedirs(BASE_CFG["log_dir"],        exist_ok=True)
    os.makedirs(BASE_CFG["checkpoint_dir"], exist_ok=True)

    train(freeze=args.freeze, resume_path=args.resume, steps_override=args.steps)

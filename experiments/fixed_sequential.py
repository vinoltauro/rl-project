"""
Fixed Sequential Training Experiment
======================================
Corrected version of the sequential forgetting experiment.

The original --freeze conv condition was broken: fc_repr was randomly
reinitialised, so the Pong chimera evaluation started at -21 before any
Breakout training — making the comparison meaningless.

This script fixes the design with two proper conditions:

  --freeze all
      Load conv + fc_repr from Pong, BOTH frozen.
      Add a new fc_out_breakout head (4 actions), train ONLY this.
      Maximum protection: nothing in the Pong pathway changes.
      Pong evaluation: frozen_conv + frozen_fc_repr + original_Pong_fc_out
      Expected: Pong stays high, Breakout learns slowly (only output head trains).

  --freeze conv
      Load conv (frozen) + fc_repr (trainable) from Pong.
      Add a new fc_out_breakout head, train fc_repr + fc_out_breakout.
      Partial protection: conv is locked, fc_repr can adapt.
      Pong evaluation: frozen_conv + adapted_fc_repr + original_Pong_fc_out
      Expected: Pong degrades as fc_repr adapts to Breakout.

Together with the already-complete --freeze none condition (full sequential),
this gives a clean spectrum:
    freeze all → freeze conv → no freeze
    (max protection → partial → none)

Metrics logged every 200k steps:
    1. Pong reward    (chimera: current backbone + original Pong fc_out)
    2. Dead neurons   (fc_repr activation on 1000 Breakout states)
    3. CKA drift      (vs original Pong fc_repr representations)

Usage:
    python experiments/fixed_sequential.py --freeze all
    python experiments/fixed_sequential.py --freeze conv
    python experiments/fixed_sequential.py --freeze all --steps 50000  # smoke test
"""

import os
import sys
import csv
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn import AtariCNN
from envs.wrappers import make_atari_env
from agents.dqn import DQNAgent
from utils.replay_buffer import ReplayBuffer
from utils.logger import Logger
from utils.checkpoint import save_checkpoint
from train import set_seeds, get_device


SOURCE_CHECKPOINT = (
    "results/checkpoints/"
    "dqn_pong_seed42_scalemedium_lr0001_buf100k_step02000000.pt"
)

CFG = {
    "env_id":              "ALE/Breakout-v5",
    "total_steps":         2_000_000,
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
    "eval_freq":           200_000,
    "eval_pong_episodes":  20,
    "dead_neuron_states":  1_000,
    "cka_states":          1_000,
    "print_freq":          10,
}

PONG_N_ACTIONS     = 6
BREAKOUT_N_ACTIONS = 4


# ── CKA ───────────────────────────────────────────────────────────────────────
def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    def centre(K):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    Kc = centre(X @ X.T)
    Lc = centre(Y @ Y.T)
    num   = np.sum(Kc * Lc)
    denom = np.sqrt(np.sum(Kc * Kc) * np.sum(Lc * Lc))
    return float(num / denom) if denom > 1e-10 else 0.0


# ── Representation helpers ────────────────────────────────────────────────────
def collect_representations(model, env_id, n_states, device, seed=99):
    reps = []
    handle = model.fc_repr.register_forward_hook(
        lambda m, i, o: reps.append(o.detach().cpu().numpy())
    )
    env = make_atari_env(env_id, seed=seed)
    obs, _ = env.reset()
    model.eval()
    with torch.no_grad():
        for _ in range(n_states):
            state_t = torch.from_numpy(obs).float().div(255.0).unsqueeze(0).to(device)
            _ = model(state_t)
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                obs, _ = env.reset()
    handle.remove()
    env.close()
    return np.vstack(reps)


def measure_dead_neurons(model, env_id, n_states, device):
    reps    = collect_representations(model, env_id, n_states, device, seed=77)
    active  = (reps > 0).mean(axis=0)
    return float((active < 0.05).mean())


def evaluate_pong(current_model, pong_fc_out_state, device, n_episodes):
    eval_model = AtariCNN(n_actions=PONG_N_ACTIONS, net_scale="medium").to(device)
    eval_model.eval()
    current = current_model.state_dict()
    eval_s  = eval_model.state_dict()
    for k, v in current.items():
        if k.startswith("conv.") or k.startswith("fc_repr."):
            eval_s[k] = v
    for k, v in pong_fc_out_state.items():
        if k.startswith("fc_out."):
            eval_s[k] = v
    eval_model.load_state_dict(eval_s)

    env     = make_atari_env("ALE/Pong-v5", seed=55)
    rewards = []
    with torch.no_grad():
        for _ in range(n_episodes):
            obs, _ = env.reset()
            ep_r   = 0.0
            done   = False
            while not done:
                state_t = torch.from_numpy(obs).float().div(255.0).unsqueeze(0).to(device)
                action  = int(eval_model(state_t).argmax(1).item())
                action  = min(action, PONG_N_ACTIONS - 1)
                obs, r, terminated, truncated, _ = env.step(action)
                ep_r += r
                done  = terminated or truncated
            rewards.append(ep_r)
    env.close()
    return float(np.mean(rewards))


# ── Backbone setup ────────────────────────────────────────────────────────────
def apply_backbone(agent, source_path, freeze, device):
    print(f"\n[backbone] Loading from {os.path.basename(source_path)}")
    ckpt         = torch.load(source_path, map_location=device, weights_only=True)
    source_state = ckpt["model_state"]
    current      = agent.online_net.state_dict()

    # Always load conv from Pong
    for k, v in source_state.items():
        if k.startswith("conv."):
            current[k] = v

    # Always load fc_repr from Pong (key fix vs original design)
    for k, v in source_state.items():
        if k.startswith("fc_repr."):
            current[k] = v

    agent.online_net.load_state_dict(current)
    agent.target_net.load_state_dict(current)

    # Freeze layers
    frozen_count = 0
    for name, param in agent.online_net.named_parameters():
        if name.startswith("conv."):
            param.requires_grad = False
            frozen_count += param.numel()
        elif freeze == "all" and name.startswith("fc_repr."):
            param.requires_grad = False
            frozen_count += param.numel()

    trainable = [p for p in agent.online_net.parameters() if p.requires_grad]
    agent.optimizer = optim.Adam(trainable, lr=CFG["lr"])

    print(f"[backbone] Frozen:    {frozen_count:,} params")
    print(f"[backbone] Trainable: {sum(p.numel() for p in trainable):,} params")
    print(f"[backbone] Freeze mode: '{freeze}'\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def train(freeze: str, steps_override: int = None):
    total_steps = steps_override or CFG["total_steps"]
    tag         = "freeze_all" if freeze == "all" else "freeze_conv_fixed"
    run_name    = f"dqn_fixed_sequential_{tag}_seed42_scalemedium"

    set_seeds(CFG["seed"])
    device = get_device()

    print(f"\n{'='*60}")
    print(f"  Fixed Sequential Experiment — condition: {freeze}")
    print(f"  Steps: {total_steps:,}")
    print(f"{'='*60}\n")

    # Load original Pong reference
    pong_ckpt        = torch.load(SOURCE_CHECKPOINT, map_location=device, weights_only=True)
    pong_state       = pong_ckpt["model_state"]
    pong_fc_out_state = {k: v for k, v in pong_state.items() if k.startswith("fc_out.")}

    ref_model = AtariCNN(n_actions=PONG_N_ACTIONS, net_scale="medium").to(device)
    ref_model.load_state_dict(pong_state)
    ref_model.eval()
    print("[setup] Collecting reference Pong representations...")
    ref_reps = collect_representations(ref_model, "ALE/Pong-v5",
                                        CFG["cka_states"], device, seed=42)

    # Build Breakout agent
    env       = make_atari_env(CFG["env_id"], seed=CFG["seed"])
    n_actions = env.action_space.n

    agent = DQNAgent(
        n_actions           = n_actions,
        device              = device,
        lr                  = CFG["lr"],
        gamma               = CFG["gamma"],
        epsilon_start       = CFG["epsilon_start"],
        epsilon_end         = CFG["epsilon_end"],
        epsilon_decay_steps = CFG["epsilon_decay_steps"],
        target_update_freq  = CFG["target_update_freq"],
        net_scale           = CFG["net_scale"],
        grad_clip           = CFG["grad_clip"],
    )
    apply_backbone(agent, SOURCE_CHECKPOINT, freeze, device)

    buffer = ReplayBuffer(capacity=CFG["buffer_capacity"],
                          obs_shape=(4, 84, 84), device=device)
    logger = Logger(log_dir=CFG["log_dir"], run_name=run_name, use_tb=False)

    # Metrics CSV
    os.makedirs("results/logs", exist_ok=True)
    metrics_path = f"results/logs/{run_name}_forgetting.csv"
    mf           = open(metrics_path, "w", newline="")
    mw           = csv.writer(mf)
    mw.writerow(["breakout_step", "pong_reward", "dead_neurons", "cka_drift"])
    mf.flush()

    def evaluate_and_log(step):
        pong_r = evaluate_pong(agent.online_net, pong_fc_out_state,
                               device, CFG["eval_pong_episodes"])
        dead   = measure_dead_neurons(agent.online_net, CFG["env_id"],
                                      CFG["dead_neuron_states"], device)
        reps   = collect_representations(agent.online_net, "ALE/Pong-v5",
                                          CFG["cka_states"], device, seed=42)
        cka    = linear_cka(ref_reps, reps)
        print(f"[eval] step={step:,} | pong_reward={pong_r:.2f} | "
              f"dead={dead:.3f} | cka={cka:.3f}")
        mw.writerow([step, pong_r, dead, cka])
        mf.flush()

    # Baseline at step 0
    print("[eval] Baseline (step 0)...")
    evaluate_and_log(0)

    # Training loop
    obs, _      = env.reset()
    ep_reward   = 0
    ep_length   = 0
    episode     = 0
    global_step = 0
    next_eval   = CFG["eval_freq"]
    next_ckpt   = CFG["checkpoint_freq"]

    print(f"\n[training] Starting ({total_steps:,} steps)...\n")

    while global_step < total_steps:
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.push(obs, action, float(reward), next_obs, done)
        obs        = next_obs
        ep_reward += reward
        ep_length += 1
        global_step += 1

        if global_step >= CFG["learning_starts"] and len(buffer) >= CFG["batch_size"]:
            batch = buffer.sample(CFG["batch_size"])
            loss, mean_q = agent.learn(batch)
            logger.log_step(loss, mean_q)

        if done:
            episode += 1
            logger.log_episode(ep_reward, ep_length, agent.epsilon)
            if episode % CFG["print_freq"] == 0:
                logger.print_progress(ep_reward, agent.epsilon)
            obs, _ = env.reset()
            ep_reward = 0
            ep_length = 0

        if global_step >= next_eval:
            print(f"\n[eval] Measuring at step {global_step:,}...")
            evaluate_and_log(global_step)
            next_eval += CFG["eval_freq"]

        if global_step >= next_ckpt:
            save_checkpoint(
                checkpoint_dir = CFG["checkpoint_dir"],
                run_name       = run_name,
                step           = global_step,
                model          = agent.online_net,
                target_model   = agent.target_net,
                optimizer      = agent.optimizer,
                episode        = episode,
                epsilon        = agent.epsilon,
                config         = CFG,
            )
            next_ckpt += CFG["checkpoint_freq"]

    print(f"\n[eval] Final evaluation...")
    evaluate_and_log(global_step)
    mf.close()
    env.close()
    logger.close()

    print(f"\n{'='*60}")
    print(f"  Done: {run_name}")
    print(f"  Metrics → {metrics_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True, choices=["all", "conv"],
                        help="all = freeze conv+fc_repr | conv = freeze conv only")
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(CFG["log_dir"],        exist_ok=True)
    os.makedirs(CFG["checkpoint_dir"], exist_ok=True)

    train(freeze=args.freeze, steps_override=args.steps)

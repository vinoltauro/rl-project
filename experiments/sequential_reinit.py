"""
Sequential Training with Dead Neuron Reinitialisation
=====================================================

Implements the experiment requested by the professor:

  1. Load the fully-trained Pong DQN checkpoint (2M steps)
  2. Measure which fc_repr neurons are dead (post-ReLU activation ≈ 0)
  3. Reinitialise the weights for those dead neurons (orthogonal init)
  4. Copy the modified backbone (conv + fc_repr) into a fresh Breakout agent
  5. Train on Breakout for 2M steps — same conditions as the sequential
     no-freeze baseline (experiments/sequential_training.py)
  6. Every 200k steps, log: pong_reward (chimera eval), dead_neurons, cka_drift

Hypothesis: dead-neuron reinitialisation at the task boundary restores network
plasticity so the model adapts faster to Breakout, while potentially reducing
catastrophic forgetting (if the dead neurons are the ones that encoded
Pong-specific features).

Compare against: experiments/sequential_training.py (no reinit baseline)

Results isolated to results/sequential_reinit/ — nothing in results/ is overwritten.

Usage:
    python experiments/sequential_reinit.py
    python experiments/sequential_reinit.py --total_steps 50000  # smoke test
"""

import os
import sys
import csv
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn import AtariCNN
from agents.dqn import DQNAgent
from envs.wrappers import make_atari_env
from utils.replay_buffer import ReplayBuffer
from utils.logger import Logger
from utils.checkpoint import save_checkpoint
from train import set_seeds, get_device


# ── Paths ─────────────────────────────────────────────────────────────────────
SOURCE_CHECKPOINT = (
    "results/checkpoints/"
    "dqn_pong_seed42_scalemedium_lr0001_buf100k_step02000000.pt"
)
RESULTS_DIR = "results/sequential_reinit"

# ── Config ────────────────────────────────────────────────────────────────────
CFG = {
    "total_steps":           2_000_000,
    "seed":                  42,
    "net_scale":             "medium",
    "n_pong_actions":        6,
    "n_breakout_actions":    4,
    "buffer_capacity":       100_000,
    "batch_size":            32,
    "learning_starts":       10_000,
    "lr":                    1e-4,
    "grad_clip":             10.0,
    "gamma":                 0.99,
    "target_update_freq":    1_000,
    "epsilon_start":         1.0,
    "epsilon_end":           0.01,
    "epsilon_decay_steps":   200_000,
    "eval_freq":             200_000,
    "checkpoint_freq":       500_000,
    "dead_thresh":           0.95,
    "dead_neuron_frames":    1_000,
    "chimera_episodes":      20,
    "cka_frames":            1_000,
    "print_freq":            100,
}


# ── Dead neuron measurement ───────────────────────────────────────────────────
def measure_dead_neurons(model: AtariCNN, env_id: str, n_frames: int,
                          device: torch.device, seed: int = 42) -> np.ndarray:
    """
    Run n_frames forward passes on random-action trajectories.
    Return a boolean mask (True = dead) over the 512 fc_repr neurons.
    Deadness criterion: post-ReLU activation is zero in >95% of frames.
    """
    env = make_atari_env(env_id, seed=seed)
    obs, _ = env.reset()
    reps   = []
    model.eval()
    with torch.no_grad():
        for _ in range(n_frames):
            s_t = (torch.from_numpy(obs).float().div(255.0)
                   .unsqueeze(0).to(device))
            _   = model(s_t)
            reps.append(model.representation.squeeze(0).cpu().numpy())
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                obs, _ = env.reset()
    env.close()
    reps     = np.stack(reps)                           # (n_frames, 512)
    zero_frac = (reps == 0.0).mean(axis=0)             # (512,)
    return zero_frac > CFG["dead_thresh"]


def reinit_dead_neurons(model: AtariCNN, dead_mask: np.ndarray) -> int:
    """
    Reinitialise weights and biases of dead neurons in fc_repr using
    orthogonal initialisation (same scheme used at training start).

    fc_repr in AtariCNN is nn.Sequential(nn.Linear(...), nn.ReLU()).
    The nn.Linear is at index 0.
    Optimizer must be recreated after this call to clear stale Adam moments.
    """
    linear      = model.fc_repr[0]                     # nn.Linear(3136, 512)
    dead_indices = np.where(dead_mask)[0]
    n_dead       = len(dead_indices)
    if n_dead == 0:
        print("[reinit] No dead neurons found — nothing to reinitialise.")
        return 0

    with torch.no_grad():
        # Reinit each dead row of the weight matrix independently
        for i in dead_indices:
            row = linear.weight[i].unsqueeze(0)       # (1, in_features)
            nn.init.orthogonal_(row, gain=np.sqrt(2))
            linear.weight[i] = row.squeeze(0)
        nn.init.constant_(linear.bias[dead_indices], 0.0)

    print(f"[reinit] Reinitialised {n_dead}/{linear.weight.shape[0]} "
          f"dead neurons ({100 * n_dead / linear.weight.shape[0]:.1f}%)")
    return n_dead


# ── CKA helpers ───────────────────────────────────────────────────────────────
def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X, Y = X.astype(np.float64), Y.astype(np.float64)
    def centre(K):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H
    Kc = centre(X @ X.T)
    Lc = centre(Y @ Y.T)
    num   = np.sum(Kc * Lc)
    denom = np.sqrt(np.sum(Kc * Kc) * np.sum(Lc * Lc))
    return float(num / denom) if denom > 1e-10 else 0.0


def collect_repr(model: AtariCNN, env_id: str, n_frames: int,
                  device: torch.device, seed: int = 99) -> np.ndarray:
    env = make_atari_env(env_id, seed=seed)
    obs, _ = env.reset()
    reps   = []
    model.eval()
    with torch.no_grad():
        for _ in range(n_frames):
            s_t = (torch.from_numpy(obs).float().div(255.0)
                   .unsqueeze(0).to(device))
            _   = model(s_t)
            reps.append(model.representation.squeeze(0).cpu().numpy())
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                obs, _ = env.reset()
    env.close()
    return np.stack(reps)


# ── Chimera Pong eval ─────────────────────────────────────────────────────────
def chimera_pong_eval(breakout_agent: DQNAgent, pong_head_state: dict,
                       n_episodes: int, device: torch.device) -> float:
    """
    Evaluate Pong retention by combining the current backbone with the
    original Pong fc_out head. This isolates backbone degradation from
    head degradation.

    1. Copy current conv + fc_repr into a fresh AtariCNN (6 Pong actions)
    2. Load original Pong fc_out weights
    3. Evaluate for n_episodes — no gradient, ε=0.0
    """
    chimera = AtariCNN(n_actions=6, net_scale=CFG["net_scale"]).to(device)

    # Copy backbone from current Breakout agent
    current_sd = breakout_agent.online_net.state_dict()
    chimera_sd = chimera.state_dict()
    for key in ["conv.0.weight", "conv.0.bias",
                 "conv.2.weight", "conv.2.bias",
                 "conv.4.weight", "conv.4.bias",
                 "fc_repr.0.weight", "fc_repr.0.bias"]:
        chimera_sd[key] = current_sd[key].clone()
    chimera.load_state_dict(chimera_sd, strict=False)

    # Restore original Pong fc_out
    chimera.fc_out.weight.data.copy_(pong_head_state["fc_out.weight"])
    chimera.fc_out.bias.data.copy_(pong_head_state["fc_out.bias"])
    chimera.eval()

    env      = make_atari_env("ALE/Pong-v5", seed=99)
    rewards  = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_r   = 0.0
        done   = False
        while not done:
            s_t = (torch.from_numpy(obs).float().div(255.0)
                   .unsqueeze(0).to(device))
            with torch.no_grad():
                action = int(chimera(s_t).argmax(1).item())
            obs, r, term, trunc, _ = env.step(action)
            ep_r += r
            done  = term or trunc
        rewards.append(ep_r)
    env.close()
    return float(np.mean(rewards))


# ── Main ──────────────────────────────────────────────────────────────────────
def train(total_steps: int):
    run_name = "dqn_sequential_reinit_seed42_scalemedium"
    set_seeds(CFG["seed"])
    device = get_device()

    ckpt_dir = os.path.join(RESULTS_DIR, "checkpoints")
    log_dir  = os.path.join(RESULTS_DIR, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Sequential + Dead Neuron Reinit")
    print(f"  Breakout training: {total_steps:,} steps")
    print(f"{'='*60}\n")

    # ── Load Pong checkpoint ──────────────────────────────────────────────────
    print(f"[setup] Loading Pong checkpoint: {SOURCE_CHECKPOINT}")
    pong_ckpt  = torch.load(SOURCE_CHECKPOINT, map_location=device, weights_only=True)
    pong_model = AtariCNN(n_actions=CFG["n_pong_actions"],
                           net_scale=CFG["net_scale"]).to(device)
    pong_model.load_state_dict(pong_ckpt["model_state"])
    pong_model.eval()

    pong_head_state = {
        "fc_out.weight": pong_model.fc_out.weight.data.clone(),
        "fc_out.bias":   pong_model.fc_out.bias.data.clone(),
    }

    # ── CKA reference reps ────────────────────────────────────────────────────
    print("[setup] Collecting reference representations for CKA...")
    ref_reps = collect_repr(pong_model, "ALE/Pong-v5",
                             CFG["cka_frames"], device, seed=42)
    print(f"[setup] Reference reps: {ref_reps.shape}")

    # ── Measure dead neurons ──────────────────────────────────────────────────
    print("\n[dead neurons] Measuring dead neurons in trained Pong fc_repr...")
    dead_mask = measure_dead_neurons(
        pong_model, "ALE/Pong-v5", CFG["dead_neuron_frames"], device, seed=42
    )
    n_dead    = int(dead_mask.sum())
    dead_frac = n_dead / len(dead_mask)
    print(f"[dead neurons] Dead: {n_dead}/{len(dead_mask)} "
          f"({100*dead_frac:.1f}%)\n")

    # ── Reinitialise dead neurons ─────────────────────────────────────────────
    print("[reinit] Applying orthogonal reinit to dead neurons...")
    n_reinited = reinit_dead_neurons(pong_model, dead_mask)

    # ── Build Breakout agent with reinited backbone ───────────────────────────
    breakout_agent = DQNAgent(
        n_actions           = CFG["n_breakout_actions"],
        net_scale           = CFG["net_scale"],
        lr                  = CFG["lr"],
        gamma               = CFG["gamma"],
        epsilon_start       = CFG["epsilon_start"],
        epsilon_end         = CFG["epsilon_end"],
        epsilon_decay_steps = CFG["epsilon_decay_steps"],
        target_update_freq  = CFG["target_update_freq"],
        grad_clip           = CFG["grad_clip"],
        device              = device,
    )
    buffer = ReplayBuffer(capacity=CFG["buffer_capacity"],
                          obs_shape=(4, 84, 84), device=device)

    # Copy reinited backbone (conv + fc_repr) into Breakout agent
    src = pong_model.state_dict()
    dst = breakout_agent.online_net.state_dict()
    for key in ["conv.0.weight", "conv.0.bias",
                 "conv.2.weight", "conv.2.bias",
                 "conv.4.weight", "conv.4.bias",
                 "fc_repr.0.weight", "fc_repr.0.bias"]:
        dst[key] = src[key].clone()
    breakout_agent.online_net.load_state_dict(dst, strict=False)
    breakout_agent.target_net.load_state_dict(
        breakout_agent.online_net.state_dict()
    )

    # Recreate optimizer — clear stale Adam moments from the reinited weights
    breakout_agent.optimizer = optim.Adam(
        breakout_agent.online_net.parameters(), lr=CFG["lr"]
    )
    print("[setup] Backbone copied, optimizer recreated\n")

    # ── Logger and CSV ────────────────────────────────────────────────────────
    logger       = Logger(log_dir=log_dir, run_name=run_name, use_tb=False)
    metrics_path = os.path.join(log_dir, f"{run_name}_forgetting.csv")
    mf  = open(metrics_path, "w", newline="")
    mw  = csv.writer(mf)
    mw.writerow(["breakout_step", "pong_reward", "dead_neurons",
                 "cka_drift", "breakout_reward_mean10"])
    mf.flush()

    # ── Training state ────────────────────────────────────────────────────────
    env = make_atari_env("ALE/Breakout-v5", seed=CFG["seed"])
    obs, _       = env.reset()
    recent_r     = []
    ep_reward    = 0.0
    ep_len       = 0
    ep_count     = 0
    next_eval    = 0
    next_ckpt    = CFG["checkpoint_freq"]

    def measure_metrics(step: int):
        """Evaluate and log all forgetting metrics."""
        pong_r = chimera_pong_eval(breakout_agent, pong_head_state,
                                    CFG["chimera_episodes"], device)
        curr_reps = collect_repr(breakout_agent.online_net, "ALE/Pong-v5",
                                  CFG["cka_frames"], device, seed=99)
        cka  = linear_cka(ref_reps, curr_reps)
        dead_m = measure_dead_neurons(breakout_agent.online_net,
                                       "ALE/Breakout-v5",
                                       CFG["dead_neuron_frames"], device)
        dead = float(dead_m.mean())
        br   = float(np.mean(recent_r[-10:])) if recent_r else 0.0

        print(f"\n[eval] step={step:,} | "
              f"pong_r(chimera)={pong_r:.2f} | "
              f"breakout_r={br:.2f} | "
              f"dead={dead:.3f} | cka={cka:.3f}")
        mw.writerow([step, pong_r, dead, cka, br])
        mf.flush()

    # ── Baseline measurement (before any Breakout training) ───────────────────
    print("[eval] Baseline measurement (step 0)...")
    measure_metrics(0)
    print(f"\n[training] Starting Breakout training ({total_steps:,} steps)...\n")
    next_eval = CFG["eval_freq"]

    # ── Main Breakout training loop ───────────────────────────────────────────
    for step in range(1, total_steps + 1):
        action          = breakout_agent.select_action(obs)
        next_obs, reward, term, trunc, _ = env.step(action)
        done            = term or trunc

        buffer.push(obs, action, float(reward), next_obs, done)
        obs       = next_obs if not done else env.reset()[0]
        ep_reward += reward
        ep_len    += 1

        if (step >= CFG["learning_starts"] and
                len(buffer) >= CFG["batch_size"]):
            batch        = buffer.sample(CFG["batch_size"])
            loss, mean_q = breakout_agent.learn(batch)
            logger.log_step(loss, mean_q)

        if done:
            ep_count += 1
            logger.log_episode(ep_reward, ep_len, breakout_agent.epsilon)
            recent_r.append(ep_reward)
            if ep_count % CFG["print_freq"] == 0:
                br10 = float(np.mean(recent_r[-10:])) if recent_r else 0.0
                print(f"[ep {ep_count}] step={step:,} | "
                      f"r={ep_reward:.1f} | mean10={br10:.1f} | "
                      f"ε={breakout_agent.epsilon:.3f}")
            ep_reward = 0.0
            ep_len    = 0

        if step >= next_eval:
            measure_metrics(step)
            next_eval += CFG["eval_freq"]

        if step >= next_ckpt:
            save_checkpoint(
                checkpoint_dir = ckpt_dir,
                run_name       = run_name,
                step           = step,
                model          = breakout_agent.online_net,
                target_model   = breakout_agent.target_net,
                optimizer      = breakout_agent.optimizer,
                episode        = ep_count,
                epsilon        = breakout_agent.epsilon,
                config         = CFG,
            )
            next_ckpt += CFG["checkpoint_freq"]

    # ── Final eval ────────────────────────────────────────────────────────────
    print(f"\n[eval] Final ({total_steps:,} steps)...")
    measure_metrics(total_steps)
    mf.close()
    env.close()
    logger.close()

    print(f"\n{'='*60}")
    print(f"  Done: {run_name}")
    print(f"  Dead neurons before: {100*dead_frac:.1f}%")
    print(f"  Neurons reinited:    {n_reinited}")
    print(f"  Metrics → {metrics_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_steps", type=int, default=CFG["total_steps"])
    args = parser.parse_args()

    train(total_steps=args.total_steps)

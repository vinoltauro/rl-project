"""
Sequential Training Experiment
================================
Train DQN on Pong to convergence, then continue training on Breakout.
Measures three things at every 200k Breakout steps:

  1. Pong reward     — chimera network (current conv+fc_repr + original Pong
                       fc_out head) evaluated for 20 episodes on Pong.
                       Directly measures how much Pong skill has been forgotten.

  2. Dead neurons    — fraction of fc_repr neurons inactive on 1000 Breakout
                       states. Tracks capacity loss as the mechanism of forgetting.

  3. CKA drift       — CKA between current fc_repr activations and original
                       Pong fc_repr activations, computed on 1000 fixed Pong
                       states. Measures representational drift from the Pong
                       starting point.

Two conditions:
  --freeze none   All layers adapt to Breakout  (forgetting baseline)
  --freeze conv   Conv layers frozen             (mitigation condition)

Usage:
    python experiments/sequential_training.py --freeze none
    python experiments/sequential_training.py --freeze conv
    python experiments/sequential_training.py --freeze none --steps 2000000
"""

import os
import sys
import csv
import argparse
import random
import time

import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn import AtariCNN
from envs.wrappers import make_atari_env
from agents.dqn import DQNAgent
from utils.replay_buffer import ReplayBuffer
from utils.logger import Logger
from utils.checkpoint import save_checkpoint, load_checkpoint
from train import set_seeds, get_device


# ── Source checkpoint ─────────────────────────────────────────────────────────
SOURCE_CHECKPOINT = (
    "results/checkpoints/"
    "dqn_pong_seed42_scalemedium_lr0001_buf100k_step02000000.pt"
)

# ── Config ────────────────────────────────────────────────────────────────────
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
    "checkpoint_dir":      "results/checkpoints",
    "log_dir":             "results/logs",
    "checkpoint_freq":     500_000,
    "eval_freq":           200_000,   # how often to measure forgetting
    "eval_pong_episodes":  20,
    "dead_neuron_states":  1_000,
    "cka_states":          1_000,
    "print_freq":          10,
}

PONG_N_ACTIONS    = 6
BREAKOUT_N_ACTIONS = 4


# ── CKA ───────────────────────────────────────────────────────────────────────
def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Centred Kernel Alignment between two activation matrices (n x d)."""
    def centre(K):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    K = X @ X.T
    L = Y @ Y.T
    Kc = centre(K)
    Lc = centre(L)
    num   = np.sum(Kc * Lc)
    denom = np.sqrt(np.sum(Kc * Kc) * np.sum(Lc * Lc))
    return float(num / denom) if denom > 0 else 0.0


# ── Representation extraction ─────────────────────────────────────────────────
def collect_representations(model: AtariCNN, env_id: str, n_states: int,
                             device: torch.device, seed: int = 99) -> np.ndarray:
    """
    Run the model on env_id and collect n_states fc_repr activations.
    Uses a forward hook — works regardless of what fc_out head is attached.
    """
    reps = []
    hook_handle = None

    def hook(module, inp, out):
        reps.append(out.detach().cpu().numpy())

    hook_handle = model.fc_repr.register_forward_hook(hook)

    env  = make_atari_env(env_id, seed=seed)
    obs, _ = env.reset()
    collected = 0

    model.eval()
    with torch.no_grad():
        while collected < n_states:
            state_t = torch.from_numpy(obs).float().div(255.0).unsqueeze(0).to(device)
            _ = model(state_t)
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                obs, _ = env.reset()
            collected += 1

    hook_handle.remove()
    env.close()
    return np.vstack(reps)   # (n_states, 512)


# ── Dead neuron measurement ───────────────────────────────────────────────────
def measure_dead_neurons(model: AtariCNN, env_id: str, n_states: int,
                          device: torch.device) -> float:
    """Fraction of fc_repr neurons inactive across n_states game states."""
    reps = collect_representations(model, env_id, n_states, device, seed=77)
    active = (reps > 0).mean(axis=0)          # fraction of states each neuron fires
    dead   = (active < 0.05).mean()           # neurons firing < 5% of the time
    return float(dead)


# ── Pong forgetting evaluation ────────────────────────────────────────────────
def evaluate_pong_forgetting(current_model: AtariCNN,
                              pong_fc_out_state: dict,
                              device: torch.device,
                              n_episodes: int) -> float:
    """
    Chimera evaluation: copy current conv+fc_repr into a fresh Pong network,
    attach the original Pong fc_out, and evaluate on Pong.

    Measures: do the current features still support Pong decision-making?
    """
    eval_model = AtariCNN(n_actions=PONG_N_ACTIONS, net_scale="medium").to(device)
    eval_model.eval()

    # Copy current conv + fc_repr weights
    current_state = current_model.state_dict()
    eval_state    = eval_model.state_dict()

    for k, v in current_state.items():
        if k.startswith("conv.") or k.startswith("fc_repr."):
            eval_state[k] = v

    # Restore original Pong fc_out
    for k, v in pong_fc_out_state.items():
        if k.startswith("fc_out."):
            eval_state[k] = v

    eval_model.load_state_dict(eval_state)

    env = make_atari_env("ALE/Pong-v5", seed=55)
    rewards = []

    with torch.no_grad():
        for _ in range(n_episodes):
            obs, _ = env.reset()
            ep_r   = 0.0
            done   = False
            while not done:
                state_t = torch.from_numpy(obs).float().div(255.0).unsqueeze(0).to(device)
                action  = int(eval_model(state_t).argmax(dim=1).item())
                action  = min(action, PONG_N_ACTIONS - 1)
                obs, r, terminated, truncated, _ = env.step(action)
                ep_r += r
                done  = terminated or truncated
            rewards.append(ep_r)

    env.close()
    return float(np.mean(rewards))


# ── Apply backbone ────────────────────────────────────────────────────────────
def apply_backbone(agent: DQNAgent, source_path: str, freeze: str,
                   device: torch.device):
    print(f"\n[backbone] Loading from: {os.path.basename(source_path)}")
    ckpt         = torch.load(source_path, map_location=device, weights_only=True)
    source_state = ckpt["model_state"]
    current      = agent.online_net.state_dict()

    conv_keys = [k for k in source_state if k.startswith("conv.")]
    for k in conv_keys:
        current[k] = source_state[k]
    print(f"[backbone] Loaded {len(conv_keys)} conv tensors from Pong")

    if freeze == "none":
        repr_keys = [k for k in source_state if k.startswith("fc_repr.")]
        for k in repr_keys:
            current[k] = source_state[k]
        print(f"[backbone] Loaded fc_repr from Pong (full fine-tune)")

    agent.online_net.load_state_dict(current)
    agent.target_net.load_state_dict(current)

    if freeze == "conv":
        frozen = 0
        for name, param in agent.online_net.named_parameters():
            if name.startswith("conv."):
                param.requires_grad = False
                frozen += param.numel()
        print(f"[backbone] Frozen {frozen:,} conv params")

    trainable = [p for p in agent.online_net.parameters() if p.requires_grad]
    agent.optimizer = optim.Adam(trainable, lr=CFG["lr"])
    print(f"[backbone] Trainable: {sum(p.numel() for p in trainable):,} params\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def train(freeze: str, steps_override: int = None):
    total_steps = steps_override or CFG["total_steps"]
    tag         = "freeze_conv" if freeze == "conv" else "sequential_full"
    run_name    = f"dqn_sequential_{tag}_seed42_scalemedium"

    set_seeds(CFG["seed"])
    device = get_device()

    print(f"\n{'='*60}")
    print(f"  Sequential Training Experiment")
    print(f"  Condition: {freeze}  |  Steps: {total_steps:,}")
    print(f"{'='*60}\n")

    # ── Load original Pong checkpoint for reference ───────────────────────────
    print("[setup] Loading original Pong checkpoint for reference...")
    pong_ckpt        = torch.load(SOURCE_CHECKPOINT, map_location=device,
                                  weights_only=True)
    pong_model_state = pong_ckpt["model_state"]

    # Extract original Pong fc_out weights (needed for chimera Pong eval)
    pong_fc_out_state = {k: v for k, v in pong_model_state.items()
                         if k.startswith("fc_out.")}

    # Build reference Pong model for initial CKA baseline
    ref_model = AtariCNN(n_actions=PONG_N_ACTIONS, net_scale="medium").to(device)
    ref_model.load_state_dict(pong_model_state)
    ref_model.eval()

    print("[setup] Collecting reference Pong representations for CKA...")
    ref_reps = collect_representations(ref_model, "ALE/Pong-v5",
                                        CFG["cka_states"], device, seed=42)
    print(f"[setup] Reference reps shape: {ref_reps.shape}\n")

    # ── Build Breakout agent ──────────────────────────────────────────────────
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

    # ── CSV for forgetting metrics ────────────────────────────────────────────
    os.makedirs("results/logs", exist_ok=True)
    metrics_path = f"results/logs/{run_name}_forgetting.csv"
    metrics_file = open(metrics_path, "w", newline="")
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow([
        "breakout_step", "pong_reward", "dead_neurons", "cka_drift"
    ])
    metrics_file.flush()
    print(f"[metrics] Saving forgetting metrics → {metrics_path}\n")

    # ── Log baseline (before any Breakout training) ───────────────────────────
    print("[eval] Baseline evaluation (step 0)...")
    pong_r_0  = evaluate_pong_forgetting(agent.online_net, pong_fc_out_state,
                                          device, CFG["eval_pong_episodes"])
    dead_0    = measure_dead_neurons(agent.online_net, CFG["env_id"],
                                      CFG["dead_neuron_states"], device)
    cka_reps  = collect_representations(agent.online_net, "ALE/Pong-v5",
                                         CFG["cka_states"], device, seed=42)
    cka_0     = linear_cka(ref_reps, cka_reps)
    print(f"[eval] step=0 | pong_reward={pong_r_0:.2f} | "
          f"dead={dead_0:.3f} | cka={cka_0:.3f}")
    metrics_writer.writerow([0, pong_r_0, dead_0, cka_0])
    metrics_file.flush()

    # ── Training loop ─────────────────────────────────────────────────────────
    obs, _      = env.reset()
    ep_reward   = 0
    ep_length   = 0
    episode     = 0
    global_step = 0
    next_eval   = CFG["eval_freq"]
    next_ckpt   = CFG["checkpoint_freq"]

    print(f"\n[training] Starting Breakout training ({total_steps:,} steps)...\n")

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

        # ── Forgetting evaluation ─────────────────────────────────────────
        if global_step >= next_eval:
            print(f"\n[eval] Measuring forgetting at Breakout step {global_step:,}...")
            pong_r = evaluate_pong_forgetting(agent.online_net, pong_fc_out_state,
                                               device, CFG["eval_pong_episodes"])
            dead   = measure_dead_neurons(agent.online_net, CFG["env_id"],
                                           CFG["dead_neuron_states"], device)
            cka_r  = collect_representations(agent.online_net, "ALE/Pong-v5",
                                              CFG["cka_states"], device, seed=42)
            cka    = linear_cka(ref_reps, cka_r)

            print(f"[eval] step={global_step:,} | pong_reward={pong_r:.2f} | "
                  f"dead={dead:.3f} | cka={cka:.3f}")
            metrics_writer.writerow([global_step, pong_r, dead, cka])
            metrics_file.flush()
            next_eval += CFG["eval_freq"]

        # ── Checkpoint ────────────────────────────────────────────────────
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

    # ── Final eval ────────────────────────────────────────────────────────────
    print(f"\n[eval] Final evaluation at step {global_step:,}...")
    pong_r = evaluate_pong_forgetting(agent.online_net, pong_fc_out_state,
                                       device, CFG["eval_pong_episodes"])
    dead   = measure_dead_neurons(agent.online_net, CFG["env_id"],
                                   CFG["dead_neuron_states"], device)
    cka_r  = collect_representations(agent.online_net, "ALE/Pong-v5",
                                      CFG["cka_states"], device, seed=42)
    cka    = linear_cka(ref_reps, cka_r)
    print(f"[eval] FINAL | pong_reward={pong_r:.2f} | dead={dead:.3f} | cka={cka:.3f}")
    metrics_writer.writerow([global_step, pong_r, dead, cka])
    metrics_file.flush()
    metrics_file.close()

    env.close()
    logger.close()

    print(f"\n{'='*60}")
    print(f"  Done: {run_name}")
    print(f"  Steps: {global_step:,} | Episodes: {episode}")
    print(f"  Final Pong reward: {pong_r:.2f}")
    print(f"  Final dead neurons: {dead:.3f}")
    print(f"  Final CKA drift: {cka:.3f}")
    print(f"  Metrics saved → {metrics_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True, choices=["conv", "none"],
                        help="none = full fine-tune | conv = freeze conv layers")
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(CFG["log_dir"],        exist_ok=True)
    os.makedirs(CFG["checkpoint_dir"], exist_ok=True)

    train(freeze=args.freeze, steps_override=args.steps)

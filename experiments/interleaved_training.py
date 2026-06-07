"""
Interleaved Training Experiment (v2 — step-level alternation)
===============================================================
A single DQN agent with a shared backbone and two output heads alternates
between Pong and Breakout at the STEP level, not the episode level.

Why step-level matters:
    Episode-level alternation with Pong (~200 steps/episode) and Breakout
    with EpisodicLife (~10 steps/episode) produces a ~20:1 gradient imbalance.
    Step-level alternation switches every switch_freq steps regardless of
    episode boundaries, ensuring both games contribute equally to the shared
    backbone regardless of episode length asymmetry.

Architecture:
    Shared: conv layers + fc_repr (512-dim)
    Pong head:     fc_out_pong     (512 → 6 actions)
    Breakout head: fc_out_breakout (512 → 4 actions)

Training:
    Active game switches every switch_freq=1000 steps.
    Separate replay buffers per game.
    Gradients from both games flow back through the shared backbone equally.

Total: 1M steps per game = 2M total steps.

Metrics logged every 100k total steps:
    1. Pong reward      (mean of recent Pong episodes)
    2. Breakout reward  (mean of recent Breakout episodes)
    3. Dead neurons     (fc_repr activation on 1000 Pong frames)
    4. CKA drift        (current backbone vs original Pong fc_repr baseline)

Usage:
    python experiments/interleaved_training.py
    python experiments/interleaved_training.py --steps_per_game 50000  # smoke test
    python experiments/interleaved_training.py --switch_freq 1000
"""

import os
import sys
import csv
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn import AtariCNN
from models.cnn_two_head import AtariCNNTwoHead
from envs.wrappers import make_atari_env
from utils.replay_buffer import ReplayBuffer
from utils.logger import Logger
from utils.checkpoint import save_checkpoint, load_checkpoint
from train import set_seeds, get_device


SOURCE_CHECKPOINT = (
    "results/checkpoints/"
    "dqn_pong_seed42_scalemedium_lr0001_buf100k_step02000000.pt"
)

CFG = {
    "steps_per_game":      1_000_000,
    "switch_freq":         1_000,
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
    "epsilon_decay_steps": 200_000,
    "log_dir":             "results/logs",
    "checkpoint_dir":      "results/checkpoints",
    "checkpoint_freq":     500_000,
    "eval_freq":           100_000,
    "dead_neuron_states":  1_000,
    "cka_states":          1_000,
    "print_freq":          500,
}

GAMES = {
    "pong":     ("ALE/Pong-v5",     6),
    "breakout": ("ALE/Breakout-v5", 4),
}


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


# ── Two-head DQN agent ────────────────────────────────────────────────────────
class InterleavedDQNAgent:

    def __init__(self, device, lr, gamma, epsilon_start, epsilon_end,
                 epsilon_decay_steps, target_update_freq, net_scale, grad_clip):
        self.device             = device
        self.gamma              = gamma
        self.epsilon            = epsilon_start
        self.epsilon_end        = epsilon_end
        self.epsilon_decay      = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.target_update_freq = target_update_freq
        self.grad_clip          = grad_clip
        self._step              = 0

        self.online_net = AtariCNNTwoHead(net_scale=net_scale).to(device)
        self.target_net = AtariCNNTwoHead(net_scale=net_scale).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

    def select_action(self, obs: np.ndarray, game: str) -> int:
        n_actions = GAMES[game][1]
        if np.random.random() < self.epsilon:
            return np.random.randint(n_actions)
        state_t = torch.from_numpy(obs).float().div(255.0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.online_net(state_t, game=game)
        return int(q_vals.argmax(1).item())

    def learn(self, batch, game: str):
        states      = batch.states.float() / 255.0
        actions     = batch.actions.long()
        rewards     = batch.rewards
        next_states = batch.next_states.float() / 255.0
        dones       = batch.dones

        with torch.no_grad():
            next_q  = self.target_net(next_states, game=game).max(1).values
            targets = rewards + self.gamma * next_q * (1 - dones)

        current_q = self.online_net(states, game=game).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        loss = F.smooth_l1_loss(current_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self._step += 1
        if self._step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
        return float(loss.item()), float(current_q.mean().item())


# ── Metric helpers ────────────────────────────────────────────────────────────
def collect_repr(model, env_id, n_states, device, game, seed=99):
    reps   = []
    handle = model.fc_repr.register_forward_hook(
        lambda m, i, o: reps.append(o.detach().cpu().numpy())
    )
    env = make_atari_env(env_id, seed=seed)
    obs, _ = env.reset()
    model.eval()
    with torch.no_grad():
        for _ in range(n_states):
            state_t = torch.from_numpy(obs).float().div(255.0).unsqueeze(0).to(device)
            _ = model(state_t, game=game)
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                obs, _ = env.reset()
    handle.remove()
    env.close()
    return np.vstack(reps)


def measure_dead_neurons(model, device):
    reps   = collect_repr(model, "ALE/Pong-v5", CFG["dead_neuron_states"],
                           device, game="pong", seed=77)
    active = (reps > 0).mean(axis=0)
    return float((active < 0.05).mean())


# ── Main ──────────────────────────────────────────────────────────────────────
def train(steps_per_game: int, switch_freq: int):
    run_name = "dqn_interleaved_v2_seed42_scalemedium"
    set_seeds(CFG["seed"])
    device = get_device()

    print(f"\n{'='*60}")
    print(f"  Interleaved Training v2 (step-level alternation)")
    print(f"  {steps_per_game:,} steps per game = {steps_per_game*2:,} total")
    print(f"  Switch frequency: every {switch_freq} steps")
    print(f"{'='*60}\n")

    # Load original Pong reference for CKA drift measurement
    print("[setup] Loading Pong reference for CKA drift...")
    pong_ckpt = torch.load(SOURCE_CHECKPOINT, map_location=device, weights_only=True)
    ref_model = AtariCNN(n_actions=6, net_scale="medium").to(device)
    ref_model.load_state_dict(pong_ckpt["model_state"])
    ref_model.eval()

    def collect_repr_standard(model, env_id, n_states, device, seed=42):
        reps   = []
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

    ref_reps = collect_repr_standard(ref_model, "ALE/Pong-v5",
                                      CFG["cka_states"], device, seed=42)
    print(f"[setup] Reference Pong reps: {ref_reps.shape}\n")

    agent = InterleavedDQNAgent(
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

    buffers = {
        g: ReplayBuffer(capacity=CFG["buffer_capacity"],
                        obs_shape=(4, 84, 84), device=device)
        for g in GAMES
    }
    loggers = {
        g: Logger(log_dir=CFG["log_dir"],
                  run_name=f"{run_name}_{g}", use_tb=False)
        for g in GAMES
    }

    os.makedirs("results/logs", exist_ok=True)
    metrics_path = f"results/logs/{run_name}_metrics.csv"
    mf  = open(metrics_path, "w", newline="")
    mw  = csv.writer(mf)
    mw.writerow(["total_step", "pong_steps", "breakout_steps",
                 "pong_reward_mean10", "breakout_reward_mean10",
                 "dead_neurons", "cka_drift_from_pong"])
    mf.flush()

    game_steps  = {"pong": 0, "breakout": 0}
    recent_r    = {"pong": [], "breakout": []}
    ep_reward   = {"pong": 0.0, "breakout": 0.0}
    ep_length   = {"pong": 0,   "breakout": 0}
    total_step  = 0
    next_eval   = CFG["eval_freq"]
    next_ckpt   = CFG["checkpoint_freq"]
    steps_since_switch = 0
    current_game = "pong"

    envs = {g: make_atari_env(GAMES[g][0], seed=CFG["seed"]) for g in GAMES}
    obs  = {g: envs[g].reset()[0] for g in GAMES}

    def get_mean_r(game):
        if not recent_r[game]:
            return 0.0
        return float(np.mean(recent_r[game][-10:]))

    def evaluate_and_log():
        dead = measure_dead_neurons(agent.online_net, device)
        curr_reps = collect_repr(agent.online_net, "ALE/Pong-v5",
                                  CFG["cka_states"], device,
                                  game="pong", seed=42)
        cka  = linear_cka(ref_reps, curr_reps)
        pr   = get_mean_r("pong")
        br   = get_mean_r("breakout")
        print(f"\n[eval] total={total_step:,} | "
              f"pong_steps={game_steps['pong']:,} | "
              f"breakout_steps={game_steps['breakout']:,}")
        print(f"  pong_r={pr:.2f} | breakout_r={br:.2f} | "
              f"dead={dead:.3f} | cka_drift={cka:.3f}")
        mw.writerow([total_step, game_steps["pong"], game_steps["breakout"],
                     pr, br, dead, cka])
        mf.flush()

    # Baseline
    print("[eval] Baseline (step 0)...")
    evaluate_and_log()
    print(f"\n[training] Starting step-level interleaved training...\n")

    while game_steps["pong"] < steps_per_game or game_steps["breakout"] < steps_per_game:

        # Skip if current game has already reached its target
        if game_steps[current_game] >= steps_per_game:
            current_game = "pong" if current_game == "breakout" else "breakout"
            steps_since_switch = 0
            continue

        game    = current_game
        action  = agent.select_action(obs[game], game=game)
        next_ob, reward, terminated, truncated, _ = envs[game].step(action)
        done    = terminated or truncated

        buffers[game].push(obs[game], action, float(reward), next_ob, done)
        obs[game]        = next_ob if not done else envs[game].reset()[0]
        ep_reward[game] += reward
        ep_length[game] += 1
        game_steps[game] += 1
        total_step       += 1
        steps_since_switch += 1

        if game_steps[game] >= CFG["learning_starts"] and \
                len(buffers[game]) >= CFG["batch_size"]:
            batch        = buffers[game].sample(CFG["batch_size"])
            loss, mean_q = agent.learn(batch, game=game)
            loggers[game].log_step(loss, mean_q)

        if done:
            loggers[game].log_episode(ep_reward[game], ep_length[game], agent.epsilon)
            recent_r[game].append(ep_reward[game])
            if len(recent_r[game]) % CFG["print_freq"] == 0:
                print(f"[{game}] ep={len(recent_r[game])} | "
                      f"steps={game_steps[game]:,} | "
                      f"r={ep_reward[game]:.1f} | "
                      f"mean10={get_mean_r(game):.1f} | "
                      f"ε={agent.epsilon:.3f}")
            ep_reward[game] = 0.0
            ep_length[game] = 0

        # Step-level switch
        if steps_since_switch >= switch_freq:
            other = "pong" if game == "breakout" else "breakout"
            if game_steps[other] < steps_per_game:
                current_game = other
            steps_since_switch = 0

        if total_step >= next_eval:
            evaluate_and_log()
            next_eval += CFG["eval_freq"]

        if total_step >= next_ckpt:
            save_checkpoint(
                checkpoint_dir = CFG["checkpoint_dir"],
                run_name       = run_name,
                step           = total_step,
                model          = agent.online_net,
                target_model   = agent.target_net,
                optimizer      = agent.optimizer,
                episode        = sum(len(r) for r in recent_r.values()),
                epsilon        = agent.epsilon,
                config         = CFG,
            )
            next_ckpt += CFG["checkpoint_freq"]

    print(f"\n[eval] Final...")
    evaluate_and_log()
    mf.close()

    for g in GAMES:
        envs[g].close()
        loggers[g].close()

    print(f"\n{'='*60}")
    print(f"  Done: {run_name}")
    print(f"  Pong: {game_steps['pong']:,} steps | "
          f"Breakout: {game_steps['breakout']:,} steps")
    print(f"  Metrics → {metrics_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps_per_game", type=int,
                        default=CFG["steps_per_game"])
    parser.add_argument("--switch_freq",    type=int,
                        default=CFG["switch_freq"])
    args = parser.parse_args()

    os.makedirs(CFG["log_dir"],        exist_ok=True)
    os.makedirs(CFG["checkpoint_dir"], exist_ok=True)

    train(steps_per_game=args.steps_per_game, switch_freq=args.switch_freq)

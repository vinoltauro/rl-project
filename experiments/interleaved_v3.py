"""
Interleaved Training v3 — Episode-Level, Joint Replay Buffer
=============================================================

Key design changes from v2 (step-level, separate buffers):

  1. Episode-level game switching — active game only switches at episode
     boundaries (done=True). Never cuts mid-episode.

  2. Single joint replay buffer — all Pong and Breakout transitions stored
     together, tagged with a game_id. When sampling a batch, transitions from
     both games are mixed. Each transition is routed through its correct output
     head during the loss computation.

  3. Round-robin game selection — at each episode boundary, switch to the game
     with fewer steps so far. Ensures both games get equal gradient contribution
     over the full run.

Architecture:
    Shared backbone: conv1-3 + fc_repr (512-dim)
    Pong head:       fc_out_pong     (512 → 6 actions)
    Breakout head:   fc_out_breakout (512 → 4 actions)

Metrics logged every eval_freq total steps:
    1. Pong reward      (mean of last 10 Pong episodes)
    2. Breakout reward  (mean of last 10 Breakout episodes)
    3. Dead neurons     (fc_repr activation on 1000 Pong frames)
    4. CKA drift        (current fc_repr vs original Pong DQN baseline)

Results isolated to results/interleaved_v3/ — nothing in results/ is overwritten.

Usage:
    python experiments/interleaved_v3.py
    python experiments/interleaved_v3.py --steps_per_game 50000  # smoke test
    python experiments/interleaved_v3.py --steps_per_game 1000000
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
from utils.logger import Logger
from utils.checkpoint import save_checkpoint
from train import set_seeds, get_device


# ── Paths ─────────────────────────────────────────────────────────────────────
SOURCE_CHECKPOINT = (
    "results/checkpoints/"
    "dqn_pong_seed42_scalemedium_lr0001_buf100k_step02000000.pt"
)
RESULTS_DIR = "results/interleaved_v3"

# ── Config ────────────────────────────────────────────────────────────────────
CFG = {
    "steps_per_game":      1_000_000,
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
    "checkpoint_freq":     500_000,
    "eval_freq":           100_000,
    "dead_neuron_states":  1_000,
    "cka_states":          1_000,
    "print_freq":          100,
}

GAMES    = {"pong": ("ALE/Pong-v5", 6), "breakout": ("ALE/Breakout-v5", 4)}
GAME_IDS = {"pong": 0, "breakout": 1}


# ── Joint Replay Buffer ───────────────────────────────────────────────────────
class JointReplayBuffer:
    """
    Circular replay buffer storing transitions from both games.
    Each transition is tagged with a game_id (0=pong, 1=breakout).
    When sampled, a batch can contain transitions from both games;
    the caller routes each through the correct network head.
    """

    def __init__(self, capacity: int, obs_shape: tuple, device: torch.device):
        self.capacity = capacity
        self.device   = device
        self._ptr     = 0
        self._size    = 0

        self._states      = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self._next_states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self._actions     = np.zeros(capacity, dtype=np.int64)
        self._rewards     = np.zeros(capacity, dtype=np.float32)
        self._dones       = np.zeros(capacity, dtype=np.float32)
        self._game_ids    = np.zeros(capacity, dtype=np.int64)

    def push(self, state, action, reward, next_state, done, game_id):
        self._states[self._ptr]      = state
        self._next_states[self._ptr] = next_state
        self._actions[self._ptr]     = action
        self._rewards[self._ptr]     = reward
        self._dones[self._ptr]       = float(done)
        self._game_ids[self._ptr]    = game_id
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self._size, size=batch_size)
        return (
            torch.from_numpy(self._states[idx]).float().div(255.0).to(self.device),
            torch.from_numpy(self._actions[idx]).to(self.device),
            torch.from_numpy(self._rewards[idx]).to(self.device),
            torch.from_numpy(self._next_states[idx]).float().div(255.0).to(self.device),
            torch.from_numpy(self._dones[idx]).to(self.device),
            torch.from_numpy(self._game_ids[idx]).to(self.device),
        )

    def __len__(self):
        return self._size

    def memory_usage_mb(self) -> float:
        return (self._states.nbytes + self._next_states.nbytes +
                self._actions.nbytes + self._rewards.nbytes +
                self._dones.nbytes + self._game_ids.nbytes) / (1024 ** 2)


# ── Agent ─────────────────────────────────────────────────────────────────────
class InterleavedAgentV3:
    """
    Two-head DQN agent for joint training on Pong and Breakout.
    learn_joint() processes a mixed batch, routing each transition
    through the correct game head before computing the loss.
    """

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

        print(f"[InterleavedV3] Initialised | "
              f"params={sum(p.numel() for p in self.online_net.parameters()):,} | "
              f"device={device}")

    def select_action(self, obs: np.ndarray, game: str) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(GAMES[game][1])
        state_t = (torch.from_numpy(obs).float().div(255.0)
                   .unsqueeze(0).to(self.device))
        with torch.no_grad():
            return int(self.online_net(state_t, game=game).argmax(1).item())

    def learn_joint(self, batch) -> float:
        """
        Compute DQN loss on a mixed-game batch.
        Each game's transitions are processed separately through their own head.
        Gradients from both games flow back through the shared backbone.
        """
        states, actions, rewards, next_states, dones, game_ids = batch

        losses = []
        for game, gid in [("pong", GAME_IDS["pong"]),
                           ("breakout", GAME_IDS["breakout"])]:
            mask = (game_ids == gid)
            if mask.sum() == 0:
                continue

            s  = states[mask]
            a  = actions[mask]
            r  = rewards[mask]
            ns = next_states[mask]
            d  = dones[mask]

            with torch.no_grad():
                next_q  = self.target_net(ns, game=game).max(1).values
                targets = r + self.gamma * next_q * (1.0 - d)

            current_q = self.online_net(s, game=game).gather(
                1, a.unsqueeze(1)
            ).squeeze(1)

            losses.append(F.smooth_l1_loss(current_q, targets))

        if not losses:
            return 0.0

        total_loss = sum(losses)
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self._step += 1
        if self._step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
        return float(total_loss.item())


# ── Metric helpers ────────────────────────────────────────────────────────────
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


def collect_repr_two_head(model: AtariCNNTwoHead, env_id: str,
                           n_states: int, device: torch.device,
                           game: str, seed: int = 99) -> np.ndarray:
    """Collect fc_repr activations (post-ReLU) from the two-head model."""
    env = make_atari_env(env_id, seed=seed)
    obs, _ = env.reset()
    reps = []
    model.eval()
    with torch.no_grad():
        for _ in range(n_states):
            state_t = (torch.from_numpy(obs).float().div(255.0)
                       .unsqueeze(0).to(device))
            _ = model(state_t, game=game)
            reps.append(model.representation.cpu().numpy())
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                obs, _ = env.reset()
    env.close()
    return np.vstack(reps)


def collect_repr_standard(model: AtariCNN, env_id: str,
                           n_states: int, device: torch.device,
                           seed: int = 42) -> np.ndarray:
    """Collect fc_repr activations from a standard single-head AtariCNN."""
    env = make_atari_env(env_id, seed=seed)
    obs, _ = env.reset()
    reps = []
    model.eval()
    with torch.no_grad():
        for _ in range(n_states):
            state_t = (torch.from_numpy(obs).float().div(255.0)
                       .unsqueeze(0).to(device))
            _ = model(state_t)
            reps.append(model.representation.cpu().numpy())
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                obs, _ = env.reset()
    env.close()
    return np.vstack(reps)


def measure_dead_neurons(model: AtariCNNTwoHead, device: torch.device,
                          n_states: int, seed: int = 77) -> float:
    reps   = collect_repr_two_head(model, "ALE/Pong-v5", n_states,
                                    device, game="pong", seed=seed)
    active = (reps > 0).mean(axis=0)
    return float((active < 0.05).mean())


# ── Game selection ────────────────────────────────────────────────────────────
def select_next_game(last_game: str, game_steps: dict,
                      target_steps: int) -> str:
    """
    Round-robin: switch to the other game if it still has steps remaining.
    Fall back to the current game if the other is done.
    Returns None when both games have reached target_steps.
    """
    other = "pong" if last_game == "breakout" else "breakout"
    if game_steps[other] < target_steps:
        return other
    if game_steps[last_game] < target_steps:
        return last_game
    return None


# ── Main training loop ────────────────────────────────────────────────────────
def train(steps_per_game: int):
    run_name = "dqn_interleaved_v3_seed42_scalemedium"
    set_seeds(CFG["seed"])
    device = get_device()

    ckpt_dir = os.path.join(RESULTS_DIR, "checkpoints")
    log_dir  = os.path.join(RESULTS_DIR, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Interleaved Training v3 (episode-level, joint buffer)")
    print(f"  {steps_per_game:,} steps per game = {steps_per_game*2:,} total")
    print(f"{'='*60}\n")

    # ── Load Pong reference for CKA drift ────────────────────────────────────
    print("[setup] Loading Pong reference checkpoint for CKA baseline...")
    pong_ckpt  = torch.load(SOURCE_CHECKPOINT, map_location=device, weights_only=True)
    ref_model  = AtariCNN(n_actions=6, net_scale="medium").to(device)
    ref_model.load_state_dict(pong_ckpt["model_state"])
    ref_reps   = collect_repr_standard(ref_model, "ALE/Pong-v5",
                                        CFG["cka_states"], device, seed=42)
    print(f"[setup] Reference reps: {ref_reps.shape}\n")

    # ── Agent and buffer ──────────────────────────────────────────────────────
    agent = InterleavedAgentV3(
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

    buffer = JointReplayBuffer(CFG["buffer_capacity"], (4, 84, 84), device)
    print(f"[buffer] Joint buffer | capacity={CFG['buffer_capacity']:,} | "
          f"memory≈{buffer.memory_usage_mb():.0f}MB\n")

    # ── Loggers ───────────────────────────────────────────────────────────────
    loggers = {
        g: Logger(log_dir=log_dir, run_name=f"{run_name}_{g}", use_tb=False)
        for g in GAMES
    }

    # ── Metrics CSV ───────────────────────────────────────────────────────────
    metrics_path = os.path.join(log_dir, f"{run_name}_metrics.csv")
    mf  = open(metrics_path, "w", newline="")
    mw  = csv.writer(mf)
    mw.writerow(["total_step", "pong_steps", "breakout_steps",
                 "pong_reward_mean10", "breakout_reward_mean10",
                 "dead_neurons", "cka_drift_from_pong"])
    mf.flush()

    # ── Training state ────────────────────────────────────────────────────────
    game_steps   = {"pong": 0, "breakout": 0}
    recent_r     = {"pong": [], "breakout": []}
    ep_reward    = 0.0
    ep_length    = 0
    total_steps  = 0
    next_eval    = CFG["eval_freq"]
    next_ckpt    = CFG["checkpoint_freq"]
    current_game = "pong"

    envs = {g: make_atari_env(GAMES[g][0], seed=CFG["seed"]) for g in GAMES}
    obs  = {g: envs[g].reset()[0] for g in GAMES}

    def get_mean_r(game):
        return float(np.mean(recent_r[game][-10:])) if recent_r[game] else 0.0

    def evaluate_and_log(step):
        dead = measure_dead_neurons(agent.online_net, device,
                                     CFG["dead_neuron_states"])
        curr_reps = collect_repr_two_head(agent.online_net, "ALE/Pong-v5",
                                           CFG["cka_states"], device,
                                           game="pong", seed=42)
        cka  = linear_cka(ref_reps, curr_reps)
        pr   = get_mean_r("pong")
        br   = get_mean_r("breakout")
        print(f"\n[eval] step={step:,} | "
              f"pong_steps={game_steps['pong']:,} | "
              f"breakout_steps={game_steps['breakout']:,}")
        print(f"  pong_r={pr:.2f} | breakout_r={br:.2f} | "
              f"dead={dead:.3f} | cka={cka:.3f}")
        mw.writerow([step, game_steps["pong"], game_steps["breakout"],
                     pr, br, dead, cka])
        mf.flush()

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("[eval] Baseline (step 0)...")
    evaluate_and_log(0)
    print(f"\n[training] Starting episode-level interleaved training...\n")

    # ── Main loop ─────────────────────────────────────────────────────────────
    while (game_steps["pong"] < steps_per_game or
           game_steps["breakout"] < steps_per_game):

        # If current game is done, find next game
        if game_steps[current_game] >= steps_per_game:
            next_g = select_next_game(current_game, game_steps, steps_per_game)
            if next_g is None:
                break
            current_game = next_g
            ep_reward    = 0.0
            ep_length    = 0
            continue

        game   = current_game
        action = agent.select_action(obs[game], game)
        next_ob, reward, term, trunc, _ = envs[game].step(action)
        done = term or trunc

        buffer.push(obs[game], action, float(reward), next_ob, done,
                    GAME_IDS[game])
        obs[game] = next_ob if not done else envs[game].reset()[0]

        ep_reward        += reward
        ep_length        += 1
        game_steps[game] += 1
        total_steps      += 1

        # Learn from joint buffer every step
        if (total_steps >= CFG["learning_starts"] and
                len(buffer) >= CFG["batch_size"]):
            batch = buffer.sample(CFG["batch_size"])
            loss  = agent.learn_joint(batch)
            loggers[game].log_step(loss, None)

        if done:
            loggers[game].log_episode(ep_reward, ep_length, agent.epsilon)
            recent_r[game].append(ep_reward)

            if len(recent_r[game]) % CFG["print_freq"] == 0:
                print(f"[{game}] ep={len(recent_r[game])} | "
                      f"steps={game_steps[game]:,} | "
                      f"r={ep_reward:.1f} | "
                      f"mean10={get_mean_r(game):.1f} | "
                      f"ε={agent.epsilon:.3f}")

            ep_reward = 0.0
            ep_length = 0

            # Episode boundary — switch games (round-robin)
            next_g = select_next_game(game, game_steps, steps_per_game)
            if next_g is not None:
                current_game = next_g

        if total_steps >= next_eval:
            evaluate_and_log(total_steps)
            next_eval += CFG["eval_freq"]

        if total_steps >= next_ckpt:
            save_checkpoint(
                checkpoint_dir = ckpt_dir,
                run_name       = run_name,
                step           = total_steps,
                model          = agent.online_net,
                target_model   = agent.target_net,
                optimizer      = agent.optimizer,
                episode        = sum(len(r) for r in recent_r.values()),
                epsilon        = agent.epsilon,
                config         = CFG,
            )
            next_ckpt += CFG["checkpoint_freq"]

    # ── Final eval ────────────────────────────────────────────────────────────
    print(f"\n[eval] Final...")
    evaluate_and_log(total_steps)
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
    args = parser.parse_args()

    train(steps_per_game=args.steps_per_game)

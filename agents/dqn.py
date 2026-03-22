"""
DQN Agent
=========
Implements Deep Q-Network (Mnih et al., 2015).

Key components:
  - Online network:  updated every step via gradient descent
  - Target network:  frozen copy, updated every `target_update_freq` steps
  - Epsilon-greedy exploration with linear decay
  - Gradient clipping for stability

Loss (standard DQN TD error):
    y = r + γ · max_{a'} Q(s', a'; θ⁻)          (target network)
    L = MSE(Q(s, a; θ), y)

Known issue: this formulation uses the same network (θ⁻) to both SELECT
and EVALUATE the best next action, leading to overestimation bias.
Double DQN (ddqn.py) fixes this with one line change.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple

from models.cnn import AtariCNN
from utils.replay_buffer import ReplayBuffer, Batch


class DQNAgent:
    """
    DQN agent with experience replay and target network.

    Args:
        n_actions:          Size of the discrete action space
        device:             Torch device (cpu / cuda)
        lr:                 Learning rate for Adam optimiser
        gamma:              Discount factor
        epsilon_start:      Initial exploration rate
        epsilon_end:        Minimum exploration rate
        epsilon_decay_steps: Steps over which epsilon is annealed
        target_update_freq: Steps between target network hard updates
        net_scale:          CNN size — "small", "medium", "large"
        grad_clip:          Max gradient norm (None to disable)
    """

    def __init__(
        self,
        n_actions:            int,
        device:               torch.device,
        lr:                   float = 1e-4,
        gamma:                float = 0.99,
        epsilon_start:        float = 1.0,
        epsilon_end:          float = 0.01,
        epsilon_decay_steps:  int   = 100_000,
        target_update_freq:   int   = 1_000,
        net_scale:            str   = "medium",
        grad_clip:            Optional[float] = 10.0,
    ):
        self.n_actions           = n_actions
        self.device              = device
        self.gamma               = gamma
        self.epsilon             = epsilon_start
        self.epsilon_end         = epsilon_end
        self.epsilon_decay       = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.target_update_freq  = target_update_freq
        self.grad_clip           = grad_clip
        self._step               = 0

        # ── Networks ─────────────────────────────────────────────────────
        self.online_net = AtariCNN(n_actions, net_scale=net_scale).to(device)
        self.target_net = AtariCNN(n_actions, net_scale=net_scale).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()   # Target net is never trained directly

        # ── Optimiser ────────────────────────────────────────────────────
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn   = nn.MSELoss()

        print(f"[DQN] Initialised | params={self.online_net.count_parameters():,} | "
              f"device={device} | net_scale={net_scale}")

    # ─────────────────────────────────────────────────────────────────────
    # Action selection
    # ─────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        Epsilon-greedy action selection.

        Args:
            state:     (4, 84, 84) numpy uint8 array
            eval_mode: If True, use greedy policy (ε=0) for evaluation

        Returns:
            Integer action index
        """
        epsilon = 0.0 if eval_mode else self.epsilon

        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)

        state_t = (
            torch.from_numpy(state)
            .float()
            .div(255.0)
            .unsqueeze(0)        # (1, 4, 84, 84)
            .to(self.device)
        )
        q_values = self.online_net(state_t)
        return int(q_values.argmax(dim=1).item())

    # ─────────────────────────────────────────────────────────────────────
    # Learning step
    # ─────────────────────────────────────────────────────────────────────
    def learn(self, batch: Batch) -> Tuple[float, float]:
        """
        Perform one gradient update on a sampled minibatch.

        Args:
            batch: Batch namedtuple from ReplayBuffer.sample()

        Returns:
            (loss value, mean max Q-value) for logging
        """
        self._step += 1

        # ── Compute TD targets ───────────────────────────────────────────
        with torch.no_grad():
            # Standard DQN: target net selects AND evaluates the best action
            next_q = self.target_net(batch.next_states)         # (B, n_actions)
            max_next_q = next_q.max(dim=1).values               # (B,)
            targets = batch.rewards + self.gamma * max_next_q * (1.0 - batch.dones)

        # ── Compute current Q-values ─────────────────────────────────────
        all_q = self.online_net(batch.states)                   # (B, n_actions)
        # Gather Q-values for the actions that were actually taken
        q_taken = all_q.gather(1, batch.actions.unsqueeze(1)).squeeze(1)  # (B,)

        # ── Loss and gradient step ───────────────────────────────────────
        loss = self.loss_fn(q_taken, targets)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)
        self.optimizer.step()

        # ── Epsilon decay ────────────────────────────────────────────────
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

        # ── Target network update ────────────────────────────────────────
        if self._step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        mean_q = float(all_q.max(dim=1).values.mean().item())
        return float(loss.item()), mean_q

    # ─────────────────────────────────────────────────────────────────────
    def state_dict_bundle(self):
        """Return everything needed to reconstruct agent state."""
        return {
            "online_net":  self.online_net.state_dict(),
            "target_net":  self.target_net.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "epsilon":     self.epsilon,
            "step":        self._step,
        }

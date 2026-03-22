"""
Double DQN Agent
================
Implements Double Q-Learning for Atari (van Hasselt et al., 2016).

This class inherits EVERYTHING from DQNAgent and overrides only the
TD target computation inside `learn()`. This is intentional — it
demonstrates that DDQN is architecturally identical to DQN.

The fix (one conceptual line):
    DQN target:
        y = r + γ · Q(s', argmax_{a'} Q(s', a'; θ⁻); θ⁻)
                                              ^target net  ^target net
        → Same network selects AND evaluates: causes overestimation

    DDQN target:
        y = r + γ · Q(s', argmax_{a'} Q(s', a'; θ);  θ⁻)
                                              ^online net ^target net
        → Online net selects, target net evaluates: unbiased estimate

Why does this matter for representations?
  - DQN's overestimation introduces noise into the gradient signal
  - DDQN's cleaner gradients should produce more structured, compact
    representations in the 512-dim layer — hypothesis to test with t-SNE
"""

import torch
from typing import Tuple

from agents.dqn import DQNAgent
from utils.replay_buffer import Batch


class DoubleDQNAgent(DQNAgent):
    """
    Double DQN agent.

    Identical to DQNAgent in every way except the TD target computation.
    All hyperparameters, initialisation, and infrastructure are inherited.

    Usage is identical to DQNAgent — just swap the class name.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"[DoubleDQN] Overriding DQN target with Double Q-learning target")

    def learn(self, batch: Batch) -> Tuple[float, float]:
        """
        Double DQN gradient update.

        The ONLY difference from DQNAgent.learn() is in the target computation:
          - Action SELECTION uses the online network (θ)
          - Action EVALUATION uses the target network (θ⁻)

        Everything else (loss fn, optimiser, epsilon decay, target update) is
        inherited from DQNAgent unchanged.
        """
        self._step += 1

        # ── Compute Double DQN TD targets ────────────────────────────────
        with torch.no_grad():
            # Step 1: Online network selects the best next action
            next_q_online = self.online_net(batch.next_states)           # (B, n_actions)
            best_next_actions = next_q_online.argmax(dim=1, keepdim=True) # (B, 1)

            # Step 2: Target network EVALUATES that action (not its own best)
            next_q_target = self.target_net(batch.next_states)           # (B, n_actions)
            max_next_q = next_q_target.gather(1, best_next_actions).squeeze(1)  # (B,)

            # This decoupling removes the maximisation bias present in DQN
            targets = batch.rewards + self.gamma * max_next_q * (1.0 - batch.dones)

        # ── Everything below is identical to DQNAgent.learn() ────────────
        all_q = self.online_net(batch.states)
        q_taken = all_q.gather(1, batch.actions.unsqueeze(1)).squeeze(1)

        loss = self.loss_fn(q_taken, targets)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.online_net.parameters(), self.grad_clip
            )
        self.optimizer.step()

        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

        if self._step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        mean_q = float(all_q.max(dim=1).values.mean().item())
        return float(loss.item()), mean_q

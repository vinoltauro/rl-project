"""
Training Logger
===============
Logs all training metrics to:
  1. CSV file  — for post-hoc analysis and plotting
  2. TensorBoard — for live monitoring during training

Metrics tracked per episode:
  - Episode reward, length, epsilon
  - Mean Q-value (to detect overestimation)
  - Mean training loss
  - Frames per second

Metrics tracked per training step:
  - Loss, Q-values (for TensorBoard smoothing)
"""

import os
import csv
import time
from collections import deque
from typing import Optional
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class Logger:
    """
    Unified training logger.

    Args:
        log_dir:     Directory to save CSV and TensorBoard logs
        run_name:    Identifier for this run, e.g. "dqn_pong_seed42"
        use_tb:      Whether to write TensorBoard summaries
        window:      Rolling window size for smoothed metrics
    """

    def __init__(
        self,
        log_dir:  str,
        run_name: str,
        use_tb:   bool = True,
        window:   int  = 10,
    ):
        self.run_name = run_name
        self.window   = window
        self.start_time = time.time()

        # ── Directories ──────────────────────────────────────────────────
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, f"{run_name}.csv")

        # ── CSV setup ────────────────────────────────────────────────────
        self._csv_file = open(self.csv_path, "w", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=[
            "episode", "total_steps", "reward", "episode_length",
            "epsilon", "mean_q", "mean_loss", "fps",
        ])
        self._csv_writer.writeheader()

        # ── TensorBoard ──────────────────────────────────────────────────
        self.writer = None
        if use_tb and TENSORBOARD_AVAILABLE:
            tb_dir = os.path.join(log_dir, "tb", run_name)
            self.writer = SummaryWriter(tb_dir)

        # ── Rolling buffers ──────────────────────────────────────────────
        self._rewards  = deque(maxlen=window)
        self._losses   = deque(maxlen=100)    # More granular loss tracking
        self._q_values = deque(maxlen=100)

        # ── Counters ─────────────────────────────────────────────────────
        self.episode     = 0
        self.total_steps = 0
        self._ep_start_step = 0
        self._ep_start_time = time.time()

    # ─────────────────────────────────────────────────────────────────────
    def log_step(self, loss: Optional[float], mean_q: Optional[float]) -> None:
        """Call after every training update (not every env step)."""
        self.total_steps += 1
        if loss is not None:
            self._losses.append(loss)
        if mean_q is not None:
            self._q_values.append(mean_q)

        if self.writer and self.total_steps % 100 == 0:
            if loss is not None:
                self.writer.add_scalar("train/loss", loss, self.total_steps)
            if mean_q is not None:
                self.writer.add_scalar("train/mean_q", mean_q, self.total_steps)

    # ─────────────────────────────────────────────────────────────────────
    def log_episode(self, reward: float, length: int, epsilon: float) -> None:
        """Call at the end of each episode."""
        self.episode += 1
        self._rewards.append(reward)

        elapsed      = time.time() - self._ep_start_time
        steps_in_ep  = self.total_steps - self._ep_start_step
        fps          = steps_in_ep / max(elapsed, 1e-6)
        mean_q       = float(np.mean(self._q_values)) if self._q_values else 0.0
        mean_loss    = float(np.mean(self._losses))   if self._losses   else 0.0
        mean_reward  = float(np.mean(self._rewards))

        row = {
            "episode":        self.episode,
            "total_steps":    self.total_steps,
            "reward":         reward,
            "episode_length": length,
            "epsilon":        round(epsilon, 4),
            "mean_q":         round(mean_q, 4),
            "mean_loss":      round(mean_loss, 6),
            "fps":            round(fps, 1),
        }
        self._csv_writer.writerow(row)
        self._csv_file.flush()

        if self.writer:
            self.writer.add_scalar("episode/reward",         reward,      self.episode)
            self.writer.add_scalar("episode/reward_smooth",  mean_reward, self.episode)
            self.writer.add_scalar("episode/length",         length,      self.episode)
            self.writer.add_scalar("episode/epsilon",        epsilon,     self.episode)
            self.writer.add_scalar("episode/mean_q",         mean_q,      self.episode)

        self._ep_start_step = self.total_steps
        self._ep_start_time = time.time()

        return mean_reward   # Return for console printing

    # ─────────────────────────────────────────────────────────────────────
    def print_progress(self, reward: float, epsilon: float) -> None:
        """Pretty-print a progress line to stdout."""
        elapsed = time.time() - self.start_time
        mean_r  = float(np.mean(self._rewards)) if self._rewards else reward
        mean_q  = float(np.mean(self._q_values)) if self._q_values else 0.0
        print(
            f"[{self.run_name}] "
            f"Ep {self.episode:5d} | "
            f"Steps {self.total_steps:8,d} | "
            f"R {reward:7.1f} | "
            f"Mean-R(10) {mean_r:7.1f} | "
            f"ε {epsilon:.3f} | "
            f"Q {mean_q:6.2f} | "
            f"Elapsed {elapsed/60:.1f}m"
        )

    # ─────────────────────────────────────────────────────────────────────
    def close(self) -> None:
        self._csv_file.close()
        if self.writer:
            self.writer.close()

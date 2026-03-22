"""
Experience Replay Buffer
========================
Stores (state, action, reward, next_state, done) transitions.
Uses a circular numpy buffer for memory efficiency — avoids Python list overhead.

Key design choices:
  - Pre-allocated numpy arrays: O(1) insert, O(1) sample
  - Stores uint8 frames (not float32) to save ~4× memory
  - Normalises to [0,1] only at sample time (on GPU if possible)
"""

import numpy as np
import torch
from typing import Tuple, NamedTuple


class Batch(NamedTuple):
    """A sampled minibatch, ready to be consumed by the agent."""
    states:      torch.Tensor   # (B, 4, 84, 84) float32 in [0,1]
    actions:     torch.Tensor   # (B,) int64
    rewards:     torch.Tensor   # (B,) float32
    next_states: torch.Tensor   # (B, 4, 84, 84) float32 in [0,1]
    dones:       torch.Tensor   # (B,) float32  (1.0 = terminal)


class ReplayBuffer:
    """
    Fixed-capacity circular replay buffer.

    Args:
        capacity:    Maximum number of transitions to store
        obs_shape:   Shape of a single observation, e.g. (4, 84, 84)
        device:      Torch device to move sampled batches to
    """

    def __init__(self, capacity: int, obs_shape: Tuple[int, ...], device: torch.device):
        self.capacity = capacity
        self.device   = device
        self._ptr     = 0       # Write pointer
        self._size    = 0       # Current fill level

        # Pre-allocate storage (uint8 to save memory)
        self._states      = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self._next_states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self._actions     = np.zeros(capacity, dtype=np.int64)
        self._rewards     = np.zeros(capacity, dtype=np.float32)
        self._dones       = np.zeros(capacity, dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────────
    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        """Store a single transition. Overwrites oldest when full."""
        self._states[self._ptr]      = state
        self._next_states[self._ptr] = next_state
        self._actions[self._ptr]     = action
        self._rewards[self._ptr]     = reward
        self._dones[self._ptr]       = float(done)

        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    # ─────────────────────────────────────────────────────────────────────
    def sample(self, batch_size: int) -> Batch:
        """
        Sample a random minibatch of transitions.

        Returns:
            Batch namedtuple with all tensors on self.device
        """
        assert self._size >= batch_size, (
            f"Buffer only has {self._size} transitions, "
            f"need at least {batch_size}"
        )
        idx = np.random.randint(0, self._size, size=batch_size)

        # Normalise pixel values to [0, 1] at sample time
        states      = torch.from_numpy(self._states[idx]).float().div(255.0).to(self.device)
        next_states = torch.from_numpy(self._next_states[idx]).float().div(255.0).to(self.device)
        actions     = torch.from_numpy(self._actions[idx]).to(self.device)
        rewards     = torch.from_numpy(self._rewards[idx]).to(self.device)
        dones       = torch.from_numpy(self._dones[idx]).to(self.device)

        return Batch(states, actions, rewards, next_states, dones)

    # ─────────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return self._size

    @property
    def is_ready(self) -> bool:
        """True once the buffer has enough transitions to start learning."""
        return self._size >= 1000  # Don't start learning on an almost-empty buffer

    def memory_usage_mb(self) -> float:
        """Approximate memory usage in MB."""
        total_bytes = (
            self._states.nbytes + self._next_states.nbytes
            + self._actions.nbytes + self._rewards.nbytes + self._dones.nbytes
        )
        return total_bytes / (1024 ** 2)


if __name__ == "__main__":
    buf = ReplayBuffer(capacity=10_000, obs_shape=(4, 84, 84), device=torch.device("cpu"))
    dummy_obs = np.zeros((4, 84, 84), dtype=np.uint8)
    for i in range(5000):
        buf.push(dummy_obs, action=0, reward=1.0, next_state=dummy_obs, done=False)
    batch = buf.sample(32)
    print(f"Buffer size:  {len(buf)}")
    print(f"Memory usage: {buf.memory_usage_mb():.1f} MB")
    print(f"States shape: {batch.states.shape}")
    print(f"Rewards:      {batch.rewards[:5]}")

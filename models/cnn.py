"""
Shared CNN Backbone
===================
Architecture from Mnih et al. (2015) — identical for DQN and Double DQN.
This is intentional: any difference in learned representations is purely
due to the algorithm or the game, not the network architecture.

Input:  (batch, 4, 84, 84)  — 4 stacked grayscale frames
Output: (batch, n_actions)  — Q-value per action

The 512-dim penultimate layer is the REPRESENTATION LAYER we extract for
t-SNE analysis. A forward hook is registered at __init__ time so callers
can access it via `model.representation` after a forward pass.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class AtariCNN(nn.Module):
    """
    Standard Atari CNN from Mnih et al. (2015).

    Layer sizes can be varied for ablation studies via the `net_scale`
    parameter: "small", "medium" (default), or "large".

    Architecture:
        Conv(32, 8×8, s=4) → ReLU
        Conv(64, 4×4, s=2) → ReLU
        Conv(64, 3×3, s=1) → ReLU
        Flatten
        Linear(512)        → ReLU   ← REPRESENTATION LAYER
        Linear(n_actions)           ← Q-values
    """

    SCALE_CONFIGS = {
        "small":  {"filters": [16, 32, 32],  "hidden": 256},
        "medium": {"filters": [32, 64, 64],  "hidden": 512},
        "large":  {"filters": [64, 128, 128], "hidden": 1024},
    }

    def __init__(self, n_actions: int, net_scale: str = "medium"):
        super().__init__()
        assert net_scale in self.SCALE_CONFIGS, \
            f"net_scale must be one of {list(self.SCALE_CONFIGS.keys())}"

        cfg = self.SCALE_CONFIGS[net_scale]
        f1, f2, f3 = cfg["filters"]
        hidden = cfg["hidden"]
        self.hidden_size = hidden

        # ── Convolutional feature extractor ──────────────────────────────
        self.conv = nn.Sequential(
            nn.Conv2d(4, f1, kernel_size=8, stride=4),   # (4,84,84) → (f1,20,20)
            nn.ReLU(),
            nn.Conv2d(f1, f2, kernel_size=4, stride=2),  # → (f2,9,9)
            nn.ReLU(),
            nn.Conv2d(f2, f3, kernel_size=3, stride=1),  # → (f3,7,7)
            nn.ReLU(),
        )

        # Compute flattened conv output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 84, 84)
            conv_out = int(np.prod(self.conv(dummy).shape[1:]))

        # ── Fully connected layers ────────────────────────────────────────
        self.fc_repr = nn.Sequential(
            nn.Linear(conv_out, hidden),
            nn.ReLU(),
        )
        self.fc_out = nn.Linear(hidden, n_actions)

        # ── Representation storage (populated by forward hook) ────────────
        self.representation: Optional[torch.Tensor] = None
        self._register_repr_hook()

        # ── Weight initialisation (orthogonal — more stable than default) ─
        self._init_weights()

    # ─────────────────────────────────────────────────────────────────────
    def _register_repr_hook(self):
        """
        Register a forward hook on fc_repr so that after every forward pass,
        self.representation holds the 512-dim activation tensor (detached,
        on CPU as numpy array for easy downstream use).
        """
        def hook(module, input, output):
            self.representation = output.detach().cpu()

        self.fc_repr.register_forward_hook(hook)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Final layer: smaller init → less overconfident initial Q-values
        nn.init.orthogonal_(self.fc_out.weight, gain=0.01)

    # ─────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 4, 84, 84) float tensor, pixel values in [0, 1]

        Returns:
            Q-values: (batch, n_actions)
        """
        x = self.conv(x)
        x = x.view(x.size(0), -1)   # Flatten
        x = self.fc_repr(x)          # ← hook fires here, stores representation
        return self.fc_out(x)

    # ─────────────────────────────────────────────────────────────────────
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_dead_neurons(self, activations: np.ndarray, threshold: float = 0.95) -> dict:
        """
        Count neurons that are inactive (output ≈ 0) more than `threshold`
        fraction of the time across a batch of activations.

        Args:
            activations: (N, hidden_size) numpy array
            threshold:   fraction of zeros above which neuron is "dead"

        Returns:
            dict with count, fraction, and indices of dead neurons
        """
        zero_fraction = (activations == 0).mean(axis=0)  # (hidden_size,)
        dead_mask = zero_fraction > threshold
        return {
            "dead_count": int(dead_mask.sum()),
            "dead_fraction": float(dead_mask.mean()),
            "dead_indices": np.where(dead_mask)[0].tolist(),
            "zero_fractions": zero_fraction,
        }


if __name__ == "__main__":
    for scale in ["small", "medium", "large"]:
        model = AtariCNN(n_actions=6, net_scale=scale)
        x = torch.zeros(4, 4, 84, 84)
        q = model(x)
        print(f"[{scale}] params={model.count_parameters():,} | "
              f"repr={model.representation.shape} | Q={q.shape}")

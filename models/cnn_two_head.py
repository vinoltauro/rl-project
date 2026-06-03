"""
Two-Head Atari CNN
==================
Shared backbone (conv + fc_repr) with two separate output heads,
one per game. Used for interleaved training where a single network
must serve both Pong and Breakout simultaneously.

Architecture:
    Input (4, 84, 84)
    → Conv layers (shared)
    → fc_repr 512-dim (shared)
    → fc_out_pong  (512 → 6)   ← Pong head
    → fc_out_breakout (512 → 4) ← Breakout head

The shared backbone receives gradients from whichever game is
currently being trained, allowing it to develop representations
that serve both tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


SCALES = {
    "small":  ([16, 32, 32],  256),
    "medium": ([32, 64, 64],  512),
    "large":  ([64, 128, 128], 1024),
}


class AtariCNNTwoHead(nn.Module):

    def __init__(self, net_scale: str = "medium"):
        super().__init__()
        filters, repr_dim = SCALES[net_scale]
        f1, f2, f3 = filters

        self.conv = nn.Sequential(
            nn.Conv2d(4, f1, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(f1, f2, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(f2, f3, kernel_size=3, stride=1), nn.ReLU(),
        )

        conv_out = f3 * 7 * 7
        self.fc_repr         = nn.Linear(conv_out, repr_dim)
        self.fc_out_pong     = nn.Linear(repr_dim, 6)
        self.fc_out_breakout = nn.Linear(repr_dim, 4)

        self.repr_dim    = repr_dim
        self.representation = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.fc_out_pong.weight,     gain=0.01)
        nn.init.orthogonal_(self.fc_out_breakout.weight, gain=0.01)

    def forward(self, x: torch.Tensor, game: str = "pong") -> torch.Tensor:
        x    = self.conv(x)
        x    = x.view(x.size(0), -1)
        repr = F.relu(self.fc_repr(x))
        self.representation = repr

        if game == "pong":
            return self.fc_out_pong(repr)
        else:
            return self.fc_out_breakout(repr)

    def n_actions(self, game: str) -> int:
        return 6 if game == "pong" else 4

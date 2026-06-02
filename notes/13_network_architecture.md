# 🏗️ Network Architecture — Visual Guide

> **One liner:** The network is a pipeline that turns raw game pixels into a decision. Each layer extracts increasingly abstract information until the final layer picks an action.

---

## 🔍 Zoom Level 1 — The Pub Explanation

Think of the network like a team of specialists processing a photo:

1. **Edge detector** — "I can see lines and shapes"
2. **Object detector** — "I can see a ball and a paddle"
3. **Pattern recogniser** — "I can see the ball is moving toward the top-left"
4. **Situation summariser** — "Based on everything, here is my 512-word summary of what is happening"
5. **Decision maker** — "Given that summary, move left"

Each layer hands its output to the next. By the time we reach the 512-dim layer, the raw pixels have been compressed into a compact, meaningful description of the game state.

---

## 🔬 Zoom Level 2 — The ASCII Diagram

```
INPUT
┌─────────────────────────────────┐
│  4 stacked grayscale frames     │
│  shape: (4, 84, 84)             │
│  = 28,224 numbers               │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  CONV LAYER 1                   │
│  32 filters, 8×8, stride 4      │
│  + ReLU activation              │
│  Output: (32, 20, 20)           │
│  Learns: edges, motion blur     │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  CONV LAYER 2                   │
│  64 filters, 4×4, stride 2      │
│  + ReLU activation              │
│  Output: (64, 9, 9)             │
│  Learns: shapes, objects        │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  CONV LAYER 3                   │
│  64 filters, 3×3, stride 1      │
│  + ReLU activation              │
│  Output: (64, 7, 7)             │
│  Learns: spatial relationships  │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  FLATTEN                        │
│  (64, 7, 7) → 3136 numbers      │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐  ← THIS IS WHAT WE ANALYSE
│  FC REPR LAYER  (fc_repr)       │
│  3136 → 512 units               │
│  + ReLU activation              │
│  Output: 512 numbers            │
│  = The agent's compressed       │
│    understanding of the state   │
│                                 │
│  ⚠️ Forward hook lives here    │
│  captures activations silently  │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  OUTPUT LAYER  (fc_out)         │
│  512 → N actions                │
│  Pong:    512 → 6 Q-values      │
│  Breakout: 512 → 4 Q-values     │
│  No activation (raw Q-values)   │
└─────────────────────────────────┘
```

---

## 🎓 Zoom Level 3 — The Professor Explanation

The architecture follows Mnih et al. (2015) exactly:
- Three convolutional layers with ReLU activations extract hierarchical visual features
- A 512-dimensional fully connected layer with ReLU serves as the representation bottleneck
- A linear output head maps representations to Q-values (one per action)

**Orthogonal initialisation** (Saxe et al., 2014) is used for all weight matrices. This produces better-conditioned gradient flow in early training compared to Xavier initialisation, leading to more stable early-phase learning.

**The forward hook mechanism:**
```python
def hook(module, input, output):
    model.representation = output.detach()

model.fc_repr.register_forward_hook(hook)
```
This captures `fc_repr` activations after every forward pass with zero overhead during training. During analysis, we run the agent in near-greedy mode (ε=0.05) for 5,000 steps and collect all 512-dim vectors.

**Why fc_out cannot be shared between games:**
Pong has 6 actions (NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE).
Breakout has 4 actions (NOOP, FIRE, RIGHT, LEFT).
The fc_out weight matrix has shape (n_actions × 512), so it cannot be transferred between games. This is why all cross-game transfer experiments keep the target game's fc_out head.

---

## 💡 The Interesting Bit

> The entire analysis of this project happens at `fc_repr` — one layer, 512 numbers. Everything else in the network is either feeding into it (conv layers) or reading from it (fc_out). The question "do agents trained on similar games develop similar representations?" is really asking: "do the conv layers compress Pong pixels and Breakout pixels into the same kind of 512-number summary?"

---

## 🔗 Parameter Counts

| Layer | Parameters |
|---|---|
| Conv 1 | 32 × (4×8×8) + 32 = 8,224 |
| Conv 2 | 64 × (32×4×4) + 64 = 32,832 |
| Conv 3 | 64 × (64×3×3) + 64 = 36,928 |
| fc_repr | 512 × 3136 + 512 = 1,606,144 |
| fc_out (Pong) | 6 × 512 + 6 = 3,078 |
| fc_out (Breakout) | 4 × 512 + 4 = 2,052 |
| **Total (Pong)** | **~1,687,206** |
| **Total (Breakout)** | **~1,686,180** |

> Note: fc_repr dominates — 95% of parameters are in that one layer.

---

## 🔗 How it connects
- [[03_representations]] — what fc_repr actually encodes
- [[05_cka]] — CKA computed at each conv layer + fc_repr
- [[08_experiments]] — mix-and-match uses conv layer boundaries

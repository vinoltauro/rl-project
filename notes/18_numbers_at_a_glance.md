# 🔢 Numbers at a Glance

> Every key number from every experiment in one place. Memorise the bold ones.

---

## Training Results (Baseline 4 Agents)

| Agent | Episodes | Final Reward | Final Q | Peak Q |
|---|---|---|---|---|
| DQN / Pong | 2,434 | **7.12 ± 5.22** | 2.64 | 2.95 |
| DDQN / Pong | 2,457 | **7.50 ± 5.03** | 2.05 | 2.48 |
| DQN / Breakout | 144,751 | **3.86 ± 4.94** | 4.06 | 5.39 |
| DDQN / Breakout | 144,948 | **4.00 ± 5.04** | 4.13 | 4.55 |

**Key number to remember:** DQN and DDQN score nearly the same. DQN's Q-values are ~29% higher on Pong (overestimation).

---

## Q-Value Overestimation

| Comparison | DQN | DDQN | Difference |
|---|---|---|---|
| Pong final Q | 2.64 | 2.05 | **~29% higher** |
| Pong peak Q | 2.95 | 2.48 | ~19% higher |
| Breakout peak Q | **5.39** | 4.55 | ~18% higher |

---

## Zero-Shot Mix-and-Match (50 episodes)

### Breakout evaluation
| Condition | Mean Reward |
|---|---|
| Native DQN/Breakout | **3.40 ± 3.82** |
| DQN/Pong conv + Breakout upper | 0.44 ± 0.78 |
| DDQN/Pong conv + Breakout upper | 0.80 ± 1.36 |
| Random conv + Breakout upper | 0.40 ± 1.00 |
| Random network | **0.42 ± 0.80** |

### Pong evaluation
| Condition | Mean Reward |
|---|---|
| Native DQN/Pong | **9.54 ± 5.08** |
| DQN/Breakout conv + Pong upper | -20.72 ± 0.45 |
| DDQN/Breakout conv + Pong upper | **-21.00 ± 0.00** |
| Random conv + Pong upper | -21.00 ± 0.00 |
| Random network | -21.00 ± 0.00 |

**Key number to remember:** Pong conv on Breakout = 0.44 (same as random 0.40). Breakout conv on Pong = -21 (worst possible).

---

## Backbone Training (5M steps on Breakout)

| Condition | Final Reward | Best Smoothed |
|---|---|---|
| Scratch DQN/Breakout | 3.86 | **4.07** |
| Frozen conv (Pong) | **2.28** | 3.03 |
| Full fine-tune from Pong | 3.12 | **4.47** |

**Key number to remember:** Frozen conv = 2.28 (below scratch 3.86). Full fine-tune peaks at 4.47 (above scratch).

---

## Sequential Training (Preliminary — Smoke Test Only)

| Step | Pong Reward |
|---|---|
| 0 (start) | ~7.50 |
| 15,000 Breakout steps | **-8.85** |

**Key number to remember:** Pong reward dropped from +7.5 to -8.85 in just 15k steps.

---

## Network Size

| | Value |
|---|---|
| Input | 4 × 84 × 84 = 28,224 pixels |
| fc_repr | **512 dimensions** |
| Total parameters (Pong) | ~1,687,206 |
| fc_repr share of params | **~95%** |
| Training steps (Pong) | 2,000,000 |
| Training steps (Breakout) | 5,000,000 |
| Checkpoints saved | Every 500,000 steps |

---

## Training Hyperparameters (Quick Reference)

| Parameter | Value |
|---|---|
| Learning rate | **1e-4** |
| Replay buffer | **100,000** transitions |
| Batch size | 32 |
| Gamma | 0.99 |
| Target update | Every 1,000 steps |
| Epsilon start → end | **1.0 → 0.01** over 100k steps |
| Gradient clip | L2 norm ≤ 10 |
| Random seed | **42** |

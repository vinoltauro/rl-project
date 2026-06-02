# 🤖 DQN and Double DQN

> **One liner:** DQN is an AI that learns to play games by trial and error. DDQN is a small fix that makes it less overconfident — and that small fix turns out to matter a lot for what the agent learns internally.

---

## 🔍 Zoom Level 1 — The Pub Explanation

Imagine you are learning to play poker by keeping notes on every hand you play. You write down the situation, what you did, and how it turned out. Over time you get better at guessing which moves are good.

DQN does exactly that. It plays millions of game frames, keeps a memory of what happened (the replay buffer), and gradually learns which actions lead to more reward.

**The overconfidence problem:** DQN has a habit of being too optimistic. It asks itself "what is the best move here?" and then overestimates how good that move actually is. This is like a poker player who always thinks their hand is better than it is.

**DDQN's fix:** Instead of one person both choosing the best move and judging how good it is, DDQN uses two slightly different versions of itself — one to choose the move, one to evaluate it. Because they disagree slightly, they cancel out each other's overconfidence.

The fix is literally **three lines of code**. Yet it changes how the agent thinks internally.

---

## 🔬 Zoom Level 2 — The Study Group Explanation

Both DQN and DDQN approximate the **Q-function** Q(s, a) — the expected future reward of taking action a in state s. They use a convolutional neural network to process raw pixel inputs.

**Two key mechanisms both use:**
1. **Experience replay** — store transitions (s, a, r, s') in a buffer, sample randomly to break temporal correlations
2. **Target network** — a frozen copy of the network used to compute training targets, updated every 1000 steps

**Where they differ — the training target:**

DQN:
```
y = r + γ · max_a' Q(s', a'; θ⁻)
```
The same (frozen) network both selects the best action AND evaluates it. This double use of the max operator causes systematic overestimation.

DDQN:
```
y = r + γ · Q(s', argmax_a' Q(s', a'; θ); θ⁻)
```
The online network selects the best action. The target network evaluates it. Two different networks, so their errors don't compound.

**In our codebase:** `DDQNAgent` inherits everything from `DQNAgent` and only overrides the `learn()` method. The difference is three lines.

---

## 🎓 Zoom Level 3 — The Professor Explanation

DQN (Mnih et al., 2015) introduced two stabilisation mechanisms — experience replay and target networks — that made Q-learning tractable for raw pixel inputs. However, the max operator in the Bellman target causes a systematic positive bias: because the same network both selects and evaluates the greedy action, estimation errors compound upward.

Van Hasselt et al. (2016) addressed this with Double DQN, decoupling action selection (online network θ) from action evaluation (target network θ⁻). The resulting targets are substantially less biased.

In this study, DQN Q-values on Pong peak at 2.95 and settle at 2.64; DDQN peaks at 2.48 and settles at 2.05 — a ~29% difference in final Q-value, despite near-identical task performance. This overestimation inflates gradient noise in DQN, which is the direct mechanism behind its higher dead neuron fraction and less structured representations.

---

## 💡 The Interesting Bit

> DQN and DDQN score almost identically on both games. But DDQN develops **clearly better internal representations** — tighter clusters, fewer dead neurons, more focused attention. This shows that **reward alone does not tell you how well an agent has learned**. Two agents can score the same and have completely different quality of understanding internally.

---

## 🔗 How it connects

- [[03_representations]] — how the overestimation affects the representation layer
- [[06_dead_neurons]] — DQN's noisier gradients cause more dead neurons
- [[09_key_findings]] — performance vs representation quality decoupling

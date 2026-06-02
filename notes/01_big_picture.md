# 🌍 The Big Picture

> **One liner:** We trained AI agents to play two Atari games and looked inside their brains to understand what they actually learned — and what happens when you force them to switch games.

---

## 🔍 Zoom Level 1 — The Pub Explanation

Imagine you trained two people to play table tennis and snooker. Both games have a ball and require hand-eye coordination. But one is about reacting fast to an opponent, and the other is about planning your next three shots.

Now you look at brain scans of both players. Do they develop similar neural patterns just because both games involve a ball? Or does the strategic difference mean their brains are actually doing something completely different?

That is exactly what this project does — except the "players" are AI agents, and instead of brain scans we analyse a mathematical layer inside the neural network.

---

## 🔬 Zoom Level 2 — The Study Group Explanation

We trained four AI agents using deep reinforcement learning:
- DQN on Pong
- DQN on Breakout
- DDQN on Pong
- DDQN on Breakout

After training, we extracted the **512-dimensional activation vector** at the penultimate layer — this is the agent's compressed internal encoding of the game state. We then asked:

- Do agents trained on similar games develop similar internal encodings?
- Does the algorithm (DQN vs DDQN) affect the quality of those encodings?
- Can you take representations learned on one game and use them on another?
- What happens to the representations when you force an agent to learn a second game?

---

## 🎓 Zoom Level 3 — The Professor Explanation

This study uses a 2×2 factorial design crossing algorithm (DQN, DDQN) with game (Pong, Breakout) to investigate representational geometry in deep RL agents. The penultimate 512-dimensional fully connected layer serves as the representation space. Analysis tools include t-SNE for visualisation, CKA for layer-wise similarity, Grad-CAM for saliency, and dead neuron analysis for capacity measurement.

The broader context is **continual learning** — understanding what happens to internal representations when an agent trained on one task is exposed to a second task. This connects to the catastrophic forgetting literature and the capacity loss work of Lyle et al. (2022).

---

## 💡 The Interesting Bit

> The games look visually similar — both have a ball and paddle — but the agents develop **completely different internal representations**. What matters is not what the pixels look like but what is **reward-predictive** in each game. In Pong that is the opponent's paddle. In Breakout it is the brick layout. The network learns to encode what is useful, not what is visually present.

---

## 🔗 How it connects

- [[02_dqn_ddqn]] — what the two algorithms actually are
- [[03_representations]] — what the 512-dim layer means
- [[08_experiments]] — all the experiments that came out of this
- [[09_key_findings]] — what we actually found

# 📖 The Story Arc — The Whole Project in 2 Minutes

> Read this before a supervisor meeting. It connects everything into one narrative.

---

## The Setup

We picked two Atari games — Pong and Breakout — that look visually similar but demand completely different strategies. We trained four agents: DQN and DDQN on each game. The question: do similar-looking games produce similar-looking internal representations?

---

## Chapter 1 — The Baseline (What agents learn)

We trained all four agents and extracted their 512-dimensional internal representations at every 500k training steps. When we projected these onto a 2D map (t-SNE), the answer was immediate: **the game you play matters more than the algorithm you use.** Two big clusters — Pong and Breakout — with DQN/DDQN sub-clusters within each.

But DDQN consistently produces cleaner, more structured representations within each game — tighter clusters, more focused attention, fewer dead neurons — even though DQN and DDQN score nearly identically. **Performance and representation quality are decoupled.**

---

## Chapter 2 — Where Does Similarity Break Down? (CKA)

The t-SNE analysis was qualitative. We wanted to know *where exactly* in the network the divergence happens. CKA analysis at each layer depth gave us the answer: early convolutional layers are highly similar across games (they detect the same visual features — ball, motion, edges), but the representation layer diverges sharply. **The boundary between generalisation and specialisation is at the conv-to-fc transition.**

---

## Chapter 3 — Can You Swap Parts? (Mix-and-Match)

If conv layers are similar, can you just plug Pong's conv into Breakout's agent and get something useful? We tried — 10 conditions, 50 episodes each, zero additional training.

The answer was unambiguous: **no.** Pong conv in Breakout scores the same as random initialisation. Breakout conv in Pong scores -21 — the worst possible score. CKA similarity does not translate to functional compatibility. The conv and fc_repr layers are co-adapted during training. Foreign conv activations, even if geometrically similar, completely confuse the upper layers.

---

## Chapter 4 — What If You Allow Training? (Backbone)

Zero-shot failed. But what if you freeze Pong's conv layers and just train the upper layers on Breakout? Or load all Pong weights and let everything adapt?

- **Frozen conv underperformed scratch** — fixed Pong features slightly constrain Breakout learning
- **Full fine-tune from Pong outperformed scratch at peak** — Pong weights are a useful warm start when everything can adapt

The pre-training only helps when you give the network freedom to reorganise. A frozen backbone is too rigid.

---

## Chapter 5 — What Happens When You Force a Switch? (Sequential Training — ongoing)

This is the continual learning chapter. We take the trained Pong agent and force it to train on Breakout. Every 200k steps we measure three things: Pong reward (is it forgetting?), dead neurons (is capacity being lost?), and CKA drift (how far have the representations moved?).

Preliminary results: Pong reward collapsed from +7.5 to -8.85 in just 15,000 Breakout steps. **Forgetting is fast.** We are now measuring whether freezing conv layers mitigates this, and whether the dead neuron increase and representation drift happen in lockstep with performance collapse.

---

## The Punchline

> Early conv layers learn general visual features that are similar across structurally related games. But this similarity is geometric, not functional — the co-adaptation between conv and fc_repr layers means the features cannot be reused without retraining. When an agent is forced to switch games, it forgets catastrophically and fast. Freezing early layers reduces but does not eliminate forgetting, because the representation layer also forgets independently.

---

## One Sentence for Each Experiment

| Experiment | One sentence |
|---|---|
| Baseline | Game shapes representations more than algorithm, but algorithm shapes quality. |
| CKA | Early layers generalise, the representation layer specialises — and we can pinpoint exactly where. |
| Mix-and-match | Similar features cannot be swapped without retraining — the interface is too tightly coupled. |
| Backbone | Pre-training helps only when all layers are free to adapt. |
| Sequential | Forgetting is fast, and we are measuring its three simultaneous signatures. |

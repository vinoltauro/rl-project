# 🏆 Key Findings

> All confirmed results so far. Updated as experiments complete.

---

## Finding 1 — Game effect dominates algorithm effect

> [!important]
> t-SNE projections separate cleanly by game regardless of algorithm. The primary organising principle of the representation space is WHAT game was played, not HOW it was trained.

**What this means in plain English:** If you look at the internal representations of all four agents together, the first thing you see is two clusters — Pong and Breakout. DQN and DDQN are sub-clusters within each game, not separate top-level clusters.

---

## Finding 2 — Representational hierarchy confirmed by CKA

> [!important]
> CKA is high at conv1/conv2, drops at conv3, drops sharply at fc_repr. Early layers generalise across games. The representation layer specialises to game-specific strategy.

**What this means in plain English:** The first two convolutional layers essentially learned the same visual features for both games — ball detection, motion, edges. But by the time you reach the 512-dim layer, the two agents have diverged substantially. The divergence happens at the boundary between "seeing" and "deciding."

---

## Finding 3 — Representational similarity ≠ functional transferability

> [!important]
> Despite high CKA at conv layers, zero-shot layer transplantation completely fails. Pong conv in Breakout agent scores 0.44 (same as random 0.40). Breakout conv in Pong agent scores -21.00 (worst possible).

**What this means in plain English:** The fact that two networks learned similar-looking features does not mean you can swap their parts. The conv and fc_repr layers are co-adapted — the fc_repr was trained specifically to interpret its own conv's activations, not someone else's.

---

## Finding 4 — Frozen backbone hurts, warm start helps

> [!important]
> Frozen Pong conv underperforms scratch (2.28 vs 3.86 final reward). Full fine-tune from Pong weights achieves best peak (4.47 smoothed). Pre-training is only useful when all layers can adapt.

**What this means in plain English:** Freezing Pong's conv layers does not give Breakout a head start — it actually slows it down. But loading the Pong weights as a starting point and letting everything adapt does help.

---

## Finding 5 — DDQN produces better representations despite similar scores

> [!important]
> DQN and DDQN score similarly on both games. But DDQN has tighter t-SNE clusters, fewer dead neurons, more focused saliency, and higher cross-game cosine similarity.

**What this means in plain English:** You cannot judge the quality of an agent's internal understanding just by its score. Two agents can play equally well while one has built a much cleaner internal model than the other.

---

## Finding 6 — DQN overestimates Q-values by 10–40%

> [!important]
> DQN final Q on Pong: 2.64. DDQN: 2.05. ~29% difference. On Breakout DQN peaks at 5.39 vs DDQN's 4.55.

**What this means in plain English:** DQN is systematically overconfident about how good its actions are. DDQN's fix brings it closer to reality. The overconfidence creates noisier training signals, which is the root cause of the dead neuron and representation quality differences.

---

## Finding 7 — Forgetting is fast 🔄 (preliminary)

> [!note] Preliminary — from smoke test only
> After just 15,000 Breakout training steps, Pong reward dropped from ~7.5 to -8.85. Forgetting appears to be rapid, not gradual.

**What this means in plain English:** The agent does not slowly forget Pong as it learns Breakout. It forgets very quickly. This will be confirmed by the full sequential training experiment.

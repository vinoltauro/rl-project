# 🧪 Experiments — What, Why, What We Found

> Each experiment is explained at three levels. Results marked 🔄 are still pending.

---

## Experiment 1 — Baseline 4-Agent Study

### What we did
Trained DQN and DDQN on Pong (2M steps) and Breakout (5M steps). Extracted 512-dim representations at every 500k checkpoint. Ran t-SNE, Grad-CAM saliency, dead neuron analysis, and cosine similarity.

### Why
To establish the baseline facts: do agents playing similar games develop similar representations? Does algorithm choice matter?

### What we found

| Finding | Result |
|---|---|
| Game effect | Representations separate clearly by game, not algorithm |
| Algorithm effect | DDQN produces tighter, more structured clusters |
| Dead neurons | DQN has more dead neurons than DDQN |
| Q-value overestimation | DQN overestimates by 10–40% vs DDQN |
| Cross-game cosine similarity | Non-trivial — shared ball/paddle features persist |
| Performance vs representation | Decoupled — both score similarly, DDQN represents better |

### What it means
The game you play shapes your internal representations more than the algorithm you use. But algorithm quality still matters for representation quality, even when it doesn't matter for score.

---

## Experiment 2 — CKA Layer Similarity

### What we did
Registered forward hooks at conv1, conv2, conv3, and fc_repr. Collected activations for Pong and Breakout agents on the same game states. Computed pairwise CKA at each depth.

### Why
To find exactly WHERE in the network the representations diverge — which layer is responsible for game-specific vs general features.

### What we found
- conv1, conv2: high CKA — early layers learn the same visual features regardless of game
- conv3: starts to drop
- fc_repr: drops sharply — this layer is game-specific

### What it means
The network has a hierarchy: early layers generalise, the representation layer specialises. The boundary is at the conv→fc transition.

---

## Experiment 3 — Zero-Shot Mix-and-Match

### What we did
Built chimera networks by transplanting conv layers from one game's agent into another game's upper layers. Evaluated performance with NO additional training across 10 conditions (5 per game), 50 episodes each.

### Why
If conv layers are similar (as CKA shows), can you just swap them between agents? This tests whether CKA similarity translates to functional transferability.

### What we found

| Condition | Breakout reward | Pong reward |
|---|---|---|
| Native agent | 3.40 | 9.54 |
| Cross-game conv | 0.44 | -20.72 |
| Random conv | 0.40 | -21.00 |
| Random network | 0.42 | -21.00 |

### What it means
**Representational similarity ≠ functional compatibility.** Even though conv layers are geometrically similar (high CKA), the fc_repr layer was trained to interpret its own conv stack's specific activation patterns. Foreign conv features break it completely. On Pong, any foreign conv layer produces the worst possible score.

---

## Experiment 4 — Frozen Backbone Training

### What we did
Loaded DQN/Pong checkpoint. Trained on Breakout for 5M steps under two conditions:
- `--freeze conv`: Pong conv layers locked, only fc layers train
- `--freeze none`: everything trains (full fine-tune from Pong weights)

Compared against scratch DQN/Breakout baseline.

### Why
If zero-shot swapping fails, can you at least use Pong conv as a fixed feature extractor and just train the upper layers on Breakout?

### What we found

| Condition | Final Reward | Best Smoothed |
|---|---|---|
| Scratch | 3.86 | 4.07 |
| Frozen conv | 2.28 | 3.03 |
| Full fine-tune | 3.12 | 4.47 |

### What it means
Frozen conv **hurts** Breakout learning — the fixed Pong features are close but not optimal for Breakout. Full fine-tune achieves the best peak because it uses Pong weights as a warm start while allowing full adaptation.

---

## Experiment 5 — Sequential Training (Forgetting Measurement) 🔄

### What we doing
Load DQN/Pong checkpoint. Train on Breakout for 2M steps. Every 200k steps measure:
1. Pong reward (chimera eval)
2. Dead neuron fraction
3. CKA drift from original Pong representations

Two conditions: `--freeze none` (currently running), `--freeze conv` (after).

### Why
This is the core continual learning experiment. Directly measures catastrophic forgetting and its mechanisms.

### What we expect to find
- Pong reward drops fast (smoke test showed -8.85 after just 15k steps)
- Dead neurons increase as Breakout training progresses
- CKA drift accelerates early then plateaus
- Frozen conv reduces but does not eliminate forgetting (forgetting happens at fc_repr level)

### Status
🔄 `--freeze none` currently running (~5 hours remaining)

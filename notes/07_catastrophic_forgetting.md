# 🔥 Catastrophic Forgetting and Continual Learning

> **One liner:** When an AI learns a second task, it often completely destroys what it learned about the first task — this is catastrophic forgetting, and it is the central problem of continual learning.

---

## 🔍 Zoom Level 1 — The Pub Explanation

Imagine you spent six months becoming a really good chess player. Then you spend six months learning to play Go instead. If you are a normal human, you can still play chess reasonably well — you have not forgotten it.

Now imagine you are an AI. You spend six months learning chess. Then you switch to learning Go. The new training completely overwrites the chess knowledge. When someone asks you to play chess again, you have completely forgotten how. You are back to square one.

This is **catastrophic forgetting** — and it is a fundamental problem with neural networks. The weights that encode chess knowledge get modified by Go training until nothing useful for chess remains.

**Why does this not happen to humans?** We have specialised memory systems, sleep consolidation, and our brain does not need to update the same neurons for every new skill. Neural networks, by contrast, use the same weights for everything.

---

## 🔬 Zoom Level 2 — The Study Group Explanation

In our setup, catastrophic forgetting looks like this:

1. Train DQN on Pong for 2M steps → agent learns to play Pong well (reward ≈ +7)
2. Continue training on Breakout for 2M steps → agent adapts to Breakout
3. Evaluate on Pong again → Pong reward has likely dropped dramatically

The forgetting happens because:
- Breakout's gradient signal updates the weights to be useful for Breakout
- Those weight changes overwrite the Pong-specific patterns
- The representation layer (fc_repr) reorganises around Breakout's strategic demands
- The old Pong-specific encoding is gone

**How we measure it:**
Every 200k Breakout training steps, we build a **chimera network**: take the current conv+fc_repr weights (now partially trained on Breakout) and attach the original Pong fc_out head. Evaluate this chimera on Pong. If the Pong reward stays high, Pong knowledge is preserved. If it drops, forgetting is occurring.

**The three signals we track:**
1. Pong reward (direct performance measure)
2. Dead neurons (capacity being lost)
3. CKA drift (how far representations have moved from the Pong starting point)

---

## 🎓 Zoom Level 3 — The Professor Explanation

Catastrophic forgetting (McCloskey & Cohen, 1989; Ratcliff, 1990) refers to the abrupt loss of previously learned information upon learning new information in artificial neural networks. Unlike biological neural systems, ANNs use shared weights across tasks — gradient updates for Task B directly interfere with the weight configurations that supported Task A.

In the RL setting, this manifests as performance collapse on the original task when training continues on a new task. The mechanisms include:

1. **Weight interference** — gradients for Task B overwrite Task A-specific weight configurations
2. **Representational drift** — the embedding space reorganises around Task B's reward signal
3. **Capacity loss** — neurons active for Task A enter the ReLU dead zone under Task B's gradient signal

**Mitigations studied in this project:**
- **Frozen conv layers** — prevent the convolutional features from being overwritten. We found this is insufficient because forgetting still occurs at the fc_repr level.
- **Full fine-tune from Pong initialisation** — allows the network to adapt while using Pong weights as a warm start. Performs best on Breakout but also forgets Pong.

**Connection to the literature:**
- Elastic Weight Consolidation (Kirkpatrick et al., 2017) — penalise changes to weights important for Task A
- Continual backpropagation (Elsayed & Mahmood, 2024) — reinitialise low-utility neurons to maintain plasticity
- Progressive Neural Networks (Rusu et al., 2016) — add new columns for new tasks, never modify old ones

---

## 💡 The Interesting Bit

> The smoke test (15k steps) already showed Pong reward dropping from ~7.5 to -8.85 in just 15,000 Breakout training steps. Forgetting is not gradual — it appears to happen very fast. The question our full experiment will answer is: **does it happen all at once (sudden collapse) or gradually (steady decline)?** The shape of the forgetting curve is scientifically interesting and nobody told us to measure it — we added it ourselves.

---

## 🔗 How it connects

- [[06_dead_neurons]] — the neuron-level mechanism of forgetting
- [[05_cka]] — CKA drift as a continuous measure of forgetting
- [[08_experiments]] — the sequential training experiment
- [[10_professor_questions]] — the professor's main research interest

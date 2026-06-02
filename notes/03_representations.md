# 🧬 Representations — What the 512-dim Layer Actually Is

> **One liner:** A representation is the AI's compressed internal summary of what is happening in the game right now — everything the agent "knows" about the current moment packed into 512 numbers.

---

## 🔍 Zoom Level 1 — The Pub Explanation

When you look at a game of Pong, your brain does not process every single pixel. Instead it instantly extracts the important stuff — where is the ball, how fast is it moving, where is my paddle. Your brain compresses the whole visual scene into a handful of useful facts.

The neural network does the same thing. The raw input is an 84×84 pixel image (4 frames stacked). By the time it reaches the second-to-last layer of the network, that has been compressed into **512 numbers**. Those 512 numbers are the agent's internal summary of the current situation.

We call this the **representation** or the **embedding**. It is what the agent actually "thinks" about — not the raw pixels, but this compressed internal description.

**The key question of this project:** do agents playing different games end up with similar-looking 512-number summaries, or completely different ones?

---

## 🔬 Zoom Level 2 — The Study Group Explanation

The network architecture is:

```
Input: (4, 84, 84) stacked grayscale frames
→ Conv layer 1: 32 filters, 8×8, stride 4  → extracts edges, motion
→ Conv layer 2: 64 filters, 4×4, stride 2  → extracts shapes, objects
→ Conv layer 3: 64 filters, 3×3, stride 1  → extracts complex patterns
→ Flatten
→ FC layer (512 units) + ReLU             ← THIS IS THE REPRESENTATION LAYER
→ Output: Q-values for each action
```

The 512-dim layer is called `fc_repr` in the codebase. A **PyTorch forward hook** is registered on it — this means every time the network processes a frame, the 512-dim vector is automatically captured without interfering with training.

After training, we run the agent for 5,000 steps and collect all 512-dim vectors. These form a cloud of points in 512-dimensional space. We then use t-SNE to project them down to 2D so we can visualise the structure.

---

## 🎓 Zoom Level 3 — The Professor Explanation

The penultimate fully connected layer serves as a 512-dimensional state embedding. This is the bottleneck through which all game-state information must pass before Q-value computation. Its geometry captures what the agent has learned to treat as similar (nearby points) and different (distant points).

Zahavy et al. (2016) showed that t-SNE projections of DQN representations reveal semantically meaningful clustering. This study extends that analysis to a controlled multi-game, multi-algorithm comparison using both t-SNE and CKA.

The representation layer is architecturally identical across all four agents — same dimensions, same position in the network, same initialisation. Any observed differences in representational geometry are therefore attributable purely to training dynamics (game content and algorithm choice), not architectural differences.

**In code:** `model.fc_repr` captures the representation. A forward hook auto-populates `model.representation` after every forward pass:
```python
def hook(module, input, output):
    model.representation = output.detach()
model.fc_repr.register_forward_hook(hook)
```

---

## 💡 The Interesting Bit

> The same 512-dim layer plays two completely different roles depending on which game trained it. In Pong it encodes "where is the opponent's paddle relative to mine and where is the ball going." In Breakout it encodes "what is the remaining brick layout and where should the ball be directed." The architecture is identical. The training signal is what shapes what gets stored in those 512 numbers.

---

## 🔗 How it connects

- [[04_tsne]] — how we visualise these 512-dim vectors
- [[05_cka]] — how we measure similarity between representations
- [[06_dead_neurons]] — how many of these 512 neurons go dead
- [[07_catastrophic_forgetting]] — what happens to these 512 numbers when you switch games

# 📐 CKA — Centred Kernel Alignment

> **One liner:** CKA is a number between 0 and 1 that tells you how similar two networks are at a specific layer — 1 means identical structure, 0 means completely different.

---

## 🔍 Zoom Level 1 — The Pub Explanation

Imagine two people both learned to recognise dogs. You show them both the same 100 photos and ask them to rate each photo from 0 to 10 for "how dog-like is this?" If their ratings are similar, they have developed similar internal ideas about dogs. If their ratings are completely different, they have developed different internal models.

CKA does exactly this, but for neural network layers. You show both networks the same set of game states. You look at the activations at a specific layer. You then ask: do these two networks respond similarly to the same inputs at this layer?

The answer is a single number: 0 (completely different) to 1 (identical).

---

## 🔬 Zoom Level 2 — The Study Group Explanation

CKA (Kornblith et al., 2019) compares two activation matrices:
- X: activations from network 1 on n inputs (shape: n × 512)
- Y: activations from network 2 on the same n inputs (shape: n × 512)

It computes the similarity between the **Gram matrices** (XX^T and YY^T) after centering:

```
CKA(X, Y) = ||Y^T X||²_F / (||X^T X||_F · ||Y^T Y||_F)
```

**Why this is better than just comparing weights or cosine similarity:**
- It is invariant to rotation and scaling — if one network learned the same features in a different order or at a different scale, CKA still says they are similar
- It captures the full distributional geometry, not just the mean vector
- It is computed on the same inputs, so it measures functional similarity

**What our CKA plot shows (`layer_similarity_cka.png`):**
We compute CKA at four depths for DQN/Pong vs DQN/Breakout (and DDQN/Pong vs DDQN/Breakout):
- conv1 → high CKA (both networks respond similarly to edges, motion)
- conv2 → still high
- conv3 → starts to drop
- fc_repr → drops sharply

---

## 🎓 Zoom Level 3 — The Professor Explanation

Linear CKA is defined as:
```
CKA(X, Y) = HSIC(X, Y) / sqrt(HSIC(X, X) · HSIC(Y, Y))
```
where HSIC is the Hilbert-Schmidt Independence Criterion with a linear kernel. Centring is applied via the centring matrix H = I - (1/n)11^T.

CKA is invariant to orthogonal transformations and isotropic scaling, making it appropriate for comparing representations across independently trained networks — unlike direct weight comparison, which would conflate representational similarity with weight permutation.

The depth profile in our results shows CKA ≈ 0.8+ at conv1/conv2, dropping to ~0.3 at fc_repr. This quantifies the representational hierarchy hypothesised from qualitative t-SNE analysis: early layers generalise across games, the representation layer specialises to game-specific strategy.

---

## 💡 The Interesting Bit

> CKA reveals something that t-SNE cannot: the **depth at which divergence happens**. You can see in the plot that the two agents' representations are almost identical at conv1 and conv2 — they have genuinely learned the same visual features. But by fc_repr they have diverged substantially. The divergence is not a gradual drift — it accelerates at the transition from conv to fully connected layers. That is the exact moment where "seeing pixels" becomes "making decisions."

---

## 🔗 How it connects

- [[03_representations]] — what we are measuring CKA on
- [[08_experiments]] — the CKA experiment and the mix-and-match experiment
- [[09_key_findings]] — CKA confirms the layer hierarchy
- [[07_catastrophic_forgetting]] — we use CKA drift to track forgetting in real time

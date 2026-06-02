# 📊 t-SNE — How to Read the Plots

> **One liner:** t-SNE takes thousands of 512-dimensional points and squashes them onto a 2D map while trying to keep similar points near each other — so you can actually see the structure with your eyes.

---

## 🔍 Zoom Level 1 — The Pub Explanation

Imagine you have 5,000 cities in the world and you want to draw them on a flat map. The real world is 3D (it is a sphere), but a map is 2D. You have to squash it somehow, and some distances will be distorted — but cities that are close together on Earth should still be close together on the map.

t-SNE does the same thing, but instead of going from 3D to 2D, it goes from **512D to 2D**. Each point is one game state (one moment in the game). Points that the agent "thinks are similar" (similar 512-dim vectors) end up near each other on the 2D map.

**How to read our t-SNE plots:**
- Each dot is one game state
- Colour = which agent/game/algorithm produced it
- Tight clusters = the agent has clear, structured internal categories
- Messy overlapping blobs = noisy, unstructured representations

---

## 🔬 Zoom Level 2 — The Study Group Explanation

t-SNE (t-distributed Stochastic Neighbour Embedding) works in two steps:

1. In the original high-dimensional space (512D), compute the probability that two points are "neighbours" using a Gaussian distribution
2. In the 2D projection space, compute similar probabilities using a t-distribution (heavier tails to handle the "crowding problem")
3. Minimise KL divergence between the two distributions — meaning the 2D layout tries to match the neighbourhood structure of the 512D original

**Important caveat:** t-SNE preserves **local structure** (nearby points stay nearby) but does NOT preserve global distances. Two separate clusters on a t-SNE plot might not actually be far apart in 512D space. This is why we also use cosine similarity and CKA for quantitative comparisons.

**What our plots show:**
- `tsne_game_effect_dqn.png` — DQN Pong vs DQN Breakout: do they cluster separately?
- `tsne_algo_effect_pong.png` — DQN Pong vs DDQN Pong: does algorithm affect structure?
- `tsne_all_agents.png` — all four agents together: what is the dominant organising principle?
- `tsne_temporal_*.png` — how the representation evolves across training checkpoints
- `tsne_by_reward.png` — are high-reward states spatially separated from low-reward states?

---

## 🎓 Zoom Level 3 — The Professor Explanation

t-SNE (Van der Maaten & Hinton, 2008) minimises KL divergence between Gaussian neighbourhood distributions in the original space and Student-t distributions in the 2D projection. The Student-t distribution in the low-dimensional space has heavier tails than a Gaussian, which alleviates the crowding problem — the tendency of dissimilar points to be forced together in low dimensions.

Perplexity (set to 30 in our implementation) controls the effective number of neighbours. The algorithm is sensitive to this hyperparameter and is run with multiple initialisations for stability.

**Limitations for this study:**
- t-SNE does not preserve inter-cluster distances — visual separation does not imply quantitative dissimilarity
- Cluster compactness is assessed visually, which is subjective
- This is why we complement t-SNE with silhouette scores and Davies-Bouldin index in the 512-dim space for quantitative backing

---

## 💡 The Interesting Bit

> The t-SNE plots immediately show that **game identity dominates algorithm identity** as the organising principle. When you plot all four agents together, you see two big clusters — one for Pong, one for Breakout — with DQN/DDQN sub-structure within each. This is visible at a glance before any numbers are computed. It is probably the single most compelling image in the whole project.

---

## 🔗 How it connects

- [[03_representations]] — the 512-dim vectors that t-SNE is visualising
- [[05_cka]] — the quantitative complement to t-SNE's qualitative pictures
- [[09_key_findings]] — the game effect dominates finding

# 🎓 How to Talk to Your Professor

> Confident, concise answers to questions your supervisor is likely to ask.

---

## "What is your research question?"

"We are studying how structurally similar but strategically different games shape the internal representations of deep RL agents, and what happens to those representations when the agent is forced to switch games. The broader context is continual learning — understanding and mitigating the natural capacity loss that occurs during sequential training."

---

## "What have you found so far?"

"Three main findings. First, game content dominates algorithm choice as the organising principle of the representation space — agents playing different games develop distinct encodings even if they share visual structure. Second, CKA analysis confirms a representational hierarchy: early convolutional layers generalise across games, while the representation layer specialises. Third, this similarity does not translate to functional transferability — zero-shot layer swapping completely fails, and frozen conv backbones actually hurt cross-game learning. Only a full fine-tune from a pre-trained initialisation provides a benefit."

---

## "What is the continual learning angle?"

"We are currently running the sequential training experiment — training on Pong then continuing on Breakout, measuring forgetting every 200k steps via three signals: Pong reward, dead neuron fraction, and CKA drift from the original Pong representations. Preliminary results suggest forgetting is rapid rather than gradual. We are testing whether freezing conv layers mitigates forgetting, and we predict it will reduce but not eliminate it because forgetting also occurs at the fc_repr level."

---

## "Why Pong and Breakout specifically?"

"They sit at an interesting point on the similarity spectrum — visually similar enough that you might expect representational overlap (same ball, paddle, physics), yet strategically different enough that divergence would not be surprising. This makes them a useful pair for probing exactly where shared structure ends and game-specific structure begins. They are also both well-established Atari benchmarks with known training dynamics."

---

## "What is the significance of the CKA result?"

"It localises the transition from general to specific representations to a specific layer. The conv layers learn visual features — ball detection, motion, edges — that are game-agnostic. The fc_repr layer encodes what is reward-predictive, which differs between games. The sharpest drop in CKA is at the conv-to-fc boundary. This has direct implications for transfer learning: early layers are transferable, the representation layer is not."

---

## "Why did frozen conv backbone perform worse than scratch?"

"The frozen Pong conv layers are geometrically similar to what Breakout's own conv would develop, as CKA confirms. But 'similar' is not the same as 'identical.' The fc_repr layer was trained to interpret the specific activation patterns produced by its own conv stack. Pong's conv produces subtly different activations for the same visual input, because it was tuned for Pong's visual statistics. The fc_repr then has to learn Breakout with a fixed, slightly mismatched input — which is harder than starting fresh."

---

## "What is superposition and does it apply here?"

"Superposition (Elhage et al., 2022) is the hypothesis that neurons can represent multiple features simultaneously if they are approximately orthogonal — this allows a small network to encode more features than it has neurons. In our context, if some neurons are representing both Pong and Breakout features, they are in superposition. Our CKA analysis at early layers suggests this may be happening — the early conv neurons are responding similarly to both games' inputs, which could indicate they have learned game-general features that serve both tasks. We have not directly tested superposition but it is an interesting framing for future work."

---

## "What are the limitations?"

"Single random seed is the biggest one — we cannot rule out that some findings are initialisation-specific. We also have only two games. And the reduced replay buffer (100k vs the standard 1M) may affect the magnitude of some effects, though both algorithms are affected equally so it does not confound the comparison."

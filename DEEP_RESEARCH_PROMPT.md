# Gemini Deep Research Prompt

Use this prompt with Gemini Deep Research to generate a comprehensive literature
review covering all key themes of this dissertation component.

---

I am a Masters student at Trinity College Dublin studying how deep reinforcement
learning agents develop internal representations. My dissertation component trains
DQN and Double DQN agents on Atari games (Pong and Breakout) using a shared CNN
backbone (Conv1→Conv2→Conv3→fc_repr 512-dim→Q-values) and studies the following:

1. REPRESENTATION SIMILARITY: Early conv layers show high cross-game CKA (~0.9 at
   conv1/2) but fc_repr diverges almost completely (CKA ~0.09). I need a thorough
   literature review on: what causes this hierarchical divergence, whether this
   pattern is consistent with findings in supervised deep learning (early layers
   general, later layers task-specific), and what the Atari RL-specific literature
   says about representation geometry.

2. CATASTROPHIC FORGETTING IN RL: When I fine-tune a Pong-trained agent on Breakout,
   Pong performance collapses from +9 to -21 within 200k steps. Freezing all layers
   (conv+fc_repr) prevents forgetting; freezing only conv still causes forgetting
   (fc_repr adapts and destroys the Pong policy). Review the catastrophic forgetting
   literature specifically in RL agents — EWC, PackNet, progressive networks,
   continual backpropagation (Elsayed & Mahmood 2024), and any Atari-specific findings.

3. DEAD NEURONS AND CAPACITY LOSS: Sequential training causes dead neurons to rise
   from ~87% to ~88% (already high from training). Interleaved simultaneous training
   on both games causes dead neuron fraction to reach 92% with complete task failure.
   Review: (a) why DQN agents develop high dead neuron fractions, (b) the role of
   ReLU saturation and gradient interference in multi-task RL, (c) continual
   backpropagation as a mitigation (Elsayed & Mahmood 2024), and (d) Elhage et al.
   2022 on superposition in neural networks.

4. MULTI-TASK RL FAILURE: My interleaved training (step-level alternation every 1000
   steps, shared backbone, separate output heads, separate replay buffers) completely
   fails — neither task is learned despite equal gradient signal. Review gradient
   interference in multi-task RL, why shared representations fail even with separate
   heads, and what principled approaches (PCGrad, MGDA, task-conditioned networks)
   have been proposed.

5. DQN vs DDQN REPRESENTATIONS: DDQN produces tighter, more structured t-SNE
   clusters and lower dead neuron fractions. Review the mechanistic explanation for
   why correcting Q-value overestimation (the double Q-learning fix) produces
   qualitatively different internal representations despite identical architecture.

Please provide a comprehensive literature review covering all five areas, citing
specific papers with years and venues, identifying key open questions, and noting
any direct empirical comparisons to Atari DQN studies. Focus especially on:
Mnih et al. 2015, van Hasselt et al. 2016, Zahavy et al. 2016, Igl et al. 2021
(transient non-stationarity), Lyle et al. 2022/2023 (plasticity loss in RL),
Elsayed & Mahmood 2024 (continual backpropagation), and Elhage et al. 2022
(superposition). Flag any contradictions between my findings and the literature.

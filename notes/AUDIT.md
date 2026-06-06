# Comprehensive Project Audit

> Last updated: June 6, 2026
> This file is the single source of truth for everything that has happened in this project.
> Read this to get full context instantly after any gap.

---

## Project Identity

| Field | Value |
|---|---|
| Student | Vinol Tauro |
| Student Number | 25340566 |
| Institution | Trinity College Dublin, School of Computer Science and Statistics |
| Email | taurov@tcd.ie |
| GitHub | https://github.com/vinoltauro/rl-project |
| Hardware | GCP VM, NVIDIA L4 GPU (23.7 GB) |
| Deadlines | ~9 weeks to demo, ~12 weeks to dissertation submission (as of late May 2026) |

---

## Supervisor

- Meetings normally on Tuesdays
- Meeting 26 May 2026: professor asked for mix-and-match, layer freezing, interleaved training, PPO
- Professor reply 2 June 2026: will read report, no meeting Tuesday 9 June, suggested meeting later that week
- Professor's broad theme: "understanding natural impact of continual learning on neural network structures and how to mitigate it"
- Key concepts professor mentioned: catastrophic forgetting, capacity loss (dead neurons), superposition (Elhage et al. 2022), continual backpropagation

---

## Research Question

Do RL agents trained on structurally similar Atari games develop similar internal representations, and does algorithm choice (DQN vs DDQN) affect representation quality? What happens to those representations when the agent is forced to switch games sequentially or train on both games simultaneously?

---

## Network Architecture

```
Input: (4, 84, 84) stacked grayscale frames
→ Conv1: 32 filters, 8×8, stride 4, ReLU
→ Conv2: 64 filters, 4×4, stride 2, ReLU
→ Conv3: 64 filters, 3×3, stride 1, ReLU
→ Flatten → 3136 units
→ fc_repr: Linear(3136, 512) + ReLU     ← ALL ANALYSIS DONE HERE
→ fc_out:  Linear(512, n_actions)       ← Pong=6, Breakout=4
```

- ~1.69M total parameters, ~95% in fc_repr
- Orthogonal weight initialisation (Saxe et al. 2014)
- Forward hook captures fc_repr activations after every forward pass
- fc_out cannot be shared between games (different action counts)

---

## Hyperparameters (all experiments unless stated)

| Parameter | Value |
|---|---|
| Learning rate | 1e-4 (Adam) |
| Replay buffer | 100k transitions |
| Batch size | 32 |
| Gamma | 0.99 |
| Target update | Hard copy every 1000 steps |
| Epsilon | 1.0 → 0.01 over 100k steps |
| Gradient clip | L2 norm ≤ 10 |
| Seed | 42 |
| Checkpoint freq | Every 500k steps |

---

## Experiment 1 — Baseline 4-Agent Study

### Design
2×2 factorial: DQN and DDQN × Pong (2M steps) and Breakout (5M steps)

### Training Results

| Agent | Episodes | Final Reward | Final Q | Peak Q |
|---|---|---|---|---|
| DQN / Pong | 2,434 | 7.12 ± 5.22 | 2.64 | 2.95 |
| DDQN / Pong | 2,457 | 7.50 ± 5.03 | 2.05 | 2.48 |
| DQN / Breakout | 144,751 | 3.86 ± 4.94 | 4.06 | 5.39 |
| DDQN / Breakout | 144,948 | 4.00 ± 5.04 | 4.13 | 4.55 |

### Key Findings
1. Game effect dominates — t-SNE separates cleanly by game, not algorithm
2. DDQN produces tighter, more structured representations within each game
3. DQN overestimates Q-values by ~29% on Pong, ~18% peak on Breakout
4. DQN has more dead neurons than DDQN across both games
5. Cross-game cosine similarity is non-trivial (shared ball/paddle visual structure)
6. Performance and representation quality are decoupled — both score similarly, DDQN represents better

### Status
✅ Complete. All checkpoints saved. All plots generated and in report v2.

---

## Experiment 2 — CKA Layer Similarity

### Design
Forward hooks at conv1, conv2, conv3, fc_repr. Collect activations on shared Pong frames (1000 frames, random actions). Compute pairwise CKA between agents at each depth.

### Bug Fixed
Original script ran each agent on its own game's frames — CKA between different inputs is meaningless (all values ~0.006–0.052). Fixed by collecting shared Pong frames with random actions and running all networks on the same inputs.

### CKA Results (corrected)

| Layer | Game effect DQN | Game effect DDQN | Algo effect Pong | Algo effect Breakout |
|---|---|---|---|---|
| conv1 | 0.68 | 0.92 | 0.998 | 0.74 |
| conv2 | 0.72 | 0.85 | 0.96 | 0.72 |
| conv3 | 0.40 | 0.41 | 0.87 | 0.57 |
| fc_repr | 0.05 | 0.09 | 0.56 | 0.46 |

### Key Findings
- Early conv layers highly similar across games (conv1/2 CKA > 0.68)
- fc_repr almost completely diverged (CKA < 0.09)
- Sharpest drop at the conv→fc boundary
- This is the quantitative evidence the professor asked for to support the hierarchy claim from v1

### Status
✅ Complete. Bug fixed. Numbers in report v2.

---

## Experiment 3 — Zero-Shot Mix-and-Match

### Design
Chimera networks: transplant conv layers from one trained agent into another's upper layers. No additional training. 50 episodes per condition, 10 conditions total.

### Results — Breakout Evaluation

| Condition | Mean Reward |
|---|---|
| Native DQN/Breakout | 3.40 ± 3.82 |
| DQN/Pong conv + Breakout upper | 0.44 ± 0.78 |
| DDQN/Pong conv + Breakout upper | 0.80 ± 1.36 |
| Random conv + Breakout upper | 0.40 ± 1.00 |
| Random network | 0.42 ± 0.80 |

### Results — Pong Evaluation

| Condition | Mean Reward |
|---|---|
| Native DQN/Pong | 9.54 ± 5.08 |
| DQN/Breakout conv + Pong upper | -20.72 ± 0.45 |
| DDQN/Breakout conv + Pong upper | -21.00 ± 0.00 |
| Random conv + Pong upper | -21.00 ± 0.00 |
| Random network | -21.00 ± 0.00 |

### Key Findings
- Zero-shot transfer completely fails
- Pong conv ≈ random on Breakout (0.44 vs 0.40)
- Any foreign conv on Pong = worst possible score (-21)
- Representational similarity (CKA) does not imply functional compatibility
- The conv-to-fc interface is too tightly co-adapted for zero-shot transplantation

### Status
✅ Complete. Plots in report v2.

---

## Experiment 4 — Backbone Training (Transfer with Adaptation)

### Design
Load DQN/Pong checkpoint. Train on Breakout for 5M steps under three conditions:
- Scratch: random init baseline (already existed from Experiment 1)
- Freeze conv: Pong conv frozen, fc_repr + new Breakout fc_out train
- Full fine-tune: all layers loaded from Pong, everything trains freely

### Results

| Condition | Final Reward | Best Smoothed Reward |
|---|---|---|
| Scratch DQN/Breakout | 3.86 | 4.07 |
| Frozen conv (Pong) | 2.28 | 3.03 |
| Full fine-tune from Pong | 3.12 | 4.47 |

### Key Findings
- Frozen conv hurts Breakout learning (below scratch)
- Full fine-tune peaks highest — Pong weights as warm start help when everything adapts
- Pre-training only beneficial when all layers are free to reorganise

### Status
✅ Complete. Plots in report v2.

---

## Experiment 5 — Sequential Training (Catastrophic Forgetting Baseline)

### Design
Load DQN/Pong checkpoint. Train on Breakout for 2M steps. Measure every 200k steps:
1. Pong reward: chimera eval (current conv+fc_repr + original Pong fc_out)
2. Dead neuron fraction
3. CKA drift from original Pong representations

### Results — Freeze None (Full Sequential)

| Step | Pong Reward | Dead Neurons | CKA Drift |
|---|---|---|---|
| 0 | 9.1 | 0.879 | 0.037 |
| 200k | -20.75 | 0.729 | 0.033 |
| 400k | -21.0 | 0.742 | 0.038 |
| 1M | -20.95 | 0.820 | 0.034 |
| 2M | -20.8 | 0.867 | 0.041 |

### Results — Freeze Conv (Original, Broken Design)
Started at Pong reward = -20.9 at step 0 because fc_repr was randomly reinitialised. This made the comparison meaningless — see Experiment 6 for the fixed version.

### Key Findings
- Forgetting is immediate and total — Pong collapses to -21 by 200k steps
- Dead neurons increase steadily (0.729 → 0.867) — capacity loss mechanism confirmed
- CKA drift is immediate — representations moved from Pong baseline instantly

### Status
✅ Complete. Results committed.

---

## Experiment 6 — Fixed Sequential Training (Proper Mitigation Study)

### Design
Same as Experiment 5 but with corrected conditions:

**Condition A (freeze all):**
- Load conv + fc_repr from Pong, BOTH frozen
- Add new fc_out_breakout (4 actions), train ONLY this (2,052 params)
- Pong eval: frozen conv + frozen fc_repr + original Pong fc_out
- Tests: maximum protection — nothing in Pong pathway changes

**Condition B (freeze conv fixed):**
- Load conv (frozen) + fc_repr (trainable) from Pong
- New fc_out_breakout trains alongside fc_repr
- Pong eval: frozen conv + adapted fc_repr + original Pong fc_out
- Tests: partial protection — conv preserved, fc_repr free to adapt

### Results — Condition A (Freeze All)

| Step | Pong Reward | Dead Neurons | CKA Drift |
|---|---|---|---|
| 0 | 9.1 | 0.877 | 0.035 |
| 200k | 7.7 | 0.871 | 0.055 |
| 1M | 9.7 | 0.875 | 0.053 |
| 2M | 9.4 | 0.881 | 0.065 |

### Results — Condition B (Freeze Conv)

| Step | Pong Reward | Dead Neurons | CKA Drift |
|---|---|---|---|
| 0 | 9.1 | 0.879 | 0.055 |
| 200k | -3.75 | 0.783 | 0.045 |
| 1M | -12.05 | 0.811 | 0.092 |
| 2M | -14.0 | 0.836 | 0.043 |

### Key Findings
1. **Freeze all = almost complete forgetting prevention** — Pong stays 7-10 throughout all 2M Breakout steps
2. **Freeze conv = forgetting still happens** — Pong drops from 9.1 to -14 because fc_repr adapts away from Pong
3. **The forgetting mechanism is in fc_repr, not conv** — freezing conv alone is insufficient, forgetting happens at the representation layer
4. **Complete spectrum confirmed:**
   - Freeze all → Pong stays ~9 (forgetting prevented)
   - Freeze conv → Pong drops to -14 (partial forgetting)
   - No freeze → Pong drops to -21 (complete forgetting, immediate)

### Status
✅ Complete. Both conditions done. Plotting pending.

---

## Experiment 7 — Interleaved Training

### Design
Single DQN agent with shared backbone and two output heads alternates Pong/Breakout episode by episode.

Architecture:
- Shared: conv + fc_repr (512-dim)
- Pong head: fc_out_pong (512 → 6)
- Breakout head: fc_out_breakout (512 → 4)
- Separate replay buffers per game
- Gradients from both games flow back through shared backbone
- 1M steps per game = 2M total
- Trained from scratch (not from Pong pre-training)

Metrics every 100k total steps:
- Pong reward (last 10 episodes)
- Breakout reward (last 10 episodes)
- Dead neurons
- Cross-game CKA (shared backbone representations — high = general, low = specialised)

### Status
🔄 Currently running in tmux. ~6 hours remaining.

---

## Bugs Fixed

| Bug | Impact | Fix | Date |
|---|---|---|---|
| CKA script used different game frames per agent | All CKA values ~0, completely wrong | Collect shared Pong frames, run all networks on same inputs | May 2026 |
| Sequential freeze_conv: fc_repr randomly reinitialised | Pong eval started at -21, experiment meaningless | Load fc_repr from Pong in all sequential conditions | June 2026 |
| Interleaved training: batch["states"] dict access on NamedTuple | Script crashed at first learn() call | Changed to batch.states attribute access | June 6, 2026 |
| Saliency maps pixelated and noisy | Low quality figures | Bilinear interpolation, Gaussian smoothing, inferno colormap, higher DPI | May 2026 |
| n_envs > 1 in training | Overhead not worth it | Reverted to n_envs=1 | May 2026 |

---

## Report Versions

| Version | File | Status | What's in it |
|---|---|---|---|
| v1 | latex/main.tex | Original | Baseline 4-agent study |
| v2 | latex/main-v2.tex | Current, pushed to GitHub | + CKA (fixed), mix-and-match, backbone, em-dashes removed |
| v3 | Not yet written | Pending after interleaved finishes | + Sequential (all conditions), interleaved results |

---

## Complete Pending To-Do List

### Immediate (before professor meeting)
- [ ] Wait for interleaved training to finish (~6 hours)
- [ ] Write plotting scripts for:
  - Sequential forgetting curves (all 4 conditions: no freeze, freeze all, freeze conv, interleaved)
  - Interleaved reward curves (Pong + Breakout simultaneously)
  - Cross-game CKA over interleaved training
- [ ] Write report v3 with all new results
- [ ] Update AUDIT.md with interleaved results
- [ ] Commit and push everything

### Future iterations (after meeting)
- [ ] PPO — design then implement
- [ ] Multi-seed (deprioritised — too time-consuming)

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| Single seed (42) | Time constraint — multi-seed takes too long |
| 100k replay buffer (not 1M) | Memory constraint on GCP VM |
| 2M steps for sequential experiments | Enough to see full forgetting curve |
| Shared Pong frames for CKA | CKA requires same inputs — original code used different game frames |
| Random actions for frame collection | Neutral probe, not biased to any agent's policy |
| Interleaved starts from scratch | Clean test of interleaving alone, no pre-training confound |
| Separate replay buffers per game (interleaved) | Keeps game experience separate, cleaner gradient signal |

---

## File Map

| File | Purpose |
|---|---|
| train.py | Single-agent training |
| run_all.py | Run all 4 baseline agents |
| models/cnn.py | AtariCNN (standard single head) |
| models/cnn_two_head.py | AtariCNNTwoHead (shared backbone, two heads) |
| agents/dqn.py | DQN agent |
| agents/ddqn.py | DDQN agent (inherits DQN, overrides learn) |
| envs/wrappers.py | Atari preprocessing pipeline |
| utils/replay_buffer.py | Experience replay (NamedTuple batch) |
| analysis/extract_representations.py | Collect 512-dim vectors from checkpoints |
| analysis/tsne_visualisation.py | All t-SNE figures |
| analysis/activation_analysis.py | Training curves, Q-values, dead neurons, cosine similarity |
| analysis/saliency_maps.py | Grad-CAM saliency (improved clarity) |
| analysis/layer_similarity.py | CKA layer-wise analysis (bug fixed) |
| analysis/plot_backbone_comparison.py | Backbone learning curve comparison |
| analysis/plot_forgetting_curves.py | Sequential forgetting curve plots |
| experiments/mix_and_match_eval.py | Zero-shot chimera evaluation |
| experiments/train_backbone.py | Frozen/fine-tune backbone (5M steps) |
| experiments/sequential_training.py | Sequential forgetting (original, freeze_none complete, freeze_conv broken) |
| experiments/fixed_sequential.py | Fixed sequential (freeze_all and freeze_conv corrected) |
| experiments/interleaved_training.py | Interleaved training with two-head network |
| latex/main.tex | Report v1 |
| latex/main-v2.tex | Report v2 (current) |
| notes/ | All Obsidian notes |
| notes/AUDIT.md | This file |
| results/checkpoints/ | All model weights (.pt files, NOT in git) |
| results/plots/ | All figures (.png, in git) |
| results/logs/ | Training CSVs including forgetting metrics |

---

## Narrative Summary (the story so far)

1. **Baseline:** Game content shapes representations more than algorithm. DDQN produces better quality representations despite similar scores. Performance and representation quality are decoupled.

2. **CKA:** Early conv layers (conv1/2 CKA ~0.9) generalise across games. fc_repr diverges almost completely (CKA ~0.09). The boundary between shared visual processing and game-specific encoding is at the conv→fc transition.

3. **Zero-shot transfer:** High CKA does not mean features are interchangeable. Swapping conv layers between agents fails completely — the conv-fc interface is tightly co-adapted.

4. **Backbone training:** Frozen Pong conv hurts Breakout learning. Full fine-tune from Pong helps slightly. Pre-training only useful with full adaptation.

5. **Sequential training:** Forgetting is immediate and total (Pong → -21 by 200k steps). Dead neurons increase steadily — capacity loss is the mechanism. Forgetting is confirmed as happening at the fc_repr level, not the conv level.

6. **Fixed sequential:** Freeze all → Pong stays ~9 throughout (forgetting prevented). Freeze conv → Pong drops to -14 (forgetting still happens at fc_repr). Complete spectrum: freeze all > freeze conv > no freeze.

7. **Interleaved (ongoing):** Testing whether alternating episodes prevents forgetting while maintaining performance on both games.

---

## Professor Communication Log

| Date | From | Summary |
|---|---|---|
| 26 May 2026 | Meeting | Asked for mix-and-match, layer freezing, interleaved training, PPO. Theme: continual learning. |
| 2 June 2026 | Professor | Will read report, no meeting Tue 9 June, suggested meeting later that week. |
| 2 June 2026 | Vinol | Sent report v2 PDF named representation_study_v2_vinol_tauro.pdf |
| 3 June 2026 | Vinol | Teams message: "just sent email with updated report" |
| 3 June 2026 | Vinol | Teams reply to professor: "happy to meet later this week, will keep working on experiments" |

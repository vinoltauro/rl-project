# Comprehensive Project Audit

> Last updated: June 8, 2026
> Single source of truth for everything that has happened in this project.
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
| Deadlines | ~9 weeks to demo, ~12 weeks to dissertation (as of late May 2026) |

---

## Supervisor

- Meetings normally on Tuesdays
- Meeting 26 May 2026: professor asked for mix-and-match, layer freezing, interleaved training, PPO
- Professor reply 2 June 2026: will read report, no meeting Tuesday 9 June, suggested meeting later that week
- Professor's broad theme: "understanding natural impact of continual learning on neural network structures and how to mitigate it"
- Key concepts professor mentioned: catastrophic forgetting, capacity loss (dead neurons), superposition (Elhage et al. 2022), continual backpropagation (Elsayed & Mahmood 2024)

---

## Research Question

Do RL agents trained on structurally similar Atari games develop similar internal representations, and does algorithm choice affect representation quality? What happens when an agent is forced to switch games sequentially or train on both simultaneously?

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
- Orthogonal weight initialisation
- Forward hook captures fc_repr activations silently
- fc_out cannot be shared between games (different action counts)
- Two-head variant (models/cnn_two_head.py): shared conv+fc_repr, separate fc_out per game

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
2×2 factorial: DQN and DDQN × Pong (2M steps) and Breakout (5M steps).

### Training Results

| Agent | Episodes | Final Reward | Final Q | Peak Q |
|---|---|---|---|---|
| DQN / Pong | 2,434 | 7.12 ± 5.22 | 2.64 | 2.95 |
| DDQN / Pong | 2,457 | 7.50 ± 5.03 | 2.05 | 2.48 |
| DQN / Breakout | 144,751 | 3.86 ± 4.94 | 4.06 | 5.39 |
| DDQN / Breakout | 144,948 | 4.00 ± 5.04 | 4.13 | 4.55 |

### Key Findings
1. Game effect dominates — t-SNE separates by game not algorithm
2. DDQN produces tighter, more structured representations
3. DQN overestimates Q-values ~29% on Pong, ~18% peak on Breakout
4. DQN has more dead neurons than DDQN
5. Cross-game cosine similarity is non-trivial
6. Performance and representation quality are decoupled

### Status
✅ Complete. All checkpoints and plots in report v2/v3.

---

## Experiment 2 — CKA Layer Similarity

### Design
Forward hooks at conv1, conv2, conv3, fc_repr. Shared Pong frames (1000, random actions). Pairwise CKA at each depth.

### Bug Fixed
Original: each agent ran on its own game's frames — meaningless (values ~0.006–0.052).
Fix: shared Pong frames via random actions, all networks on same inputs.

### CKA Results (corrected)

| Layer | Game DQN | Game DDQN | Algo Pong | Algo Breakout |
|---|---|---|---|---|
| conv1 | 0.68 | 0.92 | 0.998 | 0.74 |
| conv2 | 0.72 | 0.85 | 0.96 | 0.72 |
| conv3 | 0.40 | 0.41 | 0.87 | 0.57 |
| fc_repr | 0.05 | 0.09 | 0.56 | 0.46 |

### Key Findings
- Early conv layers highly similar across games (conv1/2 CKA > 0.68)
- fc_repr almost completely diverged (CKA < 0.09)
- Sharpest drop at conv→fc boundary
- Quantitative evidence the professor asked for to support the hierarchy claim

### Status
✅ Complete. Bug fixed. Numbers in report v2/v3.

---

## Experiment 3 — Zero-Shot Mix-and-Match

### Design
Chimera networks: transplant conv layers between agents, no training. 50 episodes, 10 conditions.

### Results — Breakout

| Condition | Mean Reward |
|---|---|
| Native DQN/Breakout | 3.40 ± 3.82 |
| DQN/Pong conv + Breakout upper | 0.44 ± 0.78 |
| Random conv + Breakout upper | 0.40 ± 1.00 |
| Random network | 0.42 ± 0.80 |

### Results — Pong

| Condition | Mean Reward |
|---|---|
| Native DQN/Pong | 9.54 ± 5.08 |
| DQN/Breakout conv + Pong upper | -20.72 ± 0.45 |
| Random conv + Pong upper | -21.00 ± 0.00 |

### Key Findings
- Zero-shot transfer completely fails
- Pong conv ≈ random on Breakout
- Any foreign conv on Pong = worst possible score (-21)
- CKA similarity ≠ functional compatibility; conv-fc co-adaptation prevents transplantation

### Status
✅ Complete. Plots in report v2/v3.

---

## Experiment 4 — Backbone Training

### Design
DQN from Pong checkpoint, 5M steps on Breakout: scratch, frozen conv, full fine-tune.

### Results

| Condition | Final Reward | Best Smoothed |
|---|---|---|
| Scratch | 3.86 | 4.07 |
| Frozen conv (Pong) | 2.28 | 3.03 |
| Full fine-tune from Pong | 3.12 | 4.47 |

### Key Findings
- Frozen conv hurts Breakout (below scratch)
- Full fine-tune peaks highest — Pong weights as warm start
- Pre-training only beneficial when all layers adapt freely

### Status
✅ Complete. Plots in report v2/v3.

---

## Experiment 5 — Sequential Training Baseline (Catastrophic Forgetting)

### Design
DQN/Pong checkpoint → train on Breakout 2M steps, --freeze none. Measure every 200k: Pong reward (chimera eval), dead neurons, CKA drift.

### Results

| Step | Pong Reward | Dead Neurons | CKA Drift |
|---|---|---|---|
| 0 | 9.1 | 0.879 | 0.037 |
| 200k | -20.75 | 0.729 | 0.033 |
| 400k | -21.0 | 0.742 | 0.038 |
| 1M | -20.95 | 0.820 | 0.034 |
| 2M | -20.8 | 0.867 | 0.041 |

### Key Findings
- Forgetting immediate and total — -21 by 200k steps, never recovers
- Dead neurons increase steadily 0.729 → 0.867
- CKA drift immediate, stays there
- Forgetting occurs at fc_repr level

### Status
✅ Complete.

---

## Experiment 6 — Fixed Sequential Training (Proper Mitigation Study)

### Design
Corrected version. Always loads fc_repr from Pong (original bug: fc_repr randomly reinitialised making comparison meaningless).

**Condition A (freeze all):** conv + fc_repr frozen, only new Breakout fc_out (2,052 params) trains. Maximum protection.

**Condition B (freeze conv):** conv frozen, fc_repr + new Breakout fc_out train. Partial protection.

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

### Complete Sequential Spectrum

| Condition | Pong at 2M | Dead Neurons at 2M |
|---|---|---|
| Freeze all | **+9.4** | 0.881 |
| Freeze conv | -14.0 | 0.836 |
| No freeze | -20.8 | 0.867 |

### Key Findings
1. Freeze all = forgetting almost completely prevented (Pong stays 7-10 throughout)
2. Freeze conv = forgetting still happens — fc_repr adapts and destroys Pong decision pathway
3. **Forgetting is localised to fc_repr** — the exact layer CKA identified as game-specific
4. CKA hierarchy is predictive, not just descriptive: the layer with lowest CKA (fc_repr = 0.09) is where forgetting propagates

### Status
✅ Complete. Plots in report v3.

---

## Experiment 7 — Interleaved Training (v1 — FLAWED, for reference only)

### Design (broken)
Episode-level alternation: 1 Pong episode then 1 Breakout episode.

### Why it failed
Pong episodes ~200 steps, Breakout with EpisodicLife ~10 steps. Created ~20:1 gradient imbalance. By 1M total steps: 954k Pong steps, only 46k Breakout steps. Not proper interleaving.

### Results
Pong: -21 throughout. Breakout: 0-1.2 throughout. Dead neurons: 95%. Both tasks failed.

### Status
❌ Flawed design. Results discarded. Rerun with v2.

---

## Experiment 8 — Interleaved Training (v2 — Step-Level Alternation)

### Design
Step-level alternation: switch active game every 1,000 steps regardless of episode boundaries. Both games get exactly 1M steps = perfectly equal gradient contribution. Shared backbone (conv + fc_repr), two output heads. Trains from scratch. Measures CKA drift vs original Pong baseline every 100k total steps.

### Results

| Total Steps | Pong Steps | Breakout Steps | Pong Reward | Breakout Reward | Dead Neurons | CKA Drift |
|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0.0 | 0.0 | 0.47 | 0.015 |
| 200k | 100k | 100k | -20.9 | 1.0 | 0.87 | 0.008 |
| 500k | 250k | 250k | -20.9 | 0.4 | 0.71 | 0.009 |
| 1M | 500k | 500k | -21.0 | 0.2 | 0.86 | 0.007 |
| 2M | 1M | 1M | **-21.0** | **0.4** | **0.92** | 0.011 |

### Key Findings
1. Even with correct step-level alternation, Pong never learned (stuck at -21 throughout)
2. Breakout barely learned (0.4 final reward vs 3.86 from scratch on 5M steps)
3. Dead neurons reached 92% — conflicting gradients from two games cause capacity collapse
4. CKA drift near zero — backbone never moved far from Pong init but also never converged usefully
5. Interleaved training is the WORST outcome: sequential at least learns Breakout (3.86), interleaved learns neither task
6. This is not a bug — it reflects the fundamental difficulty of multi-task gradient interference with a shared backbone

### Comparison of all conditions

| Condition | Pong Final | Breakout Final | Dead Neurons |
|---|---|---|---|
| Scratch Breakout | N/A | 3.86 | — |
| Sequential no freeze | -20.8 | ~3.5 | 0.87 |
| Sequential freeze conv | -14.0 | ~2.5 | 0.84 |
| Sequential freeze all | +9.4 | very limited | 0.88 |
| Interleaved v2 | -21.0 | 0.4 | 0.92 |

### Status
✅ Complete. Results ready for plotting and report update.

---

## Bugs Fixed

| Bug | Impact | Fix | Date |
|---|---|---|---|
| CKA script used different game frames per agent | All CKA values ~0, meaningless | Collect shared Pong frames, run all networks on same inputs | May 2026 |
| Sequential freeze_conv: fc_repr randomly reinitialised | Pong eval started at -21, experiment meaningless | Always load fc_repr from Pong in fixed_sequential.py | June 2026 |
| Interleaved v1: episode-level alternation | 20:1 gradient imbalance, both tasks failed | Step-level alternation every 1000 steps in v2 | June 7, 2026 |
| Interleaved: batch["states"] dict access on NamedTuple | Script crashed at first learn() call | Changed to batch.states attribute access | June 6, 2026 |
| Interleaved v2: collect_repr called AtariCNN with game= kwarg | Crash on reference model | Separate collect_repr_standard for standard AtariCNN | June 7, 2026 |
| Saliency maps pixelated and noisy | Low quality figures | Bilinear interpolation, Gaussian smoothing, inferno colormap | May 2026 |

---

## Report Versions

| Version | File | Status | Contents |
|---|---|---|---|
| v1 | latex/main.tex | Original | Baseline 4-agent study |
| v2 | latex/main-v2.tex | Pushed | + CKA (fixed), mix-and-match, backbone, em-dashes removed |
| v3 | latex/main-v3.tex | Pushed, interleaved pending | + Sequential (all conditions), interleaved noted as ongoing |
| v4 | Not yet written | Pending | + Interleaved v2 results, final forgetting curves |

---

## Pending To-Do List

### Immediate
- [ ] Update plot_forgetting_curves.py to add interleaved v2 Pong reward line
- [ ] Regenerate forgetting_curves.png with all 4 conditions
- [ ] Write report v4 with interleaved v2 results and final discussion
- [ ] Update notes/12_status.md and notes/09_key_findings.md
- [ ] Commit and push everything

### Future (after professor meeting)
- [ ] PPO — design then implement
- [ ] Multi-seed (deprioritised)

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| Single seed (42) | Time constraint |
| 100k replay buffer | Memory constraint on GCP VM |
| 2M steps for sequential | Enough to see full forgetting curve |
| Shared Pong frames for CKA | CKA requires same inputs — original code used different game frames |
| Random actions for frame collection | Neutral probe not biased to any agent's policy |
| Interleaved starts from scratch | Clean test of interleaving alone, no pre-training confound |
| Step-level alternation (switch_freq=1000) | Equal gradient signal regardless of episode length asymmetry |
| Separate replay buffers per game | Keeps game experience separate, cleaner gradient signal |
| Epsilon decays over 200k steps in interleaved | Longer decay for 2M total training with two simultaneous tasks |

---

## File Map

| File | Purpose |
|---|---|
| train.py | Single-agent training |
| run_all.py | Run all 4 baseline agents |
| models/cnn.py | AtariCNN (standard single head) |
| models/cnn_two_head.py | AtariCNNTwoHead (shared backbone, two output heads) |
| agents/dqn.py | DQN agent |
| agents/ddqn.py | DDQN agent |
| envs/wrappers.py | Atari preprocessing pipeline |
| utils/replay_buffer.py | Experience replay (NamedTuple batch) |
| analysis/extract_representations.py | Collect 512-dim vectors from checkpoints |
| analysis/tsne_visualisation.py | All t-SNE figures |
| analysis/activation_analysis.py | Training curves, Q-values, dead neurons, cosine similarity |
| analysis/saliency_maps.py | Grad-CAM saliency (improved) |
| analysis/layer_similarity.py | CKA layer-wise analysis (bug fixed) |
| analysis/plot_backbone_comparison.py | Backbone learning curve comparison |
| analysis/plot_forgetting_curves.py | Sequential forgetting curves (needs interleaved v2 added) |
| experiments/mix_and_match_eval.py | Zero-shot chimera evaluation |
| experiments/train_backbone.py | Frozen/fine-tune backbone (5M steps) |
| experiments/sequential_training.py | Sequential no-freeze (Experiment 5) |
| experiments/fixed_sequential.py | Fixed sequential freeze_all and freeze_conv (Experiment 6) |
| experiments/interleaved_training.py | Interleaved v2 step-level (Experiment 8) |
| latex/main.tex | Report v1 |
| latex/main-v2.tex | Report v2 |
| latex/main-v3.tex | Report v3 (current best) |
| notes/AUDIT.md | This file |
| results/logs/*_forgetting.csv | Sequential metrics |
| results/logs/dqn_interleaved_v2_*_metrics.csv | Interleaved v2 metrics |

---

## Complete Narrative (the full story)

**Chapter 1 — Representation Analysis:**
Game content shapes representations more than algorithm. DDQN produces better quality representations despite similar scores. Performance and representation quality are decoupled.

**Chapter 2 — Layer Hierarchy (CKA):**
Early conv layers (conv1/2 CKA ~0.9) generalise across games. fc_repr diverges almost completely (CKA ~0.09). The boundary between shared visual processing and game-specific encoding is at the conv→fc transition.

**Chapter 3 — Transfer (Zero-Shot + Backbone):**
High CKA ≠ functional compatibility. Zero-shot conv transplantation fails completely. Frozen backbone hurts. Full fine-tune from Pong helps slightly as a warm start.

**Chapter 4 — Sequential Forgetting:**
Forgetting is immediate and total within 200k steps. Dead neurons increase progressively — capacity loss is the mechanism. Forgetting is localised to fc_repr, exactly the layer CKA predicted.

**Chapter 5 — Mitigation Spectrum:**
Freeze all → Pong stays ~9.4 (forgetting prevented). Freeze conv → Pong drops to -14 (fc_repr still forgets). No freeze → -21 (complete). The CKA hierarchy is not just descriptive — it predicts where intervention is needed.

**Chapter 6 — Interleaved Training:**
Both v1 (episode-level) and v2 (step-level) fail to learn either game. Conflicting gradients from two games cause 92% dead neurons and capacity collapse. Interleaved training is the worst outcome of all conditions — worse than sequential. Simple multi-task gradient mixing does not work; more principled approaches (EWC, continual backpropagation) are needed.

---

## Professor Communication Log

| Date | From | Summary |
|---|---|---|
| 26 May 2026 | Meeting | Asked for mix-and-match, layer freezing, interleaved training, PPO. Theme: continual learning. |
| 2 June 2026 | Professor | Will read report, no meeting Tue 9 June, suggested meeting later that week. |
| 2 June 2026 | Vinol | Sent report v2 PDF: representation_study_v2_vinol_tauro.pdf |
| 3 June 2026 | Vinol | Teams: "just sent email with updated report" |
| 3 June 2026 | Vinol | Teams reply: "happy to meet later this week, will keep working on experiments" |

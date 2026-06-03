# Project Audit Log

> Comprehensive record of everything done, decisions made, results found, and what is pending.
> Updated after every significant milestone. Read this to get full context instantly.

---

## Project Identity

- **Student:** Vinol Tauro, TCD Masters (Student No. 25340566)
- **Email:** taurov@tcd.ie
- **Supervisor meetings:** Tuesdays (next meeting later this week — professor unavailable Tuesday 9 June)
- **Deadlines:** ~9 weeks to demo, ~12 weeks to dissertation submission (as of late May 2026)
- **GitHub:** https://github.com/vinoltauro/rl-project
- **Hardware:** GCP VM, NVIDIA L4 GPU (23.7 GB)

---

## Research Question

Do RL agents trained on structurally similar Atari games (Pong and Breakout) develop similar internal representations, and does algorithm choice (DQN vs DDQN) affect representation quality? What happens to those representations when the agent is forced to switch games?

**Professor's broader theme:** Understanding the natural impact of continual learning on neural network structures and how to mitigate it. Key concepts: catastrophic forgetting, capacity loss (dead neurons), superposition (Elhage et al. 2022).

---

## Architecture

- CNN: Conv(32,8x8,s4) → Conv(64,4x4,s2) → Conv(64,3x3,s1) → Flatten → FC(512, ReLU) → FC(n_actions)
- `fc_repr` = 512-dim representation layer — this is what all analysis is done on
- Forward hook auto-captures activations at `fc_repr` after every forward pass
- Pong: 6 actions, Breakout: 4 actions (fc_out cannot be shared between games)
- ~1.69M parameters total, ~95% in fc_repr

---

## Hyperparameters

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

## Iteration 1 — Baseline 4-Agent Study

### What was done
- Trained DQN and DDQN on Pong (2M steps) and Breakout (5M steps)
- Extracted 512-dim representations at every 500k checkpoint
- Ran t-SNE, Grad-CAM saliency, dead neuron analysis, cosine similarity, CKA

### Training results

| Agent | Episodes | Final Reward | Final Q | Peak Q |
|---|---|---|---|---|
| DQN / Pong | 2,434 | 7.12 ± 5.22 | 2.64 | 2.95 |
| DDQN / Pong | 2,457 | 7.50 ± 5.03 | 2.05 | 2.48 |
| DQN / Breakout | 144,751 | 3.86 ± 4.94 | 4.06 | 5.39 |
| DDQN / Breakout | 144,948 | 4.00 ± 5.04 | 4.13 | 4.55 |

### Key findings
1. Game effect dominates — t-SNE separates cleanly by game, not algorithm
2. DDQN produces tighter, more structured representations within each game
3. DQN overestimates Q-values by ~29% on Pong, ~18% peak on Breakout
4. DQN has more dead neurons than DDQN across both games
5. Cross-game cosine similarity is non-trivial (shared ball/paddle structure)
6. Performance and representation quality are decoupled

### Status
✅ Complete. All checkpoints saved. All plots generated.

---

## Iteration 2 — CKA Layer Similarity

### What was done
- Registered hooks at conv1, conv2, conv3, fc_repr
- Collected activations on shared Pong frames (random actions, 1000 frames)
- Computed pairwise CKA between Pong and Breakout agents at each depth

### Bug fixed
Original script ran each agent on its own game's frames — CKA between different inputs is meaningless, all values were near zero (~0.006 to 0.052). Fixed by collecting a shared set of Pong frames with random actions and running all networks on the same inputs.

### CKA results (corrected)

| Layer | Game effect DQN | Game effect DDQN | Algo effect Pong |
|---|---|---|---|
| conv1 | 0.68 | **0.92** | 0.998 |
| conv2 | 0.72 | **0.85** | 0.96 |
| conv3 | 0.40 | 0.41 | 0.87 |
| fc_repr | 0.05 | **0.09** | 0.56 |

### Key findings
- Early conv layers are highly similar across games (conv1/conv2 CKA > 0.68)
- fc_repr has diverged almost completely (CKA < 0.09)
- This is the quantitative evidence the professor asked for to support the hierarchy claim from v1

### Status
✅ Complete. Bug fixed. Actual numbers now in report v2.

---

## Iteration 3 — Mix-and-Match (Zero-Shot + Backbone)

### Experiment A: Zero-shot eval
Built chimera networks by transplanting conv layers between agents. No additional training. 50 episodes per condition.

**Breakout results:**
| Condition | Mean Reward |
|---|---|
| Native DQN/Breakout | 3.40 |
| DQN/Pong conv + Breakout upper | 0.44 |
| Random conv + Breakout upper | 0.40 |
| Random network | 0.42 |

**Pong results:**
| Condition | Mean Reward |
|---|---|
| Native DQN/Pong | 9.54 |
| DQN/Breakout conv + Pong upper | -20.72 |
| Random conv + Pong upper | -21.00 |

**Finding:** Zero-shot transfer completely fails. Pong conv ≈ random on Breakout. Breakout conv = worst possible score on Pong. Representational similarity does not imply functional compatibility.

### Experiment B: Backbone training (5M steps on Breakout)
| Condition | Final Reward | Best Smoothed |
|---|---|---|
| Scratch | 3.86 | 4.07 |
| Frozen conv (Pong) | 2.28 | 3.03 |
| Full fine-tune from Pong | 3.12 | 4.47 |

**Finding:** Frozen conv hurts (below scratch). Full fine-tune peaks highest. Pre-training only helps when all layers can adapt.

### Status
✅ Complete. All plots generated. Results in report v2.

---

## Iteration 4 — Sequential Training (Catastrophic Forgetting)

### What was done
Loaded DQN/Pong checkpoint. Trained on Breakout for 2M steps. Every 200k steps measured:
1. Pong reward (chimera: current conv+fc_repr + original Pong fc_out)
2. Dead neuron fraction
3. CKA drift from original Pong representations

### Results: --freeze none (full sequential)

| Step | Pong Reward | Dead Neurons | CKA Drift |
|---|---|---|---|
| 0 | 9.1 | 0.879 | 0.037 |
| 200k | -20.75 | 0.729 | 0.033 |
| 400k | -21.0 | 0.742 | 0.038 |
| 600k | -21.0 | 0.773 | 0.056 |
| 800k | -20.95 | 0.783 | 0.039 |
| 1M | -20.95 | 0.820 | 0.034 |
| 1.4M | -20.85 | 0.834 | 0.045 |
| 2M | -20.8 | 0.867 | 0.041 |

### Results: --freeze conv

| Step | Pong Reward | Dead Neurons | CKA Drift |
|---|---|---|---|
| 0 | -20.9 | 0.197 | 0.065 |
| 200k | -20.95 | 0.543 | 0.074 |
| 400k | -21.0 | 0.688 | 0.051 |
| 2M | -20.9 | 0.801 | 0.043 |

### Key findings
1. **Forgetting is immediate and total** — Pong collapses to -21 by 200k steps, never recovers
2. **Frozen conv makes no difference** — Pong reward was already -20.9 at step 0 because freezing conv but randomly initialising fc_repr means the Pong output head has no trained upper layers
3. **Dead neurons increase steadily** — from 0.729 to 0.867 over 2M steps (no freeze), confirming capacity loss as the mechanism
4. **CKA drift is immediate** — representations moved away from Pong at step 0 and stayed there

### Important interpretation note
The freeze conv baseline starts at -20.9 (not +9.1) because the fc_repr and fc_out are randomly reinitialised. This means the chimera evaluation (Pong fc_out + new random fc_repr) was never trained together. The experiment measures the wrong thing for freeze conv — to properly test whether freezing conv preserves Pong knowledge, we would need to keep fc_repr AND fc_out from Pong and only allow fine-tuning of a new Breakout head. This is a design limitation to acknowledge.

### Status
✅ Both conditions complete. Plotting and report v3 pending.

---

## Report Versions

| Version | File | Status | Contents |
|---|---|---|---|
| v1 | latex/main.tex | Original | Baseline 4-agent study |
| v2 | latex/main-v2.tex | Current, pushed | + CKA, mix-and-match, backbone, fixed CKA bug, em-dashes removed |
| v3 | Not yet written | Pending | + Sequential training results |

---

## Pending To-Do List

### Immediate (this week, before professor meeting)
- [ ] Write plotting script for sequential forgetting curves
- [ ] Generate forgetting curve plots (Pong reward, dead neurons, CKA drift — both conditions)
- [ ] Write report v3 with sequential results
- [ ] Commit and push everything

### Next iterations (after meeting)
- [ ] Interleaved training — single agent alternating Pong/Breakout episodes (professor asked)
- [ ] PPO — design first, implement after interleaved
- [ ] Fix sequential freeze_conv experiment design (freeze conv+fc_repr, only new Breakout head trains) — current design has a flaw

### Not prioritised
- [ ] Multi-seed runs — takes too long given timeline

---

## Key Decisions Made

| Decision | Reason |
|---|---|
| Only seed 42 | Time constraint — multi-seed takes too long |
| 100k replay buffer (not 1M) | Memory constraint on GCP VM |
| 2M steps for sequential (not 5M) | Enough to measure forgetting, faster turnaround |
| Shared Pong frames for CKA | CKA requires same inputs for both networks — original bug used different game frames |
| Random actions for frame collection | Neutral probe not biased to either agent's policy |

---

## Files Reference

| File | Purpose |
|---|---|
| experiments/sequential_training.py | Sequential forgetting experiment |
| experiments/mix_and_match_eval.py | Zero-shot chimera evaluation |
| experiments/train_backbone.py | Frozen/fine-tune backbone |
| analysis/layer_similarity.py | CKA layer-wise analysis (bug fixed) |
| analysis/plot_backbone_comparison.py | Backbone learning curve plot |
| analysis/saliency_maps.py | Grad-CAM saliency (improved clarity) |
| latex/main-v2.tex | Current report |
| notes/ | All Obsidian notes |
| results/logs/*_forgetting.csv | Sequential training forgetting metrics |

---

## Bugs Fixed

| Bug | Impact | Fix |
|---|---|---|
| CKA script used different game frames per agent | All CKA values were ~0, meaningless | Collect shared Pong frames with random actions, run all networks on same inputs |
| Saliency maps were pixelated and noisy | Figures looked low quality | Bilinear interpolation, Gaussian smoothing, higher DPI, inferno colormap |
| n_envs > 1 in training | Overhead not worth it for small DQN | Reverted to n_envs=1 |

---

## Contact

- Professor reply (2 June 2026): Will try to read report soon. No meeting Tuesday 9 June. Suggested meeting later this week instead.

# PROJECT CONTEXT — RL Dissertation Component
## For Claude: Everything You Need to Understand This Project

---

## 1. What This Project Is

This is **one component** of a larger Masters dissertation on Reinforcement Learning.
It is NOT the full dissertation — just one self-contained experimental component.

The student is **new to RL** but working at masters level. They need help that is:
- Technically correct and comprehensive
- Explained clearly without being patronising
- Production-quality code, not toy examples

---

## 2. Research Question

> *Do RL agents trained on structurally similar Atari games develop similar internal
> representations, and does the choice of algorithm (DQN vs Double DQN) affect the
> nature and quality of those representations — independent of the game being played?*

---

## 3. Experimental Design — 2×2 Factorial

| | Pong | Breakout |
|---|---|---|
| **DQN** | Run 1 ✅ (training in progress) | Run 2 |
| **Double DQN** | Run 3 | Run 4 |

**Why this design:**
- **Game effect** (Run 1 vs Run 2): same algorithm, different game — do similar games produce similar representations?
- **Algorithm effect** (Run 1 vs Run 3): same game, different algorithm — does DDQN produce cleaner representations than DQN?
- **Interaction effect**: does the algorithm matter more on one game than the other?

**Why Pong + Breakout:**
Both games share low-level visual features (ball, paddle, deflection physics) but differ in
high-level strategy (opponent vs bricks, vertical vs horizontal movement). This makes them
ideal for studying shared vs game-specific representations.

**Why DQN + Double DQN:**
- Structurally identical CNN backbone — only the TD target computation differs
- DQN overestimates Q-values (uses same network to select AND evaluate actions)
- DDQN fixes this with one line change (online net selects, target net evaluates)
- Any difference in representations is purely algorithmic, not architectural

---

## 4. Current Status

- **Run 1 (DQN/Pong):** Training on Google Colab T4 GPU — ~1M/2M steps complete as of last check
  - Reward has gone from -21 (random) to +8 (winning) ✅
  - Checkpoint saved at 1M steps ✅
  - Results downloaded locally ✅
- **Runs 2, 3, 4:** Not yet started — will run after Run 1 completes
- **Analysis:** Not yet run — needs all 4 training runs to complete first

---

## 5. Codebase Architecture

```
rl_project/
├── train.py                          ← Main training script (config-driven)
├── run_all.py                        ← Runs all 4 experiments + analysis sequentially
├── setup_env.py                      ← One-shot setup (detects GPU/MPS/CPU)
├── requirements.txt                  ← Dependencies
│
├── envs/
│   └── wrappers.py                   ← 7 Atari preprocessing wrappers
│                                        IMPORTANT: needs `import ale_py; gym.register_envs(ale_py)`
│                                        before gym.make() — known fix already applied
│
├── models/
│   └── cnn.py                        ← Shared CNN backbone used by BOTH agents
│                                        3 sizes: small/medium/large
│                                        512-dim penultimate layer = REPRESENTATION LAYER
│                                        Forward hook auto-populates model.representation
│
├── agents/
│   ├── dqn.py                        ← DQN agent (epsilon-greedy, replay, target net)
│   └── ddqn.py                       ← Double DQN — inherits DQN, overrides learn() only
│                                        Literally 3 lines different from DQN
│
├── utils/
│   ├── replay_buffer.py              ← Circular numpy buffer (uint8, normalises at sample time)
│   ├── logger.py                     ← CSV + TensorBoard logging
│   └── checkpoint.py                ← Save/load full training state
│
├── analysis/
│   ├── extract_representations.py   ← Collects 512-dim vectors via forward hooks
│   │                                   Saves as .npz with metadata (game, algo, step, action, reward)
│   ├── tsne_visualisation.py        ← All 6 t-SNE figure types (see Section 8)
│   ├── saliency_maps.py             ← Grad-CAM saliency, DQN vs DDQN side-by-side
│   └── activation_analysis.py      ← Dead neurons, Q-value overestimation, cosine similarity,
│                                       training curves
│
└── experiments/configs/
    ├── run1_dqn_pong.yaml            ← DQN / Pong / 2M steps
    ├── run2_dqn_breakout.yaml        ← DQN / Breakout / 5M steps
    ├── run3_ddqn_pong.yaml           ← DDQN / Pong / 2M steps
    ├── run4_ddqn_breakout.yaml       ← DDQN / Breakout / 5M steps
    ├── ablation_net_small.yaml       ← CNN scale: small
    ├── ablation_net_large.yaml       ← CNN scale: large
    ├── ablation_lr_low.yaml          ← LR: 5e-5
    ├── ablation_lr_high.yaml         ← LR: 5e-4
    └── ablation_buffer_small.yaml   ← Buffer: 10k
```

---

## 6. Network Architecture

```
Input: (4, 84, 84) — 4 stacked grayscale frames

Conv2D(32, 8×8, stride=4) → ReLU     # (32, 20, 20)
Conv2D(64, 4×4, stride=2) → ReLU     # (64, 9, 9)
Conv2D(64, 3×3, stride=1) → ReLU     # (64, 7, 7)
Flatten()                              # 3136
Linear(512) → ReLU                    # ← REPRESENTATION LAYER (forward hook here)
Linear(n_actions)                     # Q-values output
```

- **Representation layer** = the 512-dim vector before the final Q-value layer
- This is what gets extracted and fed into t-SNE
- A `register_forward_hook` on `fc_repr` populates `model.representation` after every forward pass
- Three size variants via `net_scale`: small ([16,32,32], 256), medium ([32,64,64], 512), large ([64,128,128], 1024)

---

## 7. Key Hyperparameters (Baseline — Same for All 4 Runs)

| Parameter | Value | Notes |
|---|---|---|
| Learning rate | 1e-4 | Adam |
| Replay buffer | 100,000 | Reduced from original 1M for feasibility |
| Batch size | 32 | Standard |
| Gamma | 0.99 | Discount factor |
| Target update | Every 1,000 steps | Hard update |
| Epsilon start | 1.0 | Full exploration |
| Epsilon end | 0.01 | |
| Epsilon decay | 100,000 steps | Linear |
| Gradient clip | 10.0 | |
| Frame skip | 4 | Action repeated 4 frames |
| Frame stack | 4 | 4 grayscale frames stacked |
| Input size | 84×84 | Grayscale |
| Checkpoint freq | 500,000 steps | For temporal t-SNE analysis |

**CPU mode** (auto-applied when no GPU detected):
- Buffer: 100k → 50k
- Batch: 32 → 64
- Target update: 1000 → 500
- Checkpoint freq: 500k → 250k

---

## 8. Analysis Plan — All 17 Figures

### Training Performance
| Fig | File | Description |
|---|---|---|
| 1 | training_curves_pong.png | DQN vs DDQN reward over steps on Pong |
| 2 | training_curves_breakout.png | DQN vs DDQN reward over steps on Breakout |
| 3 | qvalue_overestimation.png | Mean max Q-value: DQN drifts up, DDQN stable |

### t-SNE Representation Analysis
| Fig | File | Description |
|---|---|---|
| 4 | tsne_game_effect_dqn.png | DQN/Pong vs DQN/Breakout — game effect |
| 5 | tsne_game_effect_ddqn.png | DDQN/Pong vs DDQN/Breakout — game effect |
| 6 | tsne_algo_effect_pong.png | DQN vs DDQN on Pong — algorithm effect |
| 7 | tsne_algo_effect_breakout.png | DQN vs DDQN on Breakout — algorithm effect |
| 8 | tsne_all_agents.png | All 4 agents — which clusters dominate? |
| 9 | tsne_by_reward.png | Coloured by cumulative reward |
| 10 | tsne_temporal_*.png | Evolution over training checkpoints |

### Saliency Maps
| Fig | File | Description |
|---|---|---|
| 11 | saliency_pong.png | Grad-CAM: DQN vs DDQN on Pong |
| 12 | saliency_breakout.png | Grad-CAM: DQN vs DDQN on Breakout |

### Activation Analysis
| Fig | File | Description |
|---|---|---|
| 13 | dead_neurons.png | Fraction of dead neurons over training |
| 14 | cosine_similarity.png | Pong vs Breakout mean representation similarity |

### Ablations
| Fig | File | Description |
|---|---|---|
| 15 | ablation_network_size.png | Small vs medium vs large CNN |
| 16 | ablation_lr.png | Learning rate sensitivity |
| 17 | ablation_buffer.png | Replay buffer size effect |

---

## 9. Hypotheses to Test

These are what the analysis should find (or not find — negative results are fine):

1. **Game effect dominant:** t-SNE clusters should separate more by game than by algorithm — *what* you play shapes representations more than *how* you learn
2. **DDQN tighter clusters:** DDQN representations should be more compact in t-SNE — cleaner gradient signal → more structured representations
3. **Shared features exist:** Pong and Breakout cosine similarity should be non-trivial (>0.2) — shared ball/paddle physics create common low-level features
4. **DQN more dead neurons:** DQN's overestimation noise should cause more chronically inactive neurons than DDQN
5. **Q-value drift:** DQN Q-values should drift upward over training (overestimation), DDQN should stay grounded
6. **Ball focus in saliency:** Both agents should show highest saliency around the ball position

---

## 10. Atari Preprocessing Stack

Order matters — must be applied in this exact sequence:

```
NoopResetEnv      → random 1–30 no-ops at episode start (prevents state memorisation)
MaxAndSkipEnv     → repeat action 4 frames, max-pool last 2 (handles flickering)
EpisodicLifeEnv   → treat life loss as terminal (denser learning signal)
FireResetEnv      → press FIRE to start (Breakout requires this, safe for Pong)
WarpFrame         → RGB → 84×84 grayscale
ClipRewardEnv     → clip to {-1, 0, +1} (normalises scale across games)
FrameStack        → stack 4 frames → (4, 84, 84) input (gives motion information)
```

**Known issue already fixed:** Must call `import ale_py; gym.register_envs(ale_py)` before `gym.make()` otherwise gymnasium raises `NamespaceNotFound: ALE`.

---

## 11. Running the Project

### Setup (any machine)
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  # CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # NVIDIA GPU
pip install gymnasium[atari] ale-py numpy pandas matplotlib seaborn scikit-learn tensorboard pyyaml tqdm opencv-python AutoROM
AutoROM --accept-license
```

### Training
```bash
python run_all.py                  # All 4 runs + analysis
python run_all.py --cpu            # Force CPU mode
python run_all.py --smoke_test     # 50k steps per run (quick test)
python run_all.py --runs 1 3       # Specific runs only
python train.py --config experiments/configs/run1_dqn_pong.yaml  # Single run
python train.py --config experiments/configs/run1_dqn_pong.yaml --resume results/checkpoints/<name>.pt  # Resume
```

### Analysis (after training)
```bash
python run_all.py --analysis_only
```

### Google Colab
```python
!pip install gymnasium[atari] ale-py numpy pandas matplotlib seaborn scikit-learn tensorboard pyyaml tqdm opencv-python AutoROM -q
!AutoROM --accept-license
# Upload rl_project.zip via Files panel, then:
import zipfile, os
with zipfile.ZipFile('rl_project.zip', 'r') as z:
    z.extractall('/content/')
os.chdir('/content/rl_project')
!python run_all.py
```

---

## 12. Known Issues & Fixes

| Issue | Fix |
|---|---|
| `NamespaceNotFound: ALE` | Add `import ale_py; gym.register_envs(ale_py)` before `gym.make()` in `wrappers.py` |
| Path with spaces breaks on Windows | Move project to path with no spaces e.g. `C:\rl_project\` |
| `setup_env.py` fails on Windows with spaces in path | Don't use `setup_env.py` — run pip commands manually from README |
| Colab disconnects mid-training | Mount Drive and copy checkpoints: `shutil.copytree('/content/rl_project/results', '/content/drive/MyDrive/rl_results', dirs_exist_ok=True)` |
| `AutoROM --accept-license -q` fails | Remove `-q` flag: `AutoROM --accept-license` |

---

## 13. What Good Training Looks Like

For **Pong (DQN)**:
- Steps 0–100k: reward -21 to -18 (random/early learning)
- Steps 100k–300k: reward -15 to -5 (agent learning to hit ball)
- Steps 500k–1M: reward -5 to +10 (agent winning some games)
- Steps 1M–2M: reward +10 to +21 (near-optimal play)

**Confirmed:** Run 1 showed -21 → +8 by 1M steps, which is exactly correct.

For **Breakout (DQN/DDQN)**:
- Takes longer — needs ~2–3M steps before positive rewards
- Harder game with more complex strategy (ball angle, brick patterns)

---

## 14. Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10–3.11 | Language |
| PyTorch | 2.x | Neural networks |
| gymnasium | 0.29.x | RL environment interface |
| ale-py | 0.8.x | Atari Learning Environment |
| scikit-learn | 1.3+ | t-SNE (`sklearn.manifold.TSNE`) |
| matplotlib / seaborn | Latest | Plotting |
| tensorboard | 2.x | Live training monitoring |
| opencv-python | 4.x | Frame preprocessing |
| numpy | 1.24–1.26 | Numerics (avoid 2.0+) |

---

## 15. Key Academic References

1. **Mnih et al. (2015)** — *Human-level control through deep reinforcement learning* (Nature) — Original DQN
2. **van Hasselt et al. (2016)** — *Deep RL with Double Q-learning* (AAAI) — DDQN
3. **Zahavy et al. (2016)** — *Graying the Black Box: Understanding DQNs* (ICML) — t-SNE on DQN representations (directly relevant)
4. **Maaten & Hinton (2008)** — *Visualizing Data using t-SNE* (JMLR) — t-SNE methodology
5. **Bellemare et al. (2013)** — *The Arcade Learning Environment* — Atari benchmark
6. **Hessel et al. (2018)** — *Rainbow* (AAAI) — Context for DQN extensions
7. **Sutton & Barto (2018)** — *Reinforcement Learning: An Introduction* — Background theory

---

## 16. Important Design Decisions (Don't Change Without Good Reason)

- **Both algorithms use identical CNN** — this is the whole point. Any representational difference is algorithmic, not architectural.
- **Checkpoints every 500k steps** — needed for temporal t-SNE evolution plots (Fig 10)
- **Replay buffer stores uint8, normalises at sample time** — saves ~4× RAM vs float32
- **Forward hook on fc_repr** — extracts representations without modifying forward pass
- **ClipRewardEnv clips to {-1,0,+1}** — makes hyperparameters transferable across games
- **FrameStack of 4** — gives agent velocity/motion information without RNN
- **Seed=42** — fixed for reproducibility across all runs

---

## 17. What Still Needs To Be Done

- [ ] Run 2: DQN on Breakout (5M steps)
- [ ] Run 3: DDQN on Pong (2M steps)
- [ ] Run 4: DDQN on Breakout (5M steps)
- [ ] Ablation studies (5 configs, all on Pong for speed)
- [ ] Extract representations from all checkpoints
- [ ] Generate all 17 figures
- [ ] Write up component section for dissertation

---

*Context document version: 1.0 | Project: RL Dissertation Component | Student level: Masters (new to RL)*

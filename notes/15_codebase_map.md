# 🗺️ Codebase Map — Where Is Everything

> If the professor asks "show me the training code" or "where is CKA computed?" — this is your map.

---

## Top-Level Structure

```
rl-project/
│
├── train.py                    ← Single-agent training script
├── run_all.py                  ← Run all 4 agents (+ multi-seed support)
│
├── models/
│   └── cnn.py                  ← The CNN architecture (fc_repr lives here)
│
├── agents/
│   ├── dqn.py                  ← DQN agent (select_action, learn)
│   └── ddqn.py                 ← DDQN agent (inherits DQN, overrides learn)
│
├── envs/
│   └── wrappers.py             ← Atari preprocessing pipeline
│
├── utils/
│   ├── replay_buffer.py        ← Experience replay (uint8 storage)
│   ├── logger.py               ← CSV + TensorBoard logging
│   └── checkpoint.py           ← Save/load model weights
│
├── analysis/
│   ├── extract_representations.py  ← Collect 512-dim vectors from checkpoints
│   ├── tsne_visualisation.py       ← All t-SNE figures
│   ├── activation_analysis.py      ← Training curves, Q-values, dead neurons, cosine sim
│   ├── saliency_maps.py            ← Grad-CAM saliency figures
│   ├── layer_similarity.py         ← CKA at each layer depth
│   └── plot_backbone_comparison.py ← Backbone learning curve comparison
│
├── experiments/
│   ├── mix_and_match_eval.py       ← Zero-shot chimera evaluation
│   ├── train_backbone.py           ← Frozen/fine-tune backbone training
│   └── sequential_training.py      ← Sequential forgetting experiment
│
├── results/
│   ├── checkpoints/            ← Saved model weights (.pt files)
│   ├── plots/                  ← All figures (.png files)
│   ├── logs/                   ← Training CSV logs
│   └── representations/        ← Extracted 512-dim vectors (.npz files)
│
├── latex/
│   ├── main.tex                ← Report v1
│   └── main-v2.tex             ← Report v2 (current)
│
└── notes/                      ← You are here
```

---

## Key Files Explained

### `models/cnn.py`
The network. Defines `AtariCNN` with three conv layers, `fc_repr` (512-dim), and `fc_out`. The forward hook for capturing representations is set up here. `net_scale` parameter lets you change network size: small (256-dim), medium (512-dim), large (1024-dim).

**Find it when:** Professor asks about the architecture. Line ~80 has the forward hook.

---

### `agents/dqn.py`
Standard DQN. Key methods:
- `select_action(obs)` — epsilon-greedy action selection
- `learn(batch)` — one gradient update, returns (loss, mean_q)
- `_compute_targets(batch)` — THIS is where the DQN overestimation happens (uses target net for both select and evaluate)

**Find it when:** Professor asks about the training algorithm or overestimation.

---

### `agents/ddqn.py`
Inherits DQNAgent, overrides only `learn()`. The change is three lines in `_compute_targets()`: online net selects, target net evaluates.

**Find it when:** Professor asks "what is the difference between DQN and DDQN in your code?"

---

### `envs/wrappers.py`
The preprocessing pipeline. `make_atari_env()` applies (in order): NoopReset → MaxAndSkip(4) → EpisodicLife → FireReset → WarpFrame(84×84) → ClipReward → FrameStack(4).

**Find it when:** Professor asks about preprocessing or frame stacking.

---

### `analysis/layer_similarity.py`
CKA computation. Registers hooks at conv1, conv2, conv3, fc_repr. Collects activations for Pong and Breakout agents on the same states. Computes pairwise CKA.

**Find it when:** Professor asks about CKA methodology.

---

### `experiments/sequential_training.py`
The forgetting experiment. Every 200k Breakout steps:
1. Evaluates a chimera (current conv+fc_repr + original Pong fc_out) on Pong
2. Counts dead neurons
3. Computes CKA drift from original Pong representations

Results saved to `results/logs/*_forgetting.csv`.

**Find it when:** Professor asks about the continual learning experiment.

---

## How to Run Things

```bash
# Activate environment
source venv/bin/activate

# Train all 4 agents
python run_all.py --seeds 42 --training_only

# Run all analysis
python run_all.py --analysis_only

# Run specific experiment
python experiments/sequential_training.py --freeze none
python experiments/mix_and_match_eval.py --n_episodes 50

# Quick smoke test (50k steps)
python run_all.py --smoke_test
```

---

## Checkpoint Naming Convention

```
{algo}_{game}_seed{seed}_scalemedium_lr0001_buf100k_step{step:08d}.pt

Examples:
dqn_pong_seed42_scalemedium_lr0001_buf100k_step02000000.pt
ddqn_breakout_seed42_scalemedium_lr0001_buf100k_step05000000.pt
dqn_sequential_sequential_full_seed42_scalemedium_step01000000.pt
```

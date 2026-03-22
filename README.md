# Atari Representation Learning
## DQN vs Double DQN on Pong & Breakout — RL Dissertation Component

---

## Quick Start

```bash
# 1. Unzip and enter project
unzip rl_project.zip && cd rl_project

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate.bat

# 3. Run setup (detects GPU/MPS/CPU, installs everything, verifies)
python setup_env.py

# 4. Smoke test — verify it all works (~5 min)
python train.py --config experiments/configs/run1_dqn_pong.yaml --steps 50000

# 5. Run all 4 experiments + analysis
python run_all.py
```

---

## All Commands Reference

### Setup

```bash
python setup_env.py                          # First-time setup on any machine
```

### Training

```bash
# All 4 runs + analysis (recommended, leave overnight)
python run_all.py

# All 4 runs, CPU mode (auto-applies optimised settings)
python run_all.py --cpu

# Smoke test only (50k steps per run, ~5 min each)
python run_all.py --smoke_test

# Pong runs only (faster first results)
python run_all.py --runs 1 3

# Breakout runs only
python run_all.py --runs 2 4

# Individual runs
python train.py --config experiments/configs/run1_dqn_pong.yaml       # DQN / Pong
python train.py --config experiments/configs/run2_dqn_breakout.yaml   # DQN / Breakout
python train.py --config experiments/configs/run3_ddqn_pong.yaml      # DDQN / Pong
python train.py --config experiments/configs/run4_ddqn_breakout.yaml  # DDQN / Breakout

# Resume after disconnection
python train.py --config experiments/configs/run2_dqn_breakout.yaml \
                --resume results/checkpoints/<checkpoint_name>.pt
```

### Ablation Studies

```bash
python train.py --config experiments/configs/ablation_net_small.yaml
python train.py --config experiments/configs/ablation_net_large.yaml
python train.py --config experiments/configs/ablation_lr_low.yaml
python train.py --config experiments/configs/ablation_lr_high.yaml
python train.py --config experiments/configs/ablation_buffer_small.yaml
```

### Analysis

```bash
# Run full analysis pipeline
python run_all.py --analysis_only

# Or step by step:
python analysis/extract_representations.py --output results/representations
python analysis/tsne_visualisation.py --repr_dir results/representations --output_dir results/plots
python analysis/activation_analysis.py --log_dir results/logs --repr_dir results/representations --output_dir results/plots
python analysis/saliency_maps.py \
    --ckpt_dqn_pong      results/checkpoints/<dqn_pong_final>.pt \
    --ckpt_ddqn_pong     results/checkpoints/<ddqn_pong_final>.pt \
    --ckpt_dqn_breakout  results/checkpoints/<dqn_breakout_final>.pt \
    --ckpt_ddqn_breakout results/checkpoints/<ddqn_breakout_final>.pt \
    --output_dir results/plots
```

### TensorBoard

```bash
tensorboard --logdir results/logs/tb
# Open: http://localhost:6006
```

---

## Hardware & Time Estimates

| Hardware | Pong (2M steps) | Breakout (5M steps) | All 4 runs |
|---|---|---|---|
| NVIDIA RTX 3080+ | ~45 min | ~2h | ~6h |
| NVIDIA T4 (Colab) | ~1.5h | ~4h | ~11h |
| Apple M1/M2 (MPS) | ~2–3h | ~6–8h | ~16–22h |
| CPU (8-core) | ~6–8h | ~15–20h | **2–3 days** |

**CPU-only?** Use `--cpu` flag for optimised settings. For full runs, consider:
- Google Colab (free T4): colab.research.google.com — use the included `.ipynb`
- Kaggle (free P100, 30h/week): kaggle.com → New Notebook → GPU T100

---

## HPC / University Cluster (SLURM)

```bash
#!/bin/bash
#SBATCH --job-name=rl_atari
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/slurm_%j.out

module load python/3.11
module load cuda/11.8

source /path/to/rl_project/venv/bin/activate
cd /path/to/rl_project

python train.py --config experiments/configs/run1_dqn_pong.yaml
```

Submit all 4 as parallel jobs:
```bash
sbatch submit_run.sh   # edit config inside for each run
squeue -u $USER        # monitor
```

---

## Experimental Design (2x2 Factorial)

| | Pong | Breakout |
|---|---|---|
| **DQN** | Run 1 | Run 2 |
| **Double DQN** | Run 3 | Run 4 |

- **Game effect**: Run 1 vs Run 2 (same algo, different game)
- **Algorithm effect**: Run 1 vs Run 3 (same game, different algo)
- **Interaction**: Does algo choice matter more on one game?

---

## Figures Produced (17 total)

| Fig | File | Content |
|-----|------|---------|
| 1 | training_curves_pong.png | DQN vs DDQN reward on Pong |
| 2 | training_curves_breakout.png | DQN vs DDQN reward on Breakout |
| 3 | qvalue_overestimation.png | Q-value drift: DQN overestimates, DDQN doesn't |
| 4 | tsne_game_effect_dqn.png | t-SNE: DQN/Pong vs DQN/Breakout |
| 5 | tsne_game_effect_ddqn.png | t-SNE: DDQN/Pong vs DDQN/Breakout |
| 6 | tsne_algo_effect_pong.png | t-SNE: DQN vs DDQN on Pong |
| 7 | tsne_algo_effect_breakout.png | t-SNE: DQN vs DDQN on Breakout |
| 8 | tsne_all_agents.png | t-SNE: all 4 agents in one plot |
| 9 | tsne_by_reward.png | t-SNE coloured by cumulative reward |
| 10 | tsne_temporal_*.png | Representation evolution over training |
| 11 | saliency_pong.png | Grad-CAM: DQN vs DDQN on Pong |
| 12 | saliency_breakout.png | Grad-CAM: DQN vs DDQN on Breakout |
| 13 | dead_neurons.png | Dead neuron fraction over training |
| 14 | cosine_similarity.png | Pong vs Breakout representation similarity |
| 15-17 | ablation_*.png | Network/LR/buffer ablations |

---

## Key References

1. Mnih et al. (2015) — *Human-level control through deep reinforcement learning* (Nature)
2. van Hasselt et al. (2016) — *Deep RL with Double Q-learning* (AAAI)
3. Zahavy et al. (2016) — *Graying the Black Box: Understanding DQNs* (ICML)
4. Maaten & Hinton (2008) — *Visualizing Data using t-SNE* (JMLR)
5. Bellemare et al. (2013) — *The Arcade Learning Environment*

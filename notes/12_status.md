# 📋 Current Status and What's Next

> Updated: June 2026. Check tmux session `training` for live progress.

---

## ✅ Completed

| Experiment | Result |
|---|---|
| Baseline 4-agent training (DQN/DDQN × Pong/Breakout) | All checkpoints saved |
| CKA layer similarity analysis | Plot generated |
| Zero-shot mix-and-match eval (50 episodes) | Plots generated |
| Frozen conv backbone (5M steps) | Complete |
| Full fine-tune backbone (5M steps) | Complete |
| Backbone learning curve comparison plot | Generated |
| Report v2 (main-v2.tex) | Written and pushed to GitHub |
| Improved saliency maps | Regenerated and pushed |

---

## 🔄 Currently Running

| What | Where | Status |
|---|---|---|
| Sequential training `--freeze none` | tmux: training | ~5-6 hrs total, running now |

**Check progress:**
```bash
tmux attach -t training
# Then Ctrl+B D to detach without stopping it
```

---

## ⏳ After Current Run Finishes

1. Run `--freeze conv` sequential condition
2. Write plotting script for forgetting curves (Pong reward, dead neurons, CKA drift)
3. Generate forgetting curve plots
4. Update report → v3

---

## 📅 Future Iterations (Not Started)

| Experiment | Priority | Notes |
|---|---|---|
| Interleaved training | High | Professor asked for this — single agent alternating Pong/Breakout |
| Multi-seed runs | Medium | Run everything with seed 1 as well for robustness |
| PPO | Low | Design first, implement after sequential done |

---

## 🗂️ Key File Locations

| File | Purpose |
|---|---|
| `results/checkpoints/` | All trained model weights |
| `results/plots/` | All figures |
| `results/logs/` | Training CSV logs |
| `latex/main-v2.tex` | Current report |
| `experiments/sequential_training.py` | Sequential forgetting experiment |
| `experiments/mix_and_match_eval.py` | Zero-shot layer swap eval |
| `experiments/train_backbone.py` | Frozen/fine-tune backbone training |
| `analysis/plot_backbone_comparison.py` | Backbone learning curve plot |

---

## 💾 GitHub

Repo: `https://github.com/vinoltauro/rl-project`
All code and plots are pushed. Checkpoints are NOT in GitHub (too large) — they live on the GCP VM.

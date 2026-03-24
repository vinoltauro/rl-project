# Tonight's Plan — Professor Meeting 10 AM

Current time: ~1:30 AM | Meeting: 10:00 AM | Time available: ~8.5 hours

---

## Status

| Run | Status | Checkpoints |
|---|---|---|
| Run 1 — DQN / Pong | ✅ Complete | 4 (500k, 1M, 1.5M, 2M) |
| Run 2 — DQN / Breakout | ✅ Complete | 10 (500k → 5M) |
| Run 3 — DDQN / Pong | 🔄 Running (~1M/2M steps) | 1 so far |
| Run 4 — DDQN / Breakout | ❌ Not started | 0 |

---

## DO NOT TOUCH the training terminal
Training is running in tmux window 0. Open a second tmux window for all analysis:
```bash
Ctrl+B, then C          # opens new window — training keeps running untouched
```
Switch between windows anytime:
```bash
Ctrl+B, then 0          # training window
Ctrl+B, then 1          # analysis window
```

---

## Step 1 — Pull New Scripts (30 seconds)

```bash
cd ~/rl-project
git pull
```

---

## Step 2 — Training Curves + Q-Value Plots (instant)

Reads directly from CSV logs — done in seconds.

```bash
python analysis/activation_analysis.py \
  --log_dir results/logs \
  --repr_dir results/representations \
  --output_dir results/plots
```

**Produces:**
- `training_curves_pong.png` — DQN/Pong reward over 2M steps
- `training_curves_breakout.png` — DQN/Breakout reward over 5M steps
- `qvalue_overestimation.png` — DQN Q-value drift

---

## Step 3 — Extract Representations (~35 minutes)

Loads each checkpoint, runs the agent for 5,000 steps, saves the 512-dim vectors to `.npz` files.

```bash
python analysis/extract_partial.py
```

- Processes DQN/Pong (4 checkpoints) and DQN/Breakout (10 checkpoints)
- DDQN runs automatically skipped — no checkpoints yet
- Safe to re-run if interrupted — skips already completed files

**Leave it running and wait for it to finish before Step 4.**

---

## Step 4 — Generate All Remaining Figures (~15 minutes)

Run once Step 3 is complete:

```bash
python analysis/tsne_visualisation.py \
  --repr_dir results/representations \
  --output_dir results/plots

python analysis/activation_analysis.py \
  --log_dir results/logs \
  --repr_dir results/representations \
  --output_dir results/plots
```

**Produces:**
- `tsne_game_effect_dqn.png` — DQN/Pong vs DQN/Breakout clusters
- `tsne_by_reward.png` — representations coloured by reward
- `tsne_temporal_dqn_pong.png` — how representations evolved over training
- `dead_neurons.png` — dead neuron fraction over training
- `cosine_similarity.png` — cross-game shared structure

---

## Step 5 — Download Plots

```bash
zip -r rl_plots_partial.zip results/plots/
echo $HOME
```

Then: **gear icon (top right of SSH window) → Download file**
Enter: `/home/YOUR_USERNAME/rl_plots_partial.zip`  ← replace with output of `echo $HOME`

---

## Step 6 — Prepare for the Meeting

Read `DISSERTATION_EXPLAINER.md` — covers every figure, every concept, and has a summary section written for professor conversations.

**Opening line:**
> "Runs 1 and 2 are complete. Runs 3 and 4 are finishing overnight. Here's what we can already see."

---

## What to Show the Professor

| Figure | What to say |
|---|---|
| `training_curves_pong.png` | "DQN on Pong: reward goes from -21 (random) to near-optimal. Validates the implementation is working correctly." |
| `training_curves_breakout.png` | "Breakout is harder — needs 5M steps. Reward climbs steadily throughout." |
| `qvalue_overestimation.png` | "DQN Q-values drift upward over training — this is the overestimation bias DDQN was designed to fix. Full comparison once Run 3 finishes." |
| `tsne_game_effect_dqn.png` | "First representation result: if clusters separate by colour, Pong and Breakout produce different internal representations despite sharing ball/paddle features." |
| `tsne_temporal_dqn_pong.png` | "Representation space evolves from a random cloud early in training to structured clusters as the agent learns." |
| `dead_neurons.png` | "Fraction of the 512 representation neurons that are chronically inactive. DQN vs DDQN comparison available once Run 3 finishes." |
| `cosine_similarity.png` | "Cosine similarity between mean Pong and Breakout representations — measures how much cross-game shared structure exists." |

---

## Timeline Tonight

| Time | Action |
|---|---|
| Now | `Ctrl+B C` — open new tmux window |
| Now | Step 1: `git pull` |
| Now | Step 2: activation_analysis (instant) |
| Now | Step 3: `extract_partial.py` — start and leave running (~35 min) |
| ~2:15 AM | Step 4: tsne + activation_analysis (~15 min) |
| ~2:30 AM | Step 5: zip and download plots |
| ~2:30 AM | Step 6: read DISSERTATION_EXPLAINER.md |
| ~3:00 AM | Sleep |
| 10:00 AM | Meeting with Professor |

---

## After the Meeting — Full Analysis (all 4 runs)

Once Runs 3 and 4 complete (~18 hours from now), run:

```bash
cd ~/rl-project
python analysis/extract_partial.py
python analysis/tsne_visualisation.py --repr_dir results/representations --output_dir results/plots
python analysis/activation_analysis.py --log_dir results/logs --repr_dir results/representations --output_dir results/plots
zip -r rl_plots_final.zip results/plots/
```

This produces all 17 figures with full DQN vs DDQN comparisons.

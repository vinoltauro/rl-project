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

## Step 1 — Pull New Scripts onto the VM

Open VM SSH terminal and run:

```bash
cd ~/rl-project
git pull
```

You should see `extract_partial.py` and `DISSERTATION_EXPLAINER.md` appear.

---

## Step 2 — Generate Training Curves + Q-Value Plots

These read directly from the CSV logs — no waiting, runs in seconds.

```bash
python analysis/activation_analysis.py \
  --log_dir results/logs \
  --repr_dir results/representations \
  --output_dir results/plots
```

**Produces:**
- `training_curves_pong.png` — DQN/Pong reward over 2M steps
- `training_curves_breakout.png` — DQN/Breakout reward over 5M steps
- `qvalue_overestimation.png` — DQN Q-value drift (DQN side only for now)

---

## Step 3 — Extract Representations (~35 minutes)

This loads each checkpoint, runs the agent for 5,000 steps, and saves the 512-dim vectors.

```bash
python analysis/extract_partial.py
```

**What it does:**
- DQN/Pong — processes 4 checkpoints → 4 `.npz` files
- DQN/Breakout — processes 10 checkpoints → 10 `.npz` files
- DDQN runs — automatically skipped (no checkpoints yet)

**Leave it running.** If it gets interrupted, re-run — it skips files already done.

---

## Step 4 — Generate t-SNE and Remaining Figures (~15 minutes)

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
- `tsne_by_reward.png` — representations coloured by reward (DQN runs only)
- `tsne_temporal_dqn_pong.png` — how DQN/Pong representations evolved over training
- `dead_neurons.png` — dead neuron fraction over training (DQN only)
- `cosine_similarity.png` — cross-game similarity for DQN

---

## Step 5 — Download Plots

```bash
zip -r rl_plots_partial.zip results/plots/
```

Then in the SSH terminal: **gear icon (top right) → Download file**
Enter path: `/home/YOUR_USERNAME/rl_plots_partial.zip`

> To find your username: `echo $HOME`

---

## Step 6 — Prepare for the Meeting

Read through `DISSERTATION_EXPLAINER.md` — it explains every figure, every concept, and has a summary section at the bottom specifically for professor conversations.

**Key message to open with:**
> "Runs 1 and 2 are complete. Runs 3 and 4 are finishing overnight. Here's what the first two runs already show."

---

## What You Are Showing the Professor

| Figure | File | What to say |
|---|---|---|
| Training curves | `training_curves_pong.png` | "DQN on Pong goes from -21 (random) to near-optimal. Expected behaviour, validates implementation." |
| Training curves | `training_curves_breakout.png` | "Breakout takes longer — harder game, 5M steps. Reward climbs steadily." |
| Q-value drift | `qvalue_overestimation.png` | "DQN Q-values drift upward over training — this is the overestimation bias DDQN was designed to fix. We'll see the comparison once Run 3 completes." |
| t-SNE game effect | `tsne_game_effect_dqn.png` | "This is the first representation result. If the clusters separate by colour, Pong and Breakout produce different internal representations despite sharing visual features." |
| t-SNE temporal | `tsne_temporal_dqn_pong.png` | "This shows how the representation space evolved during training — from a random cloud to structured clusters." |
| Dead neurons | `dead_neurons.png` | "Tracks what fraction of the 512 representation neurons are chronically inactive. Once Run 3 is done we can compare DQN vs DDQN." |
| Cosine similarity | `cosine_similarity.png` | "Measures how similar the mean Pong and Breakout representations are. Non-trivial similarity would suggest shared ball/paddle features created shared structure." |

---

## After the Meeting — Full Analysis

Once Runs 3 and 4 complete (~18 hours from now):

```bash
cd ~/rl-project
python analysis/extract_partial.py        # extracts DDQN checkpoints too
python analysis/tsne_visualisation.py --repr_dir results/representations --output_dir results/plots
python analysis/activation_analysis.py --log_dir results/logs --repr_dir results/representations --output_dir results/plots
zip -r rl_plots_final.zip results/plots/
```

This gives all 17 figures including the full DQN vs DDQN comparisons.

---

## Timeline Tonight

| Time | Action |
|---|---|
| Now | `git pull` on VM |
| Now | Run Step 2 (instant) |
| Now | Start Step 3 (35 min) |
| ~2:15 AM | Run Step 4 (15 min) |
| ~2:30 AM | Download plots |
| ~2:30 AM | Read DISSERTATION_EXPLAINER.md |
| ~3:00 AM | Sleep |
| 10:00 AM | Meeting |

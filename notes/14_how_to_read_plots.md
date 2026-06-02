# 📈 How to Read Every Plot

> A plain-English guide to what each figure shows and what to look for.

---

## training_curves_pong.png / training_curves_breakout.png

**What it is:** Episode reward over training steps, smoothed with a rolling average.

**How to read it:**
- X-axis: training steps (millions)
- Y-axis: episode reward (higher = better)
- Two lines: DQN (one colour) and DDQN (another)

**What to look for:**
- Both lines should rise from negative/zero reward to positive reward as training progresses
- The lines should end up close together (similar final performance)
- Pong: starts at -21 (losing every point), crosses into positive around 1M steps
- Breakout: starts near 0, slowly improves — takes longer because the game is harder

**What it tells you:** Both algorithms learn. Neither is dramatically better at the task.

---

## qvalue_overestimation.png

**What it is:** Mean maximum Q-value over training for each agent.

**How to read it:**
- X-axis: training steps
- Y-axis: mean Q-value (how optimistic the agent is about its best action)
- Four lines: DQN/Pong, DDQN/Pong, DQN/Breakout, DDQN/Breakout

**What to look for:**
- DQN lines should be noticeably higher than DDQN lines
- DQN lines may keep rising throughout training (overestimation accumulating)
- DDQN lines should be flatter and lower

**What it tells you:** DQN is systematically overconfident. DDQN corrects this.

---

## tsne_game_effect_dqn.png / tsne_game_effect_ddqn.png

**What it is:** 5,000 game states projected to 2D. Each dot is one moment in a game. Colours = which game (Pong or Breakout).

**How to read it:**
- Look for two separate clouds of dots
- If they are well-separated → the agent developed different internal representations for each game
- If they overlap → the agent is encoding both games similarly

**What to look for:** Clear separation between the two colours.

**What it tells you:** The game shapes the representation more than the algorithm.

---

## tsne_algo_effect_pong.png / tsne_algo_effect_breakout.png

**What it is:** Same as above but colours = DQN vs DDQN (same game).

**How to read it:**
- Both colours should occupy roughly the same region (both learned the same game)
- Look at the **tightness** of each colour's cluster
- DDQN cluster should be more compact and structured than DQN

**What to look for:** DDQN dots form a tighter, more organised shape.

**What it tells you:** DDQN produces cleaner internal representations within the same game.

---

## tsne_all_agents.png

**What it is:** All four agents' representations projected together. 4 colours.

**How to read it:**
- The dominant structure should be two big groups by game
- Within each game group, DQN and DDQN should form sub-clusters

**What to look for:** Game = primary separation. Algorithm = secondary separation.

**What it tells you:** The hierarchy of what matters: game first, algorithm second.

---

## tsne_by_reward.png

**What it is:** t-SNE coloured by cumulative episode reward (green = high reward, red = low reward).

**How to read it:**
- Look for spatial patterns in the colour gradient
- If high-reward states cluster in one region and low-reward in another → representations encode value structure
- If colours are randomly mixed → representations do not encode value

**What to look for:** Some spatial separation between green and red.

**What it tells you:** The representation layer encodes some information about how well the agent is doing, not just what the game looks like.

---

## tsne_temporal_*.png

**What it is:** Representations at different training checkpoints (500k steps, 1M, 1.5M, 2M). Each checkpoint has a different colour or is shown in a separate panel.

**How to read it:**
- Early checkpoints: diffuse, unstructured cloud
- Later checkpoints: tighter, more organised clusters
- Look for when structure first emerges

**What to look for:** DDQN should develop structure earlier than DQN.

**What it tells you:** How the agent's internal model develops over training.

---

## layer_similarity_cka.png

**What it is:** A bar chart or line plot showing CKA between Pong and Breakout agents at four layer depths.

**How to read it:**
- X-axis: layer depth (conv1 → conv2 → conv3 → fc_repr)
- Y-axis: CKA value (0 = completely different, 1 = identical)
- Two lines/bars: DQN comparison and DDQN comparison

**What to look for:**
- High values at conv1 and conv2
- Drop at conv3
- Sharp drop at fc_repr

**What it tells you:** Where in the network the two games' representations diverge.

---

## saliency_pong.png / saliency_breakout.png

**What it is:** Game frames with a heat map overlay showing which pixels the agent pays most attention to.

**How to read it:**
- Bright/warm colours = pixels that strongly influence the agent's decision
- Dark/cool colours = pixels the agent ignores
- Each row is one game moment. Three columns: raw frame, DQN saliency, DDQN saliency

**What to look for:**
- Both agents should highlight the ball
- DDQN saliency should be more tightly focused (less spread out)
- On Breakout, DDQN should show more attention to bricks in the ball's path

**What it tells you:** The agent has learned to attend to the right objects. DDQN attends more precisely.

---

## dead_neurons.png

**What it is:** Fraction of dead neurons over training checkpoints.

**How to read it:**
- X-axis: training step
- Y-axis: fraction of 512 neurons that are inactive (0 to 1)
- Two lines per game: DQN and DDQN

**What to look for:**
- DQN line should be consistently above DDQN
- Both may increase over training (capacity loss accumulates)

**What it tells you:** DQN wastes more of its representational capacity.

---

## cosine_similarity.png

**What it is:** Cross-game cosine similarity between mean Pong and mean Breakout representations over training.

**How to read it:**
- X-axis: training step
- Y-axis: cosine similarity (-1 to 1, higher = more similar)
- Two lines: DQN and DDQN

**What to look for:**
- Both lines above zero (non-trivial similarity despite game differences)
- DDQN line slightly higher than DQN

**What it tells you:** Some shared visual content is preserved in the representations, more so under DDQN.

---

## mix_and_match_breakout.png / mix_and_match_pong.png

**What it is:** Bar chart of mean episode reward for 5 chimera conditions.

**How to read it:**
- First bar = native agent (upper bound, should be highest)
- Last bar = random network (lower bound)
- Middle bars = various cross-game combinations
- Error bars = standard deviation across 50 episodes

**What to look for:**
- Key test bar (cross-game conv) should be close to the lower bound, not the upper bound
- This confirms zero-shot transfer does not work

**What it tells you:** Representational similarity does not equal functional transferability.

---

## backbone_comparison_reward.png

**What it is:** Three learning curves on Breakout: scratch, frozen conv, full fine-tune.

**How to read it:**
- X-axis: training steps (millions)
- Y-axis: episode reward (smoothed)
- Three lines: scratch (grey), frozen conv (blue), full fine-tune (red)

**What to look for:**
- Frozen conv should be below scratch (it hurts)
- Full fine-tune should peak above scratch (warm start helps)

**What it tells you:** Pre-training only helps when all layers can adapt.

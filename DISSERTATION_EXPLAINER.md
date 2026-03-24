# Dissertation Component: Atari Representation Learning
## A Complete Explanation — DQN vs Double DQN on Pong & Breakout

---

## Table of Contents

1. [What We Are Studying and Why It Matters](#1-what-we-are-studying-and-why-it-matters)
2. [Background: Reinforcement Learning](#2-background-reinforcement-learning)
3. [Deep Q-Networks (DQN)](#3-deep-q-networks-dqn)
4. [The Overestimation Problem](#4-the-overestimation-problem)
5. [Double DQN — The Fix](#5-double-dqn--the-fix)
6. [Why Atari? Why Pong and Breakout?](#6-why-atari-why-pong-and-breakout)
7. [The Research Question](#7-the-research-question)
8. [Experimental Design — 2×2 Factorial](#8-experimental-design--22-factorial)
9. [The Preprocessing Pipeline](#9-the-preprocessing-pipeline)
10. [The Neural Network Architecture](#10-the-neural-network-architecture)
11. [The Representation Layer — The Heart of the Study](#11-the-representation-layer--the-heart-of-the-study)
12. [How Training Works — Step by Step](#12-how-training-works--step-by-step)
13. [Key Hyperparameters and Why](#13-key-hyperparameters-and-why)
14. [The Analysis Pipeline — All 17 Figures](#14-the-analysis-pipeline--all-17-figures)
15. [t-SNE — What It Is and Why We Use It](#15-t-sne--what-it-is-and-why-we-use-it)
16. [Hypotheses — What We Expect to Find](#16-hypotheses--what-we-expect-to-find)
17. [Training Progress So Far](#17-training-progress-so-far)
18. [Ablation Studies](#18-ablation-studies)
19. [Codebase Architecture](#19-codebase-architecture)
20. [Academic References](#20-academic-references)

---

## 1. What We Are Studying and Why It Matters

This project investigates what happens *inside* a deep reinforcement learning agent's brain as it learns to play Atari video games.

When a neural network learns to play a game, it develops its own internal representation of the game world — a compressed, abstract encoding of what it sees. We cannot read this representation directly (it's just a vector of numbers), but we can analyse its structure to understand what the agent has learned.

The central question is: **does the game you play, or the algorithm you use to learn, more strongly shape the internal representations that develop?**

This matters for several reasons:

- **Transfer learning:** If two different games produce similar representations, knowledge from one game might transfer to another. This has implications for building general-purpose game-playing agents.
- **Algorithm design:** If DDQN (a refinement of DQN) produces more structured representations, this explains *why* it performs better — not just *that* it does.
- **Interpretability:** Understanding what an agent has internally learned is a step toward making RL more transparent and explainable.

This is not merely a performance comparison. There are already hundreds of papers measuring which algorithm scores higher. This study asks a deeper question: **why do the algorithms differ at a representational level?**

---

## 2. Background: Reinforcement Learning

### 2.1 The Core Idea

Reinforcement Learning (RL) is a framework in which an **agent** learns to make decisions by interacting with an **environment**. The agent takes **actions**, receives **rewards** (feedback signals), and learns a **policy** — a mapping from situations to actions — that maximises cumulative reward over time.

Unlike supervised learning, there is no labelled training dataset. The agent must discover what works through trial and error.

### 2.2 The Markov Decision Process (MDP)

RL problems are formally modelled as **Markov Decision Processes**, defined by:

- **State (s):** A description of the current situation (e.g., pixel values of the game screen)
- **Action (a):** Something the agent can do (e.g., move left, fire)
- **Reward (r):** A scalar signal received after each action (+1 for scoring, -1 for losing a life)
- **Transition function P(s' | s, a):** The probability of moving to state s' after taking action a in state s
- **Discount factor (γ):** A value between 0 and 1 (we use 0.99) that determines how much the agent values future rewards vs immediate ones

The **Markov property** means that the future is fully determined by the current state — past history is irrelevant given the current state. In Atari, this is approximately true: seeing 4 consecutive frames tells you current ball position, velocity, and game context.

### 2.3 The Q-Function

The **Q-function** (or action-value function) Q(s, a) answers the question: *"How good is it to take action a in state s, assuming I act optimally afterwards?"*

Formally:
```
Q*(s, a) = E [ r₀ + γr₁ + γ²r₂ + ... | s₀=s, a₀=a, following optimal policy π* ]
```

The optimal policy is simply: in any state s, take the action a with the highest Q-value:
```
π*(s) = argmax_a Q*(s, a)
```

If we could compute Q* exactly, we would have a perfect policy. The challenge is that in Atari, the state space (all possible screen configurations) is astronomically large — far too large to store Q-values in a table. This motivates the use of neural networks to **approximate** Q*.

### 2.4 The Bellman Equation

The Q-function satisfies a recursive consistency condition called the **Bellman equation**:

```
Q*(s, a) = E [ r + γ · max_{a'} Q*(s', a') ]
```

In words: the value of being in state s and taking action a equals the immediate reward r plus the discounted value of the best action in the next state s'.

This gives us a training objective: the Q-network's predictions should be **consistent** with this equation. The error between the current prediction and the Bellman target is called the **TD error** (Temporal Difference error), and minimising it is how the agent learns.

---

## 3. Deep Q-Networks (DQN)

### 3.1 What DQN Is

DQN (Deep Q-Network) was introduced by Mnih et al. at DeepMind in 2015 (published in *Nature*). It was the first method to successfully learn to play a wide range of Atari games directly from raw pixel inputs, achieving superhuman performance on many titles.

The key idea is to use a **convolutional neural network** to approximate the Q-function. Instead of a lookup table, the network takes raw game pixels as input and outputs Q-values for every possible action simultaneously.

### 3.2 Two Critical Innovations

DQN introduced two stabilising tricks that made deep Q-learning work in practice:

**1. Experience Replay**

Rather than learning from each experience immediately and discarding it, DQN stores transitions (s, a, r, s', done) in a **replay buffer** (a circular queue of 100,000 transitions in our implementation). During training, random **mini-batches** of 32 are sampled from this buffer.

Why does this help?
- **Breaks temporal correlations:** Consecutive game frames are highly correlated. Learning from them in sequence would cause the network to overfit to recent experience. Random sampling decorrelates the data.
- **Data efficiency:** Each transition can be used for multiple gradient updates, not just once.
- **Stabilises training:** Reduces the variance of gradient estimates.

**2. Target Network**

DQN maintains **two copies** of the neural network:
- **Online network (θ):** Updated at every training step via gradient descent
- **Target network (θ⁻):** A periodically-copied frozen snapshot of the online network

The TD target (the "label" for supervised learning) is computed using the target network:
```
y = r + γ · max_{a'} Q(s', a'; θ⁻)
```

The loss is then:
```
L = MSE( Q(s, a; θ), y )
```

Why two networks? Without the target network, we would be chasing a moving target — the labels (y) would change every gradient step because they depend on the same network we're updating. This causes catastrophic instability. The target network provides a stable reference point, updated only every 1,000 steps (a hard copy, not a soft blend).

### 3.3 The DQN Training Loop

At each timestep:
1. Observe current state s (4 stacked frames)
2. With probability ε, take a random action (exploration); otherwise take argmax Q(s, ·; θ) (exploitation)
3. Execute action a, receive reward r, observe next state s'
4. Store (s, a, r, s', done) in replay buffer
5. If buffer has enough samples, sample a batch and perform one gradient update
6. Periodically copy θ → θ⁻ (target update)
7. Decay ε (reduce exploration over time)

---

## 4. The Overestimation Problem

### 4.1 Why DQN Overestimates Q-Values

DQN has a systematic flaw: it tends to **overestimate Q-values** over time.

The source is the `max` operator in the TD target:
```
y = r + γ · max_{a'} Q(s', a'; θ⁻)
```

The target network is used to both **select** the best next action (argmax) and **evaluate** its value (the Q-value at that action). When there is noise or error in Q-value estimates (which there always is early in training), `max` will systematically pick the action with the highest *estimated* value — which is likely a noisy overestimate, not the true best.

This is a statistical fact: if you take the maximum of several noisy estimates, the result is biased upward. The bias compounds over training because overestimated targets generate further overestimated targets.

### 4.2 What This Looks Like in Practice

As DQN trains, its Q-values tend to **drift upward** beyond what they should be. You can observe this empirically: the mean maximum Q-value increases monotonically over training, even after performance has plateaued. This is Figure 3 in our analysis.

For representations, this matters because the gradient signal driving learning contains this systematic error. The hypothesis is that noisy, overestimated gradients push the 512-dim representation layer toward a less structured, more diffuse geometry.

---

## 5. Double DQN — The Fix

### 5.1 The DDQN Idea

Double DQN (van Hasselt et al., 2016) fixes the overestimation bias with a single conceptual change to the target computation.

The insight: separate **action selection** from **action evaluation** by using different networks for each:

**DQN target (flawed):**
```
y = r + γ · Q( s', argmax_{a'} Q(s', a'; θ⁻);  θ⁻ )
                              ↑ selected by θ⁻   ↑ evaluated by θ⁻
```
Same network does both — overestimation bias

**DDQN target (corrected):**
```
y = r + γ · Q( s', argmax_{a'} Q(s', a'; θ);   θ⁻ )
                              ↑ selected by θ    ↑ evaluated by θ⁻
```
Online net selects, target net evaluates — bias removed

### 5.2 Why This Removes the Bias

The online network θ and target network θ⁻ have different error patterns (they are trained at different times and have different random noise in their estimates). If θ overestimates action a₁, θ⁻ is unlikely to also overestimate it by the same amount. On average, this decoupling cancels the bias.

This is the same principle as double estimators in statistics: use one data split to select and another to evaluate, to get an unbiased estimate of the maximum.

### 5.3 The Implementation — Just 3 Lines Different

In our codebase, `DoubleDQNAgent` inherits everything from `DQNAgent` and only overrides the `learn()` method. The only change is in the target computation:

```python
# DQN (in dqn.py):
next_q = self.target_net(batch.next_states)
max_next_q = next_q.max(dim=1).values               # target net picks best action

# DDQN (in ddqn.py):
next_q_online = self.online_net(batch.next_states)
best_next_actions = next_q_online.argmax(dim=1, keepdim=True)   # online net picks action
next_q_target = self.target_net(batch.next_states)
max_next_q = next_q_target.gather(1, best_next_actions).squeeze(1)  # target net evaluates it
```

Everything else — the CNN architecture, the replay buffer, epsilon decay, target update frequency, the optimiser — is **identical**. This is a critical design decision: any observed difference in representations between DQN and DDQN must be due to this single algorithmic change, not any other factor.

---

## 6. Why Atari? Why Pong and Breakout?

### 6.1 The Atari Benchmark

The Arcade Learning Environment (ALE, Bellemare et al., 2013) provides a standardised interface to 57 classic Atari 2600 games. It has become the standard benchmark for deep RL because:

- **High-dimensional input:** Raw 210×160 RGB pixel frames — no hand-crafted features
- **Diverse challenges:** Different games require different strategies, making it a strong test of generality
- **Known difficulty:** Decades of human play provide a meaningful baseline

All games share the same input format (pixel observations) and the same general framework (score points, avoid losing lives), making cross-game comparison principled.

### 6.2 Why Pong?

Pong is a two-player paddle game. One player controls a paddle on the left; the opponent (AI) controls the right. The ball bounces between paddles. A point is scored when the ball passes the opponent's paddle.

**Structurally:**
- Simple dynamics: ball position, velocity, and two paddle positions are all that matter
- Sparse actions: move up, move down, stay still
- Short episodes: games last ~1,000 frames
- Reward is easy: +1 for scoring, -1 for conceding
- **Training time:** 2 million steps (~3–4 hours on a GPU)

Pong is the "easiest" classic Atari game for DQN. An untrained agent scores -21 (loses every point). A well-trained DQN typically achieves +21 (wins every point). This fast, clear learning signal makes Pong ideal as a first validation and for ablation studies.

### 6.3 Why Breakout?

Breakout is a single-player game. The player controls a paddle at the bottom. A ball bounces upward and must break through a wall of bricks. Losing the ball ends the game (or costs a life).

**Structurally:**
- Similar low-level features to Pong: ball, paddle, deflection physics
- More complex high-level strategy: ball angles change based on brick position, catching the ball near paddle edges matters, breaking through layers requires planning
- **Training time:** 5 million steps (~7–9 hours on a GPU)

### 6.4 Why These Two Together?

The pairing is scientifically deliberate:

| Feature | Pong | Breakout |
|---|---|---|
| Ball | ✅ | ✅ |
| Paddle | ✅ | ✅ |
| Deflection physics | ✅ | ✅ |
| Opponent | ✅ (one opponent) | ❌ (no opponent) |
| Brick pattern | ❌ | ✅ |
| Vertical movement | Primary | Ball only |
| Episode structure | Back-and-forth | One-directional until ball lost |

The hypothesis: the **shared** low-level features (ball, paddle, physics) should produce shared representational structure. The **different** high-level strategy (opponent vs bricks, different movement axes) should produce game-specific structures. t-SNE can reveal which dominates.

---

## 7. The Research Question

> *Do RL agents trained on structurally similar Atari games develop similar internal representations, and does the choice of algorithm (DQN vs Double DQN) affect the nature and quality of those representations — independent of the game being played?*

This breaks into three sub-questions:

**Q1 — Game effect:** When the same algorithm plays two different games, do the resulting internal representations cluster by game? If yes, the game played (what you learn on) shapes representations more than the algorithm.

**Q2 — Algorithm effect:** When two different algorithms play the same game, do their representations differ structurally? If yes, the learning algorithm shapes how knowledge is encoded internally, not just what performance is achieved.

**Q3 — Interaction effect:** Does the algorithm choice matter more on one game than the other? This is the full factorial interaction — whether Q1 and Q2 effects compound or cancel.

---

## 8. Experimental Design — 2×2 Factorial

The study uses a fully crossed **2×2 factorial design**:

|  | **Pong** | **Breakout** |
|---|---|---|
| **DQN** | Run 1 (2M steps) | Run 2 (5M steps) |
| **Double DQN** | Run 3 (2M steps) | Run 4 (5M steps) |

This design isolates each factor:

- **Game effect:** Compare Run 1 vs Run 2 (same DQN algorithm, different games) or Run 3 vs Run 4 (same DDQN algorithm, different games)
- **Algorithm effect:** Compare Run 1 vs Run 3 (same Pong game, different algorithms) or Run 2 vs Run 4 (same Breakout game, different algorithms)
- **Interaction:** 2×2 ANOVA structure reveals whether the algorithm effect is consistent across games

All four runs use **identical hyperparameters** and the **same CNN architecture**. The only things that vary are the game being played and whether the DQN or DDQN target is used. This makes the design clean: any observed representational differences are attributable to one of these two factors.

### Seed Reproducibility

All four runs use `seed=42` for:
- Environment initialisation (determines the sequence of no-op resets at episode starts)
- PyTorch random state (weight initialisation, batch sampling order)
- NumPy random state

This ensures results are reproducible on the same hardware.

---

## 9. The Preprocessing Pipeline

Raw Atari frames are 210×160 RGB images at 60Hz. Before any learning can happen, a standard preprocessing pipeline (the "DeepMind stack") transforms them into a suitable format. **The order of wrappers is fixed and matters** — changing the order changes the behaviour.

### Wrapper 1: NoopResetEnv

**What it does:** At the start of each episode, takes a random number (1–30) of "no-op" (do nothing) actions before handing control to the agent.

**Why:** Without this, every episode starts from exactly the same screen state. The agent could memorise a fixed sequence of optimal opening moves rather than learning a general policy. The random no-ops ensure the agent sees diverse initial states.

### Wrapper 2: MaxAndSkipEnv

**What it does:** Repeats the selected action for 4 consecutive game frames. Returns the pixel-wise maximum of the last 2 frames as the observation. Returns the summed reward over all 4 frames.

**Why (frame skip):** At 60Hz, consecutive frames are nearly identical — moving the paddle by 1 pixel. Applying the same action for 4 frames allows the agent to make meaningful decisions at 15Hz, dramatically speeding up training (4× fewer gradient updates needed for the same elapsed game time).

**Why (max-pooling):** Atari's hardware had a limitation: due to the console's sprite rendering system, objects could only appear on alternating frames (a technique called "flickering"). A ball might be invisible on every other frame. Taking the pixel-wise maximum of the last 2 frames ensures objects are always visible.

### Wrapper 3: EpisodicLifeEnv

**What it does:** Treats each life loss as an episode terminal signal, even if the game itself continues. The actual game reset (which costs no more lives) only happens when all lives are gone.

**Why:** In games with multiple lives (Breakout has 5), without this wrapper the agent receives only a small penalty for losing a life and the episode continues. This dilutes the learning signal. By treating life loss as terminal, the agent learns to value survival more strongly, which speeds up learning.

### Wrapper 4: FireResetEnv

**What it does:** Presses the FIRE button at the start of each episode.

**Why:** Some Atari games (including Breakout) require pressing FIRE to launch the ball. Without this wrapper, the agent could sit at the start screen indefinitely, never receiving any reward. For Pong, this wrapper is a no-op (it does nothing harmful).

### Wrapper 5: WarpFrame

**What it does:** Converts the 210×160 RGB frame to an 84×84 grayscale image using bilinear interpolation.

**Why (grayscale):** Colour is largely irrelevant for deciding game strategy. Converting to grayscale reduces the input dimension by 3×.

**Why (84×84):** This is the standard resolution established by Mnih et al. (2015). Small enough to be computationally tractable; large enough to preserve game-relevant structure (ball position, paddle position, brick patterns).

### Wrapper 6: ClipRewardEnv

**What it does:** Clips all rewards to {-1, 0, +1} using `np.sign(reward)`.

**Why:** Different Atari games have wildly different reward scales. In Breakout, breaking lower bricks scores 1 point; upper bricks score 7. In Space Invaders, enemy types score different amounts. Without clipping, learning rate and loss scale would need to be tuned per-game. Clipping normalises the reward scale so the same hyperparameters (especially learning rate) work across all games. This is why our hyperparameters transfer between Pong and Breakout without any change.

**Trade-off:** Clipping destroys reward magnitude information. The agent cannot distinguish scoring 1 point from scoring 7. For this study, this is acceptable because we care about representation structure, not fine-grained reward optimisation.

### Wrapper 7: FrameStack

**What it does:** Stacks the last 4 grayscale frames into a single (4, 84, 84) tensor, which becomes the input to the neural network.

**Why:** A single frame is a snapshot with no temporal information. The agent cannot distinguish a ball moving left from a ball moving right just from one frame. By stacking 4 consecutive frames, the agent sees:
- Current object positions
- Object velocities (via positional differences between frames)
- Short-term trajectory information

This gives the agent implicit motion information without requiring a recurrent architecture (like an LSTM). Four frames at 15Hz = approximately 267ms of history, which is sufficient for Pong and Breakout.

### The Final Input

After the full pipeline, each observation is:
- Shape: (4, 84, 84) — 4 channels (frames), 84×84 pixels each
- Dtype: uint8 (0–255), stored in the replay buffer
- Normalised to float [0, 1] at **sample time** (not in the buffer), saving ~4× RAM

---

## 10. The Neural Network Architecture

Both DQN and DDQN use an **identical CNN architecture** — the one from Mnih et al. (2015). This is a deliberate and essential design choice: any representational difference we observe between DQN and DDQN cannot be attributed to architectural differences.

### Architecture Overview

```
Input:   (4, 84, 84)     — 4 stacked grayscale frames

Conv2D(32, 8×8, stride=4) → ReLU   output: (32, 20, 20)
Conv2D(64, 4×4, stride=2) → ReLU   output: (64, 9, 9)
Conv2D(64, 3×3, stride=1) → ReLU   output: (64, 7, 7)

Flatten()                            output: 3136 values

Linear(3136 → 512)        → ReLU   ← REPRESENTATION LAYER
Linear(512 → n_actions)             ← Q-values output
```

**Parameter count (medium scale):** ~1.7 million trainable parameters

### The Convolutional Layers

Each convolutional layer performs local pattern detection:

**Layer 1 — Conv(32, 8×8, stride=4):**
- Large 8×8 kernels with stride 4 — captures coarse spatial structure
- Reduces 84×84 to 20×20 (a 4× spatial downsampling)
- 32 feature maps — learns 32 different low-level patterns (edges, textures)
- Typically learns to detect: ball position, paddle outline, brick edges

**Layer 2 — Conv(64, 4×4, stride=2):**
- 4×4 kernels, stride 2 — captures mid-level structure
- Reduces 20×20 to 9×9
- 64 feature maps — combines Layer 1 features into higher-level patterns
- Typically learns: object shapes, relative positions of objects

**Layer 3 — Conv(64, 3×3, stride=1):**
- 3×3 kernels, stride 1 — captures fine-grained spatial relationships
- Maintains 7×7 — no further downsampling
- Combines Layer 2 features into complex spatial arrangements
- Encodes the full spatial structure of the game state

### The Fully Connected Layers

After the convolutions, the 64×7×7 = 3,136 feature map values are flattened into a single vector.

**Linear(3136 → 512) + ReLU — The Representation Layer:**
This is the most important layer for this study. It compresses 3,136 values down to 512 values, forcing the network to learn a compact, abstract encoding of the game state. This 512-dimensional vector is what we analyse.

**Linear(512 → n_actions) — The Q-value Head:**
Maps the representation to Q-values for each possible action:
- Pong has 6 actions (NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE — though in practice only 3 are used)
- Breakout has 4 actions (NOOP, FIRE, RIGHT, LEFT)

### The Forward Hook

To extract the 512-dimensional representation without modifying the forward pass, we use a **PyTorch forward hook**:

```python
def hook(module, input, output):
    self.representation = output.detach().cpu()

self.fc_repr.register_forward_hook(hook)
```

Every time the network processes an input (whether during training or evaluation), the hook fires and saves the 512-dim activation to `model.representation`. The analysis scripts then collect these vectors across thousands of game steps to build the dataset for t-SNE.

### Network Size Variants (for Ablation Studies)

Three scale variants are provided via the `net_scale` parameter:

| Scale | Filters | Hidden | Parameters |
|---|---|---|---|
| small | [16, 32, 32] | 256 | ~0.4M |
| **medium (default)** | **[32, 64, 64]** | **512** | **~1.7M** |
| large | [64, 128, 128] | 1024 | ~6.9M |

The main 4 runs all use `medium`. The ablation studies test whether representation quality changes with network capacity.

---

## 11. The Representation Layer — The Heart of the Study

### 11.1 What Is a Representation?

In neural networks, a **representation** is the activation pattern of an intermediate layer in response to an input. It is a learned, compressed encoding of the input that captures the features most relevant for the task.

The 512-dimensional representation vector is what the agent has learned to "see" about a game state. After training:
- A state showing "ball moving toward the top-left corner of the Pong court" becomes a specific 512-dim vector
- A state showing "breakout brick pattern with 3 rows remaining on the right side" becomes a different 512-dim vector
- If the two games share structure (ball, paddle), their representations might share structure too

### 11.2 Why 512 Dimensions?

512 is a balance between:
- **Expressiveness:** Enough dimensions to encode complex game states
- **Compression:** Enough compression to force abstraction away from pixel details
- **Tractability:** Small enough to apply t-SNE efficiently

The representation layer is the bottleneck between the raw pixel features extracted by the CNN and the final Q-value prediction. This bottleneck forces the network to preserve only task-relevant information.

### 11.3 What We Do With Representations

For each of the 4 trained agents, we:

1. Load a trained checkpoint
2. Run the agent in the game environment for 5,000 steps (in near-greedy mode: ε=0.05)
3. At each step, record the 512-dim representation vector (via the forward hook)
4. Also record: the action taken, the reward received, the cumulative episode reward
5. Save everything as a `.npz` file

This gives us 5,000 data points per checkpoint, each a 512-dim vector describing one moment in a game.

We repeat this for **every checkpoint** (saved every 500,000 training steps), so we can track how representations evolve during training.

---

## 12. How Training Works — Step by Step

### 12.1 The Training Loop

Each training run executes millions of the following steps:

```
For each timestep t:
    1. Observe state s_t (4 stacked frames from wrapped env)
    2. Select action:
       - With probability ε: random action (exploration)
       - With probability 1-ε: argmax Q(s_t, a; θ) (exploitation)
    3. Execute action, receive reward r_t, observe next state s_{t+1}
    4. Store (s_t, a_t, r_t, s_{t+1}, done) in replay buffer
    5. If |buffer| ≥ batch_size:
       a. Sample batch of 32 transitions from buffer
       b. Compute TD targets using target network
       c. Compute loss = MSE(Q(s, a; θ), targets)
       d. Backpropagate gradients through online network
       e. Clip gradients to max norm 10.0
       f. Adam optimiser step
    6. Decay ε: ε ← max(0.01, ε - (1.0 - 0.01) / 100000)
    7. Every 1000 steps: θ⁻ ← θ (hard target network update)
    8. Every 500k steps: save checkpoint to disk
```

### 12.2 Epsilon-Greedy Exploration

The agent uses **epsilon-greedy** exploration to balance:
- **Exploration:** Trying new actions to discover better strategies
- **Exploitation:** Using what it already knows to maximise reward

ε starts at 1.0 (completely random) and decays linearly to 0.01 over 100,000 steps. After that, the agent mostly exploits its learned policy (99% greedy) but still occasionally explores (1% random).

Early in training, this is crucial: the agent knows nothing, so random exploration is efficient. Later, the agent has learned enough that random actions mostly hurt performance.

### 12.3 The Replay Buffer

The replay buffer stores up to 100,000 transitions in a circular array (new entries overwrite the oldest). Key implementation details:

- **Storage format:** States stored as `uint8` (0–255), not `float32` (0.0–1.0). This saves 4× RAM — critical since each state is 4×84×84 = 28,224 bytes, and 100k of them would be 2.8GB as float32 but only 700MB as uint8.
- **Normalisation:** Division by 255.0 happens when sampling a batch, not when storing. The GPU operates on the float32 values.
- **Random sampling:** Each `sample()` call draws a random batch of 32 transitions without replacement. This ensures the agent doesn't overfit to recent experience.

### 12.4 Gradient Clipping

Before applying the optimiser update, all gradient values are clipped to a maximum L2 norm of 10.0:

```python
nn.utils.clip_grad_norm_(online_net.parameters(), 10.0)
```

This prevents **exploding gradients** — a situation where a single very unexpected experience (large TD error) causes an extremely large gradient update that destabilises the network. Gradient clipping ensures updates remain bounded.

---

## 13. Key Hyperparameters and Why

All 4 runs use identical hyperparameters. This is deliberate — it ensures comparability.

| Hyperparameter | Value | Justification |
|---|---|---|
| Learning rate | 1e-4 | Standard Adam LR for Atari DQN; too high causes instability |
| Replay buffer | 100,000 | Reduced from original 1M for memory feasibility; still sufficient |
| Batch size | 32 | Standard; good balance of compute and gradient quality |
| Gamma (discount) | 0.99 | High discount — agent values future rewards highly |
| Target update | 1,000 steps | Hard copy every 1k steps; frequent enough to track learning |
| Epsilon start | 1.0 | Begin fully random to explore broadly |
| Epsilon end | 0.01 | Never fully greedy — 1% random action prevents exploitation lock-in |
| Epsilon decay | 100,000 steps | ~5% of Pong training; long enough to explore, short enough to converge |
| Gradient clip | 10.0 | Prevents gradient explosions from rare large TD errors |
| Frame skip | 4 | Standard; agent acts at 15Hz (60 / 4) |
| Frame stack | 4 | Provides ~267ms of motion history |
| Input size | 84×84 | Standard since Mnih et al. (2015) |
| Checkpoint freq | 500,000 steps | Needed for temporal t-SNE; provides 4 snapshots for Pong, 10 for Breakout |
| Seed | 42 | Fixed for reproducibility |
| Pong steps | 2,000,000 | Sufficient for convergence; well-established benchmark |
| Breakout steps | 5,000,000 | Harder game; needs more training |

---

## 14. The Analysis Pipeline — All 17 Figures

### Step 1: Extract Representations

**Script:** `analysis/extract_representations.py`

For each trained agent and each checkpoint, the script:
1. Loads the saved model weights
2. Runs the agent in the game for 5,000 steps
3. Records the 512-dim representation at each step
4. Saves as `.npz` files with metadata (game, algorithm, training step, action, reward)

Output: ~60 `.npz` files (15 checkpoints × 4 agents, approximately), each containing 5,000 representation vectors.

### Step 2: t-SNE Visualisation (Figures 4–10)

**Script:** `analysis/tsne_visualisation.py`

### Step 3: Activation Analysis (Figures 1–3, 13–14)

**Script:** `analysis/activation_analysis.py`

### Step 4: Saliency Maps (Figures 11–12)

**Script:** `analysis/saliency_maps.py`

---

### All 17 Figures — Detailed Explanations

#### Figure 1: Training Curves — Pong
**File:** `training_curves_pong.png`

**What it shows:** Episode reward over training steps for DQN and DDQN on Pong. The raw reward is plotted with a rolling average smoothing (window=20 episodes) and an interquartile range shading.

**What to look for:**
- Both algorithms should start at -21 (losing every point to the built-in AI)
- Both should converge toward +21 (winning every point) by 2M steps
- DDQN may converge faster or reach higher peak performance
- The shaded region shows reward variance — DDQN may show lower variance (more stable learning)

**What it tells us:** Performance comparison — the "headline result" that contextualises the representation analysis.

---

#### Figure 2: Training Curves — Breakout
**File:** `training_curves_breakout.png`

**What it shows:** Same as Figure 1 but for Breakout over 5M steps.

**What to look for:**
- Breakout rewards are not on -21 to +21 scale — they accumulate as bricks are broken
- The game is significantly harder: expect slow initial learning (~2M steps before consistent positive rewards)
- DDQN is expected to show an advantage in Breakout's more complex environment

---

#### Figure 3: Q-Value Overestimation
**File:** `qvalue_overestimation.png`

**What it shows:** Mean maximum Q-value over training for all 4 runs (two subplots: Pong, Breakout). The mean max Q-value is the average, across a minibatch, of the maximum Q-value across actions — i.e., the agent's estimated value of the best action from each state.

**What to look for:**
- DQN lines should trend upward over training — this is overestimation bias accumulating
- DDQN lines should remain more stable (lower, more realistic Q-values)
- The gap between DQN and DDQN is direct empirical evidence of overestimation

**Why it matters academically:** This is the core quantitative claim of van Hasselt et al. (2016) — we reproduce and extend it. It directly connects algorithm design to measurable bias.

---

#### Figure 4: t-SNE — Game Effect (DQN)
**File:** `tsne_game_effect_dqn.png`

**What it shows:** t-SNE scatter plot with two colours: blue points (DQN/Pong representations) and orange points (DQN/Breakout representations). Both sets are projected together into 2D.

**What to look for:**
- **Separated clusters:** Points of the same colour cluster together, far from the other colour — the game being played produces fundamentally different representations. The agent's internal model is game-specific.
- **Mixed/overlapping clusters:** Both colours appear in the same t-SNE regions — the games share representational structure despite their visual differences.
- **Substructure:** Within each game cluster, are there sub-clusters? These might correspond to different game phases (early vs late in a Breakout level, serving vs rallying in Pong).

**Interpretation:** If clusters separate by game, it confirms that the type of game you play determines the representation structure more than the algorithm — which is our H1 hypothesis.

---

#### Figure 5: t-SNE — Game Effect (DDQN)
**File:** `tsne_game_effect_ddqn.png`

Same as Figure 4 but for DDQN. Comparing Figures 4 and 5:
- Does DDQN show cleaner game-specific separation than DQN?
- Does DDQN show different overlap patterns?

---

#### Figure 6: t-SNE — Algorithm Effect (Pong)
**File:** `tsne_algo_effect_pong.png`

**What it shows:** t-SNE with two colours: blue (DQN/Pong) and green (DDQN/Pong). Same game, different algorithms.

**What to look for:**
- **Separated clusters:** The algorithm used to train shapes representations differently even on the same game. DDQN produces a distinct representational geometry.
- **Overlapping:** Both algorithms converge to similar representations for Pong — the game constrains representations more than the algorithm.
- **Compactness:** DDQN clusters may be tighter (lower intra-cluster variance) — a sign of more structured, less noisy representations.

**This is the key algorithm effect figure.** It directly tests whether DDQN's cleaner gradient signal produces more organised representations.

---

#### Figure 7: t-SNE — Algorithm Effect (Breakout)
**File:** `tsne_algo_effect_breakout.png`

Same as Figure 6 but for Breakout. Combined with Figure 6, this answers whether the algorithm effect is consistent across games or stronger on one.

---

#### Figure 8: t-SNE — All 4 Agents
**File:** `tsne_all_agents.png`

**What it shows:** All four agents (up to 1,500 points each, 6,000 total) in a single t-SNE plot with four colours.

**What to look for:**
- **2 game clusters (each containing both DQN and DDQN):** Game effect dominates — the game determines representations more than the algorithm
- **4 separate clusters (one per agent):** Both game and algorithm effects are strong and independent
- **2 algorithm clusters (each containing both games):** Algorithm dominates — extremely unlikely given the different game structures
- **One big blob:** No meaningful structure — representations are arbitrary and uninterpretable

This is the most visually compelling figure for a dissertation — it shows at a glance whether game or algorithm is the primary organising principle of the representation space.

---

#### Figure 9: t-SNE — Coloured by Reward
**File:** `tsne_by_reward.png`

**What it shows:** Four subplots (one per agent), each a t-SNE of that agent's representations. Points are coloured by cumulative episode reward (red = low reward, green = high reward).

**What to look for:**
- **Gradient structure:** High-reward and low-reward states should cluster in different t-SNE regions if the representation encodes game value
- **No gradient structure:** Representations are purely perceptual (about what is visible), not evaluative (about how good the situation is)

**Why it matters:** If high-reward states cluster separately, the representation layer encodes not just "what I see" but "how good the situation is" — evidence of value-sensitive representations, not just perceptual features.

---

#### Figure 10: Temporal Evolution of Representations
**File:** `tsne_temporal_dqn_pong.png` (and one for DDQN/Breakout)

**What it shows:** A grid of t-SNE plots, one per checkpoint (e.g., steps 500k, 1M, 1.5M, 2M for Pong). Each plot shows the representation geometry at that point in training.

**What to look for:**
- **Early training:** Diffuse, unstructured cloud — representations are random
- **Mid training:** Clusters beginning to form as the agent learns basic strategies
- **Late training:** Well-separated, compact clusters — mature, structured representations
- **DDQN vs DQN:** Does DDQN's structure emerge earlier? Are DDQN's final representations more compact?

This is the most compelling evidence for *when* representational structure emerges during learning.

---

#### Figure 11: Saliency Maps — Pong
**File:** `saliency_pong.png`

**What it shows:** Grad-CAM (Gradient-weighted Class Activation Mapping) heatmaps overlaid on game frames. For each agent-game combination, several game frames are shown with a colour overlay indicating which pixels most influenced the Q-value prediction.

**How Grad-CAM works:**
Gradients of the chosen action's Q-value with respect to the last convolutional layer activations are computed. Large gradients indicate that changing those pixels would most affect the predicted Q-value — these are the "important" pixels. The gradients are averaged over feature maps and upsampled to the original frame size.

**What to look for:**
- **Ball focus:** Does the agent primarily attend to the ball position? (Expected — moving the paddle toward the ball is the key skill)
- **Paddle focus:** Does it also track its own paddle? (Yes, necessary for positioning)
- **DQN vs DDQN differences:** Does DDQN show more focused, concentrated attention? DQN's overestimation noise might cause diffuse or inconsistent attention patterns.
- **Background noise:** Does the agent attend to irrelevant screen regions? (Score display, background patterns)

---

#### Figure 12: Saliency Maps — Breakout
**File:** `saliency_breakout.png`

Same as Figure 11 but for Breakout. Additional elements to look for:
- **Brick focus:** Does the agent attend to specific bricks about to be hit?
- **Tunnel effect:** Experienced Breakout players dig a tunnel along one wall — does the saliency shift toward the sides as this strategy emerges?

---

#### Figure 13: Dead Neuron Analysis
**File:** `dead_neurons.png`

**What it shows:** The fraction of neurons in the 512-dim representation layer that are "dead" (chronically inactive) at each training checkpoint, plotted over training steps for all 4 agents.

**What a dead neuron is:** A neuron is defined as dead if its output is 0.0 in more than 95% of observed states. Since the representation layer uses ReLU activation (output = max(0, input)), neurons can become permanently negative — they never fire. These neurons contribute nothing to the representation.

**Why dead neurons matter:** Dead neurons represent wasted capacity — parameters that take up memory and compute but carry no information. A network with many dead neurons is using its capacity inefficiently.

**What to look for:**
- **DQN > DDQN dead fraction:** DQN's overestimation noise creates erratic gradient signals that may kill more neurons. DDQN's cleaner gradients should maintain more active neurons.
- **Increasing over training:** Some dead neuron accumulation is normal as specialisation occurs
- **Plateau:** A stable dead neuron fraction indicates a mature network

---

#### Figure 14: Cosine Similarity
**File:** `cosine_similarity.png`

**What it shows:** The cosine similarity between the mean Pong representation and the mean Breakout representation at each checkpoint, separately for DQN and DDQN.

**What cosine similarity measures:**
```
cos_sim(a, b) = (a · b) / (||a|| × ||b||)
```
A value of 1.0 means the two vectors point in the same direction (maximum similarity). 0.0 means they are orthogonal (no shared structure). -1.0 means they point in opposite directions.

For mean representations: we average the 512-dim vectors over 5,000 steps for each agent, then compute the cosine similarity between the Pong mean and the Breakout mean.

**What to look for:**
- **Similarity > 0.2:** Non-trivial shared structure — both games activate similar directions in representation space, reflecting shared ball/paddle physics
- **DDQN higher than DQN:** DDQN's cleaner gradients produce representations more aligned with the underlying game structure, revealing more cross-game similarity
- **Trend over training:** Does similarity increase as the agent learns? (Suggests shared features emerge through learning, not random initialisation)

---

#### Figures 15–17: Ablation Studies

These three figures test the sensitivity of the main results to specific hyperparameters. All ablations are run on Pong (faster game) with DQN only.

**Figure 15 — Network Size Ablation (`ablation_network_size.png`):**
Trains small, medium, and large CNN variants. Tests: Does a larger network produce more structured representations? Is the main result (game effect vs algorithm effect) stable across network sizes?

**Figure 16 — Learning Rate Ablation (`ablation_lr.png`):**
Trains with LR = 5e-5 (low), 1e-4 (baseline), and 5e-4 (high). Tests: Is representation quality sensitive to learning rate? Does a lower LR (slower, more stable learning) produce cleaner representations?

**Figure 17 — Replay Buffer Ablation (`ablation_buffer.png`):**
Trains with buffer size 10,000 (small) vs 100,000 (baseline). Tests: Does a smaller, less diverse replay buffer produce more correlated, less structured representations?

---

## 15. t-SNE — What It Is and Why We Use It

### 15.1 The Problem With High-Dimensional Data

Our representations are 512-dimensional vectors. Humans cannot perceive 512 dimensions. We need a way to visualise the geometric structure of these vectors — how they cluster, how they spread, which groups are similar to which.

### 15.2 Principal Component Analysis (PCA) — Why Not?

PCA projects high-dimensional data to lower dimensions by finding directions of maximum variance (principal components). It's linear, fast, and well-understood.

However, for this task PCA has a key limitation: it only captures linear relationships. Representations might have non-linear structure — clusters that are curved, nested, or interleaved in ways that linear projection collapses into a single overlapping blob.

### 15.3 t-SNE: t-Distributed Stochastic Neighbour Embedding

t-SNE (Maaten & Hinton, 2008) is a **non-linear dimensionality reduction** method specifically designed to reveal cluster structure in high-dimensional data.

**How it works (conceptually):**

1. **In high-dimensional space:** For each data point, compute a probability distribution over its neighbours — nearby points get high probability, distant points get low probability. Uses a Gaussian distribution.

2. **In 2D space:** Initialise points randomly. Define a similar probability distribution over 2D neighbours, but using a **t-distribution** (heavier tails than Gaussian).

3. **Optimise:** Adjust the 2D positions to make the 2D neighbourhood distributions match the high-dimensional ones as closely as possible. Uses gradient descent to minimise KL divergence between the two distributions.

**Why the t-distribution?** The heavier tails of the t-distribution in 2D allow distant clusters to stay far apart, while the Gaussian in high-D keeps similar points together. This "crowding problem" fix is what makes t-SNE much more effective than earlier non-linear methods.

**The result:** Points that were close in 512D tend to appear close in 2D. Points that were far apart in 512D tend to appear far apart. Clusters that existed in 512D become visually clear clusters in 2D.

### 15.4 Key Properties and Limitations

**Preserves local structure:** t-SNE faithfully represents which points are neighbours in high-D. You can trust that a cluster in the t-SNE plot corresponds to a genuine cluster in the representation space.

**Does NOT preserve global distances:** The distance between two clusters in a t-SNE plot is **not** proportional to the true distance between those clusters in high-D. Do not interpret "these two clusters are far apart in the plot" as "these two clusters are very different" — they might simply be placed far apart by the optimisation.

**Stochastic:** Running t-SNE twice with different random seeds gives different-looking plots, even from the same data. The cluster assignments are consistent; the specific 2D layout varies. We fix `random_state=42` for reproducibility.

**Perplexity:** The key hyperparameter (we use 30) controls the effective number of neighbours considered. Values between 5 and 50 are typical. Higher perplexity emphasises global structure; lower emphasises local.

### 15.5 Why t-SNE Is the Right Tool Here

t-SNE was directly applied to DQN representations in Zahavy et al. (2016), "Graying the Black Box", which showed that DQN representations cluster by game state semantics (player position, game phase). Our study extends this to the cross-game, cross-algorithm comparison.

The method is well-established, interpretable to a scientific audience, and directly interpretable by visual inspection — no arbitrary thresholds needed. You look at the plot and can immediately see whether clusters separate by game, algorithm, or both.

---

## 16. Hypotheses — What We Expect to Find

Based on theory and prior literature, we have 6 testable hypotheses:

### H1 — Game Effect Dominates (t-SNE)
**Prediction:** In the all-agents t-SNE (Fig 8), points will cluster primarily by game (Pong vs Breakout), not by algorithm (DQN vs DDQN).

**Reasoning:** The games have fundamentally different visual structures. The CNN must learn different spatial features for each game. This game-specific visual structure should create distinct representation clusters, regardless of the algorithm used.

**Falsified if:** Points cluster primarily by algorithm — which would suggest the learning dynamics (not the game content) determine representational geometry.

### H2 — DDQN Produces Tighter Clusters (t-SNE compactness)
**Prediction:** In algorithm effect plots (Figs 6–7), DDQN representations will show more compact, lower-variance clusters than DQN.

**Reasoning:** DDQN's corrected gradient signal is less noisy. Less noise in training → more consistent gradient updates → more stable, organised internal representations. DQN's overestimation noise may push some neurons into inconsistent activation patterns, spreading the cluster.

**Falsified if:** DQN and DDQN show identical cluster compactness.

### H3 — Non-Trivial Cross-Game Similarity
**Prediction:** Cosine similarity between Pong and Breakout mean representations will exceed 0.2 (Fig 14).

**Reasoning:** Both games feature a ball and a paddle. The CNN's convolutional layers will develop similar ball-detection and paddle-detection filters regardless of game, since these are visually identical objects. The shared low-level visual features should create shared representational directions.

**Falsified if:** Cosine similarity ≈ 0.0 — representations are completely orthogonal, meaning the games share no internal representational structure.

### H4 — DQN Has More Dead Neurons
**Prediction:** DQN will show a higher fraction of dead neurons in the 512-dim layer throughout training (Fig 13).

**Reasoning:** DQN's noisy, biased gradient signal is more likely to drive neurons into chronically negative states (ReLU dead zone). DDQN's cleaner gradients maintain a more active representation layer.

**Falsified if:** DQN and DDQN have similar dead neuron fractions.

### H5 — DQN Q-Values Drift Upward
**Prediction:** DQN's mean max Q-value will increase monotonically over training beyond what Breakout and Pong scores can justify. DDQN's will remain stable (Fig 3).

**Reasoning:** This is the core theoretical claim of van Hasselt et al. (2016) — we are reproducing it empirically.

**Falsified if:** Both algorithms show similar Q-value dynamics — which would suggest our reduced buffer size (100k vs the original 1M) eliminates the overestimation effect.

### H6 — Ball-Focused Saliency in Both Agents
**Prediction:** Grad-CAM saliency maps for both DQN and DDQN on both games will show highest activation around the ball position (Figs 11–12).

**Reasoning:** The ball is the primary task-relevant object in both games. Tracking it is necessary and sufficient for a basic policy. Both algorithms should learn to attend to it, regardless of other representational differences.

**Interesting extension:** Does DDQN show more focused saliency (smaller high-activation region centred tightly on the ball) while DQN shows more diffuse attention?

---

## 17. Training Progress So Far

### Run 1 — DQN / Pong ✅ Complete

| Checkpoint | Steps | Reward | Notes |
|---|---|---|---|
| 1 | 500,000 | ~-10 | Starting to hit ball |
| 2 | 1,000,000 | +8 | Winning most points |
| 3 | 1,500,000 | — | Continuing to improve |
| 4 | 2,000,000 | — | Near-optimal play |

**Confirmed:** Reward progression from -21 (random) to +8 by 1M steps is exactly as expected from the literature. This validates the implementation is correct.

Normal DQN/Pong learning trajectory:
- Steps 0–100k: Reward -21 to -18 (random play, agent just hitting the ball occasionally)
- Steps 100k–300k: Reward -15 to -5 (agent learning to reach the ball)
- Steps 500k–1M: Reward -5 to +10 (agent winning some games)
- Steps 1M–2M: Reward +10 to +21 (near-optimal play)

### Run 2 — DQN / Breakout ✅ Complete (5M steps, 11 checkpoints)

### Run 3 — DDQN / Pong 🔄 In Progress (~500k/2M steps as of March 24, 2026)

### Run 4 — DDQN / Breakout ❌ Not Started Yet (starts when Run 3 finishes)

### Hardware

All training runs on **GCP VM: n1-standard-4 + NVIDIA T4 GPU (16GB VRAM)**, using Spot pricing (~$0.13/hr). Estimated total cost for all 4 runs: ~$4–8.

---

## 18. Ablation Studies

After the 4 main runs complete, 5 ablation experiments are run on Pong with DQN only. Their purpose is to test the robustness of the main findings:

| Ablation | Config | What it tests |
|---|---|---|
| Small network | filters [16,32,32], hidden=256 | Does network capacity affect representation quality? |
| Large network | filters [64,128,128], hidden=1024 | Is more capacity always better? |
| Low LR | lr=5e-5 | Does slower learning produce more structured representations? |
| High LR | lr=5e-4 | Does faster learning destabilise representations? |
| Small buffer | buffer=10,000 | Does less diverse replay produce worse representations? |

**Key question:** If all ablations produce similar t-SNE cluster structure, our findings are robust. If representation structure changes dramatically with hyperparameters, we must caveat our claims accordingly.

---

## 19. Codebase Architecture

```
rl_project/
│
├── train.py                    ← Main training entry point. Reads a YAML config,
│                                  builds the environment and agent, runs the
│                                  training loop, saves checkpoints and logs.
│
├── run_all.py                  ← Orchestrator. Runs all 4 training runs
│                                  sequentially, auto-resuming from the latest
│                                  checkpoint. Then runs the full analysis pipeline.
│
├── setup_env.py                ← One-time setup. Detects GPU/MPS/CPU, installs
│                                  correct PyTorch version, installs all packages.
│
├── requirements.txt            ← Python package dependencies
│
├── envs/
│   └── wrappers.py             ← All 7 Atari preprocessing wrappers + make_atari_env()
│
├── models/
│   └── cnn.py                  ← The shared CNN (AtariCNN). Identical for DQN and
│                                  DDQN. Forward hook on fc_repr for representation
│                                  extraction.
│
├── agents/
│   ├── dqn.py                  ← DQNAgent. Select action, learn(), target update,
│   │                              epsilon decay, replay buffer interaction.
│   └── ddqn.py                 ← DoubleDQNAgent. Inherits DQNAgent. Only overrides
│                                  learn() with the Double Q-learning target.
│                                  3 lines of code different from DQN.
│
├── utils/
│   ├── replay_buffer.py        ← Circular buffer, uint8 storage, float32 at sample
│   ├── logger.py               ← Writes CSV logs and TensorBoard summaries
│   └── checkpoint.py           ← Save/load full training state (model, optimizer,
│                                  epsilon, step count)
│
├── analysis/
│   ├── extract_representations.py  ← Loads checkpoints, runs agents, collects
│   │                                  512-dim vectors, saves as .npz
│   ├── tsne_visualisation.py       ← All t-SNE figures (Figs 4–10)
│   ├── activation_analysis.py     ← Training curves, Q-value, dead neurons,
│   │                                  cosine similarity (Figs 1–3, 13–14)
│   └── saliency_maps.py            ← Grad-CAM saliency (Figs 11–12)
│
└── experiments/configs/
    ├── run1_dqn_pong.yaml          ← DQN, ALE/Pong-v5, 2M steps
    ├── run2_dqn_breakout.yaml      ← DQN, ALE/Breakout-v5, 5M steps
    ├── run3_ddqn_pong.yaml         ← DDQN, ALE/Pong-v5, 2M steps
    ├── run4_ddqn_breakout.yaml     ← DDQN, ALE/Breakout-v5, 5M steps
    ├── ablation_net_small.yaml     ← DQN/Pong, small CNN
    ├── ablation_net_large.yaml     ← DQN/Pong, large CNN
    ├── ablation_lr_low.yaml        ← DQN/Pong, lr=5e-5
    ├── ablation_lr_high.yaml       ← DQN/Pong, lr=5e-4
    └── ablation_buffer_small.yaml  ← DQN/Pong, buffer=10k
```

### Data Flow

```
Game pixels (210×160×3, uint8)
    ↓ [wrappers.py]
Preprocessed state (4×84×84, uint8)
    ↓ stored in
Replay buffer (100k circular)
    ↓ sampled as
Mini-batch (32×4×84×84, uint8)
    ↓ normalised to float32 [0,1]
    ↓ passed to
CNN (models/cnn.py)
    ↓
Conv features → Flatten → [forward hook fires]
    ↓                              ↓
Q-values (n_actions)        Representation (512-dim)
    ↓                              ↓
Loss + backprop             Saved to .npz files
    ↓                              ↓
Gradient update             t-SNE / analysis
```

---

## 20. Academic References

### Core Papers

**1. Mnih et al. (2015) — "Human-level control through deep reinforcement learning"**
*Nature, 518, 529–533*

The original DQN paper. Introduced the convolutional neural network for Atari Q-learning, the experience replay buffer, and the target network. First demonstrated superhuman performance on a wide range of Atari games directly from pixels. All our architecture decisions (CNN structure, 84×84 input, 4-frame stack) derive from this paper.

**2. van Hasselt, Guez & Silver (2016) — "Deep Reinforcement Learning with Double Q-learning"**
*AAAI 2016*

Introduced Double DQN. Identified and formally characterised the overestimation bias in DQN. Showed that decoupling action selection (online network) from action evaluation (target network) removes this bias. Demonstrated improvement across all 49 Atari games. The DDQN agent in our study is a direct implementation of this paper.

**3. Zahavy et al. (2016) — "Graying the Black Box: Understanding DQNs"**
*ICML 2016*

The most directly related prior work to our study. Applied t-SNE to DQN representations trained on multiple Atari games. Found that representations cluster by game state semantics — the agent's internal state corresponds to meaningful game situations. Our study extends this to the cross-game, cross-algorithm comparison they did not perform.

**4. Maaten & Hinton (2008) — "Visualizing Data using t-SNE"**
*Journal of Machine Learning Research, 9, 2579–2605*

The original t-SNE paper. Our visualisation methodology is drawn directly from this work. Essential citation for the analysis methodology.

**5. Bellemare, Naddaf, Veness & Bowling (2013) — "The Arcade Learning Environment: An Evaluation Platform for General Agents"**
*JAIR, 47, 253–279*

Introduced the ALE (Arcade Learning Environment). Defined the Atari benchmark. Essential citation for the experimental environment.

### Background References

**6. Hessel et al. (2018) — "Rainbow: Combining Improvements in Deep Reinforcement Learning"**
*AAAI 2018*

Combined 6 improvements to DQN (including Double DQN, prioritised replay, duelling networks, multi-step returns, distributional RL, and noisy networks) into a single "Rainbow" agent. Provides context for where DDQN fits in the broader landscape of DQN extensions. Useful for dissertation background section.

**7. Sutton & Barto (2018) — "Reinforcement Learning: An Introduction" (2nd ed.)**
*MIT Press*

The standard textbook for reinforcement learning theory. Chapter 6 covers TD learning and Q-learning. Essential background reference for the MDP framework, Bellman equations, and value function approximation.

**8. Goodfellow, Bengio & Courville (2016) — "Deep Learning"**
*MIT Press*

Standard reference for deep learning background — convolutional networks, backpropagation, regularisation. Supports the neural network architecture section of the dissertation.

**9. Selvaraju et al. (2017) — "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"**
*ICCV 2017*

Introduced Grad-CAM, the saliency map technique used in our Figures 11–12. Essential citation for the saliency analysis methodology.

### Summary Table

| Paper | Year | Relevance |
|---|---|---|
| Mnih et al. | 2015 | DQN — our base algorithm |
| van Hasselt et al. | 2016 | DDQN — our comparison algorithm |
| Zahavy et al. | 2016 | t-SNE on DQN — direct prior work |
| Maaten & Hinton | 2008 | t-SNE methodology |
| Bellemare et al. | 2013 | Atari environment |
| Hessel et al. | 2018 | Context: DQN extensions |
| Sutton & Barto | 2018 | RL theory background |
| Goodfellow et al. | 2016 | Deep learning background |
| Selvaraju et al. | 2017 | Grad-CAM methodology |

---

## Summary for a Conversation with Your Professor

**In one sentence:** We train DQN and Double DQN agents on Pong and Breakout, extract the 512-dimensional internal representations they develop, and use t-SNE to compare whether game content or algorithmic choice more strongly shapes those representations.

**The key design principle:** Both algorithms use an *identical* neural network. The only difference is 3 lines of code in the loss function. Therefore any observed representational difference is purely algorithmic.

**Why DDQN might produce better representations:** DDQN removes a systematic overestimation bias by decoupling action selection from action evaluation. Cleaner gradient signal → more structured internal representations → tighter t-SNE clusters and fewer dead neurons.

**The expected headline finding:** t-SNE clusters separate primarily by game (Pong vs Breakout), not by algorithm. The game you play shapes your internal model of the world more than the learning rule you use — but the learning rule modulates the *quality* of that representation.

**What makes this a contribution:** The game-effect vs algorithm-effect comparison has not been done before with this level of isolation (same architecture, same hyperparameters, same games, only algorithm varies). Zahavy et al. (2016) showed t-SNE reveals semantics within a single agent; we extend to cross-game and cross-algorithm comparison.

---

*Document version: 1.0 | Created: 2026-03-24 | rl_project — Masters Dissertation Component*

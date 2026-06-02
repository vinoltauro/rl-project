# 📖 Glossary — Every Term in One Place

> Quick definitions. When you forget a term mid-conversation, look here first.

---

## A

**Action space** — the set of all possible moves an agent can make. Pong has 6 actions, Breakout has 4.

**Activation** — the output value of a neuron. If it is zero, the neuron is "dead" for that input.

**Adam** — the optimiser we use. Adaptive learning rate algorithm. More robust than plain gradient descent.

---

## B

**Backbone** — the conv layers of the network, used as a fixed feature extractor. "Frozen backbone" = conv layers are locked, only upper layers train.

**Batch size** — how many experiences are sampled from the replay buffer for each training update. We use 32.

**Bellman equation** — the recursive relationship that defines Q-values: Q(s,a) = r + γ·max Q(s',a'). The foundation of Q-learning.

---

## C

**Capacity loss** — the progressive reduction in the number of active neurons over training. Dead neurons represent lost capacity.

**Catastrophic forgetting** — when a neural network learns a new task and completely overwrites what it learned about an old task.

**CKA (Centred Kernel Alignment)** — a similarity metric for neural network layers. Returns 0 (completely different) to 1 (identical). See [[05_cka]].

**Chimera network** — a hybrid network built by taking conv layers from one trained agent and upper layers from another.

**Checkpoint** — a saved snapshot of the model weights at a specific training step.

**Conv layer (convolutional layer)** — a network layer that applies learned filters to detect local patterns in images. Our network has three.

**Cosine similarity** — measures the angle between two vectors. 1 = pointing same direction, 0 = perpendicular, -1 = opposite. We use it to compare mean Pong vs Breakout representations.

---

## D

**Dead neuron** — a neuron that outputs zero for all inputs because it is stuck in the ReLU dead zone. Cannot be recovered by gradient updates.

**DDQN (Double DQN)** — variant of DQN that uses the online network to select the best action and the target network to evaluate it. Reduces overestimation bias. See [[02_dqn_ddqn]].

**DQN (Deep Q-Network)** — reinforcement learning algorithm that uses a CNN to approximate Q-values from raw pixels. See [[02_dqn_ddqn]].

---

## E

**Embedding** — another word for representation. The compressed vector that encodes the current state.

**Epsilon (ε)** — the probability of taking a random action (exploration). Starts at 1.0 (fully random), decays to 0.01 (mostly greedy) over 100k steps.

**Experience replay** — storing past transitions in a buffer and sampling randomly for training. Breaks temporal correlations.

---

## F

**fc_out** — the output layer of the network. Maps the 512-dim representation to Q-values. Shape depends on number of actions (6 for Pong, 4 for Breakout).

**fc_repr** — the 512-dimensional fully connected representation layer. The layer we analyse. See [[03_representations]].

**Fine-tune** — continue training a pre-trained network on a new task. "Full fine-tune" = all layers are updated.

**Forward hook** — a PyTorch mechanism to capture layer activations without modifying the network. We use it to silently record fc_repr activations.

**Frame stacking** — stacking the last 4 grayscale frames as input channels. Gives the agent implicit velocity information.

**Frame skip** — repeating the same action for 4 consecutive frames. Speeds up training and matches human reaction time.

---

## G

**Gamma (γ)** — discount factor. How much the agent values future rewards vs immediate rewards. We use 0.99 (very patient).

**Grad-CAM** — visualisation technique that highlights which pixels most influenced the agent's Q-value decision. See [[14_how_to_read_plots]].

**Gradient clipping** — capping gradient magnitudes to prevent exploding gradients. We clip at L2 norm ≤ 10.

---

## H

**Hard update** — copying target network weights directly from online network. We do this every 1,000 steps.

---

## L

**Learning rate** — how big each gradient update step is. We use 1e-4 with Adam.

**Linear CKA** — the specific variant of CKA we use. Computationally efficient and well-suited for comparing layer activations.

---

## O

**Online network** — the network being actively trained. Contrast with target network.

**Overestimation bias** — DQN's tendency to overestimate Q-values due to the max operator in the Bellman target.

---

## Q

**Q-value** — the expected future cumulative reward of taking action a in state s. What the network outputs.

**Q-function** — the mapping from (state, action) pairs to Q-values. The network approximates this.

---

## R

**ReLU (Rectified Linear Unit)** — activation function: f(x) = max(0, x). Simple and effective. Creates the dead neuron problem.

**Replay buffer** — memory storing past (state, action, reward, next_state, done) transitions. We use 100k capacity.

**Representation** — the 512-dim vector at fc_repr. The agent's internal description of the current game state. See [[03_representations]].

**Reward clipping** — clipping rewards to {-1, 0, +1}. Normalises across different games.

---

## S

**Saliency map** — a visual showing which parts of the input image most influence the network's output.

**Sequential training** — training on one task, then continuing on another without revisiting the first.

**Silhouette score** — quantitative measure of cluster quality. High score = well-separated clusters.

**Superposition** — the hypothesis (Elhage et al., 2022) that neurons can represent multiple features simultaneously using near-orthogonal directions.

---

## T

**Target network** — a periodically frozen copy of the online network used to compute stable training targets.

**t-SNE** — dimensionality reduction technique that preserves local structure. We use it to visualise 512-dim representations in 2D. See [[04_tsne]].

**Transfer learning** — using knowledge learned on one task to help learn another task.

---

## Z

**Zero-shot transfer** — applying a model trained on Task A directly to Task B with no additional training.

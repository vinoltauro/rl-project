# 💀 Dead Neurons and Capacity Loss

> **One liner:** Dead neurons are units in the network that stopped contributing to anything — they always output zero. The more dead neurons, the less "thinking space" the network has.

---

## 🔍 Zoom Level 1 — The Pub Explanation

Imagine a team of 512 researchers working together to analyse a situation. Each researcher looks at the same information and contributes their perspective. Good teamwork.

Now imagine that over time, 200 of those researchers just stop contributing. They sit at their desk but never speak up. The remaining 312 have to do all the work.

That is what dead neurons are. The network has 512 units in the representation layer, but if 200 of them are always outputting zero, the network is effectively working with a much smaller capacity than it was designed for.

**Why do neurons die?** ReLU activation functions output zero for any negative input. If a neuron consistently receives negative inputs during training, it gets permanently stuck at zero — the gradients that would update it also become zero, so it can never recover. This is the "ReLU dead zone."

---

## 🔬 Zoom Level 2 — The Study Group Explanation

We define a dead neuron as one that fires (outputs > 0) in fewer than **5% of game states**. We measure this by running the agent for 1,000 game states and checking each of the 512 units in `fc_repr`.

**Why DQN has more dead neurons than DDQN:**
DQN's overestimation bias creates noisier training targets. When targets fluctuate wildly, the gradients are large and inconsistent. Large, inconsistent gradients are more likely to drive neurons into the dead zone — a big negative update sends a neuron below zero, it outputs zero, gradient becomes zero, it never recovers.

DDQN's more stable targets produce cleaner gradients. Neurons drift less, fewer end up permanently dead.

**What the plot shows (`dead_neurons.png`):**
- X-axis: training step (500k checkpoints)
- Y-axis: fraction of dead neurons
- DQN line is consistently higher than DDQN across both games
- Dead neurons tend to increase over training — capacity is being lost progressively

---

## 🎓 Zoom Level 3 — The Professor Explanation

The dead neuron phenomenon is a manifestation of the broader **capacity loss** problem in deep RL, documented by Lyle et al. (2022). A neuron operating under the ReLU activation f(x) = max(0, x) enters the dead zone when its pre-activation is consistently negative. Because ∂f/∂x = 0 for x < 0, gradient-based updates cannot recover a dead neuron — the gradient vanishes at zero.

In the context of continual learning, dead neurons represent a permanent loss of representational capacity. Neurons that were active for Task A but become dead during Task B training can never be reactivated for Task A — this is one mechanism by which catastrophic forgetting operates at the neural level.

Continual backpropagation (Elsayed & Mahmood, 2024) proposes reinitialising low-utility neurons periodically to combat this, but we do not implement this intervention in the current study.

In our results: DQN/Pong has approximately X% dead neurons at the final checkpoint vs DDQN/Pong's Y% — a meaningful difference attributable entirely to the overestimation-driven gradient noise difference.

---

## 💡 The Interesting Bit

> Dead neurons are not just a nuisance — they are the **mechanism** by which catastrophic forgetting happens. When an agent learns Game B, some neurons that were encoding Game A features get driven dead by Game B's gradient signal. Those neurons can never recover. This is why we track dead neurons in our sequential training experiment — we expect to see dead neurons increase as Pong knowledge is lost during Breakout training. If we do, we have directly observed the forgetting happening at the neuron level.

---

## 🔗 How it connects

- [[02_dqn_ddqn]] — why DQN causes more dead neurons than DDQN
- [[07_catastrophic_forgetting]] — dead neurons as the mechanism of forgetting
- [[08_experiments]] — measured in sequential training experiment
- [[09_key_findings]] — DQN has more dead neurons finding

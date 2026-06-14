# Professor Meeting — 13 June 2026

---

## Raw Notes

- Report needs more structure
- Mechanisms/experiments need separate sections — what, how, pros, cons
- Sections should have: purpose, conduct, results
- Need proper terms for lower layers (representation layers) and higher layers (strategy/policy)
- "Predate, make consistent" — unclear, needs clarification at next meeting
- Professor not convinced with CKA — not policy-specific, uses random frames
- CKA comparison should be done during policy training (on-policy frames)
- Train same network architecture — 2 training policies for different games, compare representation layers (explain this better in report)
- Interleaved: professor wants 1 joint replay buffer (not 2 separate), episode-level alternation, game selected at start of episode, fixed steps per game rounded to episode boundary
- Sequential: after 1 game, deinitialise dead neurons before training on second game — read up on continual backpropagation
- Page 3: explain neural agent
- Page 16: impact of extra layer that learned game rules — interesting, was this trained till convergence?
- Look into superposition for the 2 Atari games

---

## Careful Analysis (reviewed for accuracy)

### 1. Report Structure
Each experiment section needs to be structured as:
**Purpose → Conduct/How → Results → Interpretation**

Currently the report separates Experimental Design (Section 3) from Results (Section 4). The professor wants them co-located per experiment — motivation, method, result, and what it means, all in one place. The Background section also needs each mechanism covered with: what it is, how it works, pros, cons — not just description.

This is a significant restructure of `main-v4.tex`.

---

### 2. Terminology — NEEDS CLARIFICATION BEFORE IMPLEMENTING

**⚠️ There is a potential terminology conflict that must be resolved with the professor first.**

The professor said: lower layers → "representation layers", higher layers → "strategy, policy."

But throughout our entire report and codebase we call `fc_repr` the "representation layer." The professor may be using a different framing:
- **His framing:** conv1/2/3 = "representation layers" (they build visual representations of input); fc_repr/fc_out = "strategy/policy layers" (they encode game-specific decisions)
- **Our framing:** conv1/2/3 = feature extraction; fc_repr = "representation layer" (the 512-dim state encoding)

Both usages exist in the literature. This is not a simple rename — if the professor considers fc_repr a "strategy layer," it reframes how we describe the central finding of the whole paper.

**Do not rename anything until this is clarified at the next meeting.**

Regardless of outcome, stop using informal "lower/higher layers" language in the report.

---

### 3. "Predate, make consistent"
Genuinely unclear. Possible interpretations:
- **"Update and make consistent"** — revise the report so terminology is uniform throughout
- **"Predicate and make consistent"** — each section should open with a clear purpose statement before the methodology
- Could be a mishearing of "iterate" or "preface"

**Ask the professor directly at the next meeting. Do not implement anything based on a guess.**

---

### 4. CKA — the problem is deeper than initially described

**⚠️ Initial analysis was too optimistic. This is a more substantive methodological issue.**

The professor's concern has two distinct parts:

**Part A — "not policy specific, random frames":**
Current CKA uses 1,000 frames from random actions. These are states the agents never visit under their actual policies, so the similarity measure is over a distribution that doesn't reflect real agent behaviour.

**Part B — "do during policy training":**
This likely means compute CKA at **multiple checkpoints during training** — a temporal analysis of how cross-game representational similarity evolves as the policy develops. Not just post-hoc at the final checkpoint. This is a separate and additional analysis.

**The harder problem with on-policy frames for cross-game CKA:**
For cross-GAME comparison (Pong agent vs Breakout agent), there is no clean on-policy probe set. The Pong agent's policy states are Pong game states; the Breakout agent's are Breakout states — completely different distributions. If you use Pong agent's states as the shared probe, it is "on-policy" for the Pong agent but entirely off-distribution for the Breakout agent, and vice versa.

The random frames approach was actually a principled choice (neutral, unbiased, avoids state distribution mismatch) — it just wasn't justified that way in the report.

**What likely needs to happen:**
- Either justify the current approach explicitly: *"Random frames provide a controlled probe set that avoids distribution mismatch between agents trained on different games"*
- Or use each agent's own replay buffer states for CKA (policy-specific but then cross-agent comparison is no longer on equal footing)
- AND add temporal CKA at each checkpoint — this is a new analysis regardless

For cross-ALGORITHM CKA (DQN vs DDQN on same game): on-policy frames are feasible here since both agents play the same game and share state distributions. This is where "on-policy" matters most and is cleanest to implement.

**The existing `.npz` files in `results/representations/` contain on-policy rollouts (ε=0.05). These can be used for the algorithm-effect CKA immediately. The game-effect CKA probe set question needs more thought.**

---

### 5. "Train same architecture — 2 training policies, compare representation layers"

We did this — it is the 2×2 baseline. The professor's point is about framing. The report needs to state clearly that the design specifically isolates the effect of training policy and game: one architecture, two independent training regimes, two games. The architecture is held constant so any representational difference is attributable only to the game or the TD update rule.

Add a sentence like: *"Both DQN and DDQN use an identical CNN backbone. Any difference in learned representations is therefore purely a function of the game played or the TD target formulation, not the network architecture."*

No new experiment needed. Report clarification only.

---

### 6. Interleaved — redesign (new experiment, more complex than first described)

**Current (v2):** Step-level alternation every 1,000 steps, 2 separate replay buffers, each sampled independently.

**Professor wants (v3):**
- Switch game **per episode** — finish the current episode before switching, not mid-episode
- **1 joint replay buffer** — all Pong and Breakout transitions stored together
- At start of each episode, select which game to play (balanced — maintain roughly equal episode or step counts)
- "Fixed steps per game rounded to an episode" = the total budget per game is defined in steps but the actual switch happens at the next episode boundary

**Key technical implication of the joint buffer:**
When sampling a batch for learning, the batch will contain a mix of Pong and Breakout transitions. Each transition must be routed through the correct output head based on its `game_id`. This requires per-sample routing within the batch — not trivial. The loss is: for Pong transitions use the Pong head, for Breakout transitions use the Breakout head, sum the losses, backprop through shared backbone.

This is meaningfully different from v2 and is a proper new experiment to implement.

---

### 7. Sequential — dead neuron reinitialisation (new experiment)

**⚠️ This is related to continual backpropagation but is not the same thing.**

What the professor described: after training on Pong (where ~88% of fc_repr neurons are dead), **identify the dead neurons and reinitialise their weights** before starting Breakout training. This is a one-shot intervention at the task boundary.

**Continual backpropagation** (Elsayed & Mahmood 2024, Dohare et al. 2024) is different — it continuously reinitialises low-utility neurons throughout training, not just at task boundaries. Read the paper for background and motivation, but the specific experiment the professor asked for is simpler:

1. Train Pong to completion → measure dead neurons (~88% of fc_repr)
2. Reinitialise the weights of those dead neurons (reset to orthogonal init, same as original)
3. Train on Breakout from this state (same as sequential no-freeze condition otherwise)
4. Measure: does Breakout learning speed up? Does Pong forgetting change?
5. Compare against the standard sequential no-freeze baseline

Papers to read first: Elsayed & Mahmood (2024), Dohare et al. (2024) "Loss of Plasticity in Deep Continual Learning."

---

### 8. Page 3 — explain neural agent

The introduction jumps into DQN without grounding what a "neural agent" is. Need a short paragraph establishing that the agent IS a neural network — pixels go in, Q-values come out, the network weights ARE the policy. This bridges the RL "agent" framing and the neural network framing for readers coming from either side.

---

### 9. Page 16 — extra layer / convergence

"An extra layer which learned the game's rules" — the professor is referring to `fc_repr` specifically: the layer above the convolutions that encodes game-specific strategy. He found interesting the result that this layer, when preserved intact (freeze all condition), allows Pong performance to be retained perfectly.

**"Was this for frozen?"** — He may specifically be asking about the `fixed_sequential.py --freeze all` condition (both conv AND fc_repr frozen), which is the condition where Pong stays at ~9.4. Worth clarifying whether he means train_backbone.py (the 5M Breakout backbone experiment) or fixed_sequential.py freeze_all.

**Convergence:** For the backbone experiment (train_backbone.py, 5M steps), scratch DQN on Breakout reached 3.86 which is almost certainly not convergence — Breakout typically needs 10M+ steps with a 100k buffer. This needs an explicit limitation statement in the report. Decision needed: add a caveat, or re-run to 10M steps.

---

### 10. Superposition — read first, then assess

Read: **Elhage et al. (2022) — "Toy Models of Superposition"**

Connection to our results: after sequential training, 88% of fc_repr neurons are dead. The remaining 12% (~62 neurons out of 512) may be encoding hundreds of features through superposition — representing more information than their count suggests. This connects the dead neuron / capacity collapse results to a theoretical framework.

No experiment needed yet. Read the paper, then decide if a concrete analysis is warranted. Most likely this belongs in the Discussion section as a theoretical interpretation of the capacity collapse finding.

---

## Questions to Ask at Next Meeting

1. **Terminology:** Do you consider `fc_repr` (the 512-dim penultimate layer) a "representation layer" or a "strategy/policy layer"? We currently call it a "representation layer" throughout the report.
2. **"Predate, make consistent"** — what did you mean by this?
3. **Page 16:** Were you referring to the backbone experiment (train_backbone.py) or the freeze-all sequential condition (fixed_sequential.py)? Should we re-run to 10M steps or just add a limitation caveat?
4. **CKA frames:** For cross-game CKA, both networks must see the same inputs. On-policy states from Pong agent are off-distribution for Breakout agent. Is the justification for random frames acceptable, or do you have a preferred probe set in mind?

---

## Action Items

| # | Item | Type | Priority | Notes |
|---|---|---|---|---|
| 1 | Temporal CKA at each checkpoint during training | New analysis | High | Separate from the probe-set question |
| 2 | On-policy CKA for algorithm-effect comparison (DQN vs DDQN same game) | Code change | High | Feasible — same game, same state distribution |
| 3 | Justify or revise cross-game CKA probe set | Report + possible code | High | Discuss with professor first |
| 4 | Interleaved v3 — joint buffer, episode-level, per-sample routing | New experiment | High | More complex than v2 |
| 5 | Dead neuron reinit before Breakout (sequential condition) | New experiment | High | Read continual backprop papers first |
| 6 | Report restructure — purpose/conduct/results per experiment | Report rewrite | High | Major restructure |
| 7 | Clarify terminology (fc_repr = representation or strategy layer?) | Clarification | High | Must resolve before any report edits |
| 8 | Explain neural agent (page 3) | Report edit | Medium | Short paragraph |
| 9 | Clarify page 16 — which experiment, convergence decision | Clarification + report | Medium | |
| 10 | Standardise layer terminology once clarified | Report edit | Medium | After item 7 resolved |
| 11 | Read superposition paper (Elhage et al. 2022) | Reading | Medium | Discussion section angle |
| 12 | Clarify "predate/predicate" | Ask professor | Low | Do not implement until clear |

---

## What We're Doing Now (this sprint)

Two new experiments. Results are isolated to dedicated subdirectories so nothing in `results/checkpoints/`, `results/logs/`, or `results/plots/` is overwritten.

### 1. Interleaved v3 — `experiments/interleaved_v3.py`

**What:** Episode-level alternation with a joint replay buffer (professor's specification).  
**How:**
- One `JointReplayBuffer` stores transitions from both games, each tagged `game_id ∈ {0=pong, 1=breakout}`
- Game switch happens only at episode boundaries (round-robin: switch to whichever game has fewer steps)
- When sampling a batch for learning, Pong transitions are routed through `fc_out_pong`, Breakout transitions through `fc_out_breakout`; gradients from both games flow back through the shared backbone in a single backward pass
- Architecture: `AtariCNNTwoHead` (shared conv1-3 + fc_repr, two separate fc_out heads)
- 1M steps per game = 2M total

**Results go to:** `results/interleaved_v3/checkpoints/` and `results/interleaved_v3/logs/`  
**Key metric file:** `results/interleaved_v3/logs/dqn_interleaved_v3_seed42_scalemedium_metrics.csv`

---

### 2. Sequential Reinit — `experiments/sequential_reinit.py`

**What:** Standard sequential transfer (Pong → Breakout) with dead neuron reinitialisation at the task boundary.  
**How:**
1. Load Pong 2M checkpoint
2. Run 1000 forward passes on Pong frames, measure post-ReLU fc_repr activations
3. Any neuron dead in >95% of frames gets orthogonal reinit (`nn.init.orthogonal_`, gain=√2)
4. Copy modified backbone (conv + fc_repr) into fresh 4-action Breakout agent
5. Recreate optimizer (clear stale Adam momentum from reinited weights)
6. Train Breakout 2M steps
7. Every 200k steps: chimera Pong eval, dead neuron count, CKA drift

`fc_repr` in `AtariCNN` is `nn.Sequential(nn.Linear(3136, 512), nn.ReLU())` — reinit targets `model.fc_repr[0]` (the Linear at index 0).

**Results go to:** `results/sequential_reinit/checkpoints/` and `results/sequential_reinit/logs/`  
**Key metric file:** `results/sequential_reinit/logs/dqn_sequential_reinit_seed42_scalemedium_forgetting.csv`

---

### Pending (after next professor meeting)

3. Temporal CKA at each checkpoint during training  
4. On-policy CKA for DQN vs DDQN comparison on same game  
5. Justify or revise cross-game CKA probe set (discuss with professor first)  
6. Full report restructure — purpose/conduct/results per experiment section  
7. Clarify terminology (fc_repr = representation or strategy layer?) — **must resolve before any report edits**  
8. Page 3: add neural agent explanation paragraph  
9. Page 16: convergence caveat or re-run to 10M steps  
10. Standardise layer terminology once #7 is resolved  
11. Read: Elhage et al. (2022) Toy Models of Superposition  
12. Read: Elsayed & Mahmood (2024) + Dohare et al. (2024) continual backprop papers  
13. Clarify "predate/predicate" with professor

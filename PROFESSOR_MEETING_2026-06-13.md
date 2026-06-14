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

## Point-by-Point Analysis

### 1. Report Structure
Each experiment section needs to be structured as:
**Purpose → Setup/How → Results → Interpretation**

Currently Results is one giant section separate from Experimental Design. The professor wants them co-located per experiment. This is a significant restructure of `main-v4.tex`.

---

### 2. Terminology — standardise throughout report
- Conv1 / Conv2 / Conv3 → **"perceptual layers"** or **"feature extraction layers"**
- fc_repr → **"representation layer"**
- fc_out → **"policy layer"** or **"decision layer"**

Stop using informal "lower/higher layers" language.

---

### 3. "Predate, make consistent"
Best interpretation: **"predicate"** — each section/experiment should open with a clear statement of its *purpose* before describing the methodology. Also likely means making notation and terminology consistent throughout. **Clarify at next meeting.**

---

### 4. CKA — not policy-specific (most substantive technical concern)

**Professor's concern:** Current CKA uses 1,000 frames collected with **random actions** — neutral but not representative of what agents actually attend to during play.

**What he wants:** Frames collected **on-policy** (using each agent's actual learned policy, ε≈0.05), so CKA reflects similarity in representations agents actually use when making decisions.

**Good news:** `extract_representations.py` already collects on-policy frames (ε=0.05) and all `.npz` files exist in `results/representations/`. We just need to re-run `layer_similarity.py` using those files instead of re-collecting with random actions. Small code change, data already exists.

---

### 5. "Train same architecture — 2 training policies, compare representation layers"

**We already did this** — it is the entire 2×2 baseline (DQN/Pong vs DQN/Breakout, DDQN/Pong vs DDQN/Breakout). The professor just wants the report to say this more explicitly:

> *"Both algorithms share an identical CNN — any representational difference is purely a function of the game played or the TD update rule, not the architecture."*

No new experiment needed. Just a report clarification.

---

### 6. Interleaved — redesign (new experiment)

**Current (v2):** Step-level alternation every 1,000 steps, 2 separate replay buffers.

**Professor wants (v3):**
- Switch game **per episode** (not per fixed step count)
- **1 shared replay buffer** — all experiences from both games mixed together
- At the start of each episode, select which game to play (balanced, e.g. round-robin)
- Fixed total steps per game, rounded to episode boundary

The joint buffer means the agent can sample Breakout transitions while executing a Pong episode — tests whether shared experience helps or causes interference.

**This is a new experiment to implement.**

---

### 7. Sequential — reinitialise dead neurons (new experiment)

After training on Pong (~88% of fc_repr neurons are dead), **before starting Breakout training**: reinitialise the dead neurons to restore plasticity.

This is **continual backpropagation** — read: **Elsayed & Mahmood (2024), "Maintaining Plasticity in Continual Learning via Regenerative Regularization"** (or the related "Loss of Plasticity in Deep Continual Learning" paper).

We already track exactly which neurons are dead. The question: does reinitialising them before Breakout training reduce forgetting and/or improve Breakout performance?

**This is a new experiment to implement.**

---

### 8. Page 3 — explain neural agent

Introduction jumps into DQN without grounding what a "neural agent" is. Need a short paragraph explaining that the agent is a neural network mapping pixel inputs to Q-values, and that "neural agent" is used throughout to mean this.

---

### 9. Page 16 — extra layer / convergence question

This refers to the **backbone/frozen conv experiment** (Experiment 4 — train_backbone.py). Professor found the idea interesting that fc_repr "learned the rules of the game."

**Was it trained till convergence?** Probably not — 5M steps on Breakout with scratch DQN got 3.86 final reward which was still climbing. Need to either:
- Acknowledge explicitly in the report as a limitation
- Or run longer (10M steps) — decision needed

---

### 10. Superposition — read and assess

Read: **Elhage et al. (2022) — "Toy Models of Superposition"**

The idea: networks represent more features than they have neurons by superimposing them. Connects to our dead neuron results — 88% dead neurons post-sequential means the remaining 12% may be doing enormous work via superposition. This could be a novel angle in the Discussion section connecting capacity collapse to superposition theory.

No experiment needed immediately. Read first, then decide if a concrete analysis is warranted.

---

## Action Items

| # | Item | Type | Priority |
|---|---|---|---|
| 1 | CKA with on-policy frames | Small code change — data already exists | High |
| 2 | Interleaved v3 — joint buffer, episode-level | New experiment | High |
| 3 | Continual backpropagation — reinit dead neurons before Breakout | New experiment | High |
| 4 | Report restructure — purpose/conduct/results per experiment | Report rewrite | High |
| 5 | Standardise layer terminology throughout report | Report edit | Medium |
| 6 | Explain neural agent (page 3) | Report edit | Medium |
| 7 | Clarify page 16 convergence — limitation or re-run | Report edit / decision | Medium |
| 8 | Read superposition paper (Elhage et al. 2022) | Reading | Medium |
| 9 | Clarify "predate/predicate" with professor | Clarification at next meeting | Low |

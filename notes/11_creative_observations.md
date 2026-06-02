# 💡 Creative Observations and Open Questions

> Unexpected findings, interesting angles, things worth raising in a meeting.

---

## Observation 1 — Forgetting is shockingly fast

In the smoke test (15k Breakout steps), Pong reward dropped from ~7.5 to -8.85. That is a catastrophic collapse in under 15,000 steps when training took 2,000,000 steps.

**The interesting question:** Is forgetting fast because the representations reorganise immediately, or because the output layer adapts first and then drags the representations with it? CKA drift at early checkpoints should tell us.

---

## Observation 2 — Frozen conv made things worse, not just "no better"

Everyone expects frozen conv to be somewhere between scratch and full fine-tune. Instead it was *below* scratch. This is a counterintuitive result.

**Why this is interesting:** It suggests the conv-fc interface is so tightly coupled that even geometrically similar but non-identical conv activations are actively harmful for the fc layers. The fc_repr was not just trained to process "conv-style features" — it was trained to process *this specific conv stack's* specific activation distribution.

---

## Observation 3 — Performance and representation quality are decoupled

DQN and DDQN score essentially the same. But DDQN's internal representations are clearly better by every measure. You would never know this from the leaderboard.

**Why this matters beyond RL:** This is a general point about evaluation. In any AI system, task performance is an insufficient proxy for understanding quality. An agent that scores well through a noisy approximation is less safe, less generalisable, and less interpretable than one that scores the same through a clean internal model.

---

## Observation 4 — The asymmetry question (not yet tested)

We only ran Pong→Breakout sequential training. What about Breakout→Pong?

**Hypothesis:** Breakout→Pong might show *less* forgetting of Breakout than Pong→Breakout shows of Pong. Why? Breakout has a more complex state space — more neurons were actively specialised for Breakout. When you then train on Pong (simpler), the gradient signal may not be strong enough to overwrite all the Breakout-specific encoding. Alternatively, Pong's simpler strategy might only need a small corner of the representation space, leaving most Breakout encoding intact.

This asymmetry, if it exists, would be a genuinely novel finding.

---

## Observation 5 — CKA drift as an early warning system

We are tracking CKA drift during Breakout training. If CKA drops before Pong reward drops, this means representation drift is a **leading indicator** of forgetting — the network starts "thinking differently" before its performance actually collapses.

This would be a practically useful finding: you could monitor CKA in real time and detect forgetting before it manifests in performance metrics.

---

## Observation 6 — Dead neurons link three things

Dead neurons connect:
1. DQN's overestimation bias (causes more dead neurons)
2. Representation quality (more dead neurons = less structured representations)
3. Catastrophic forgetting (neurons die as the old task's encoding is overwritten)

This means dead neurons are not just a side effect — they are a **mechanistic link** in the chain from "noisy training signal" to "forgetting." This is the kind of causal story that makes a strong dissertation chapter.

---

## Observation 7 — The warm start paradox

Full fine-tune from Pong achieves the best Breakout peak (4.47 smoothed). But it almost certainly shows the most Pong forgetting. 

**The paradox:** The benefit of pre-training (better Breakout performance) may come precisely because the network reorganises *away* from Pong representations. Better cross-game transfer = more forgetting. This is a fundamental tradeoff in continual learning.

---

## Open Questions

- Does CKA drift precede or coincide with performance collapse?
- Is forgetting asymmetric between Pong→Breakout and Breakout→Pong?
- At what layer does forgetting primarily occur — conv or fc_repr?
- Can a single network represent both games simultaneously without forgetting? (interleaved training)
- Would continual backpropagation (Elsayed & Mahmood, 2024) prevent the dead neuron increase?

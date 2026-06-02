# 🧠 Project Notes — DQN/DDQN Atari Representation Study

> [!tip] How to use these notes
> Each file has three zoom levels — pub explanation, study group, professor level.
> Read at whatever depth you need. Updated after every experiment.

---

## 📁 Files

- [[01_big_picture]] — What this project is and why it matters
- [[02_dqn_ddqn]] — What DQN and DDQN are
- [[03_representations]] — What the 512-dim layer actually is
- [[04_tsne]] — What t-SNE is and how to read the plots
- [[05_cka]] — What CKA is and what it tells us
- [[06_dead_neurons]] — Dead neurons and capacity loss
- [[07_catastrophic_forgetting]] — What happens when an agent switches games
- [[08_experiments]] — Every experiment: what, why, what we found
- [[09_key_findings]] — All results summarised in one place
- [[10_professor_questions]] — How to answer supervisor questions confidently
- [[11_creative_observations]] — Unexpected findings and open questions
- [[12_status]] — What is running right now and what is next

---

## ⚡ One-Line Summary of the Whole Project

> We trained DQN and DDQN on Pong and Breakout, dissected what they learned internally, tested whether those representations transfer across games, and measured what happens to the network when it is forced to switch games mid-training.

---

## 🗂️ Project Config at a Glance

| Setting | Value |
|---|---|
| Games | Pong (2M steps), Breakout (5M steps) |
| Algorithms | DQN, DDQN |
| Network | CNN → 512-dim fc_repr → Q-values |
| Replay buffer | 100k transitions |
| Batch size | 32 |
| Learning rate | 1e-4 (Adam) |
| Epsilon | 1.0 → 0.01 over 100k steps |
| Target update | Hard copy every 1000 steps |
| Seed | 42 |
| Checkpoint every | 500k steps |
| Hardware | NVIDIA L4 GPU (GCP) |

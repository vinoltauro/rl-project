"""
Mix-and-Match Layer Evaluation (Zero-Shot)
==========================================
Tests whether convolutional layers trained on one Atari game produce
useful features when plugged directly into an agent trained on another
game — with NO additional training.

This is a direct empirical test of the claim that early conv layers
learn general visual features (motion, edges, ball tracking) that
transfer across structurally similar games.

The key insight: conv and fc_repr layers are independent of the action
space. Only fc_out depends on n_actions. So we can freely swap conv
layers between games while keeping the upper layers game-matched.

Conditions evaluated on Breakout:
  1. Native DQN/Breakout             — upper bound
  2. DQN/Pong conv + DQN/Breakout upper  — KEY TEST (Pong→Breakout transfer)
  3. DDQN/Pong conv + DQN/Breakout upper — cross-algo chimera
  4. Random conv + DQN/Breakout upper    — ablation (is Pong conv better than random?)
  5. Random network                  — true lower bound

Conditions evaluated on Pong:
  6. Native DQN/Pong                 — upper bound
  7. DQN/Breakout conv + DQN/Pong upper  — reverse KEY TEST (Breakout→Pong)
  8. DDQN/Breakout conv + DQN/Pong upper — reverse cross-algo chimera
  9. Random conv + DQN/Pong upper    — ablation
  10. Random network                 — true lower bound

Usage:
    python experiments/mix_and_match_eval.py
    python experiments/mix_and_match_eval.py --n_episodes 20 --output_dir results/plots
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn import AtariCNN
from envs.wrappers import make_atari_env


# ── Checkpoint registry ───────────────────────────────────────────────────────
CHECKPOINT_DIR = "results/checkpoints"

CHECKPOINTS = {
    "dqn_pong":      "dqn_pong_seed42_scalemedium_lr0001_buf100k_step02000000.pt",
    "dqn_breakout":  "dqn_breakout_seed42_scalemedium_lr0001_buf100k_step05000000.pt",
    "ddqn_pong":     "ddqn_pong_seed42_scalemedium_lr0001_buf100k_step02000000.pt",
    "ddqn_breakout": "ddqn_breakout_seed42_scalemedium_lr0001_buf100k_step05000000.pt",
}

N_ACTIONS = {
    "ALE/Pong-v5":     6,
    "ALE/Breakout-v5": 4,
}


# ── Experimental conditions ───────────────────────────────────────────────────
CONDITIONS = [
    # ── Breakout evaluation ───────────────────────────────────────────────────
    {
        "label":       "Native\nDQN/Breakout",
        "conv_from":   "dqn_breakout",
        "upper_from":  "dqn_breakout",
        "eval_game":   "ALE/Breakout-v5",
        "color":       "#4CAF50",
        "group":       "breakout",
        "description": "Upper bound",
    },
    {
        "label":       "DQN/Pong conv\n+ Breakout upper",
        "conv_from":   "dqn_pong",
        "upper_from":  "dqn_breakout",
        "eval_game":   "ALE/Breakout-v5",
        "color":       "#2196F3",
        "group":       "breakout",
        "description": "KEY TEST: Pong conv → Breakout",
    },
    {
        "label":       "DDQN/Pong conv\n+ Breakout upper",
        "conv_from":   "ddqn_pong",
        "upper_from":  "dqn_breakout",
        "eval_game":   "ALE/Breakout-v5",
        "color":       "#9C27B0",
        "group":       "breakout",
        "description": "Cross-algo chimera",
    },
    {
        "label":       "Random conv\n+ Breakout upper",
        "conv_from":   None,
        "upper_from":  "dqn_breakout",
        "eval_game":   "ALE/Breakout-v5",
        "color":       "#FF9800",
        "group":       "breakout",
        "description": "Ablation: random conv baseline",
    },
    {
        "label":       "Random\nnetwork",
        "conv_from":   None,
        "upper_from":  None,
        "eval_game":   "ALE/Breakout-v5",
        "color":       "#9E9E9E",
        "group":       "breakout",
        "description": "True lower bound",
    },
    # ── Pong evaluation ───────────────────────────────────────────────────────
    {
        "label":       "Native\nDQN/Pong",
        "conv_from":   "dqn_pong",
        "upper_from":  "dqn_pong",
        "eval_game":   "ALE/Pong-v5",
        "color":       "#4CAF50",
        "group":       "pong",
        "description": "Upper bound",
    },
    {
        "label":       "DQN/Breakout conv\n+ Pong upper",
        "conv_from":   "dqn_breakout",
        "upper_from":  "dqn_pong",
        "eval_game":   "ALE/Pong-v5",
        "color":       "#FF5722",
        "group":       "pong",
        "description": "KEY TEST: Breakout conv → Pong",
    },
    {
        "label":       "DDQN/Breakout conv\n+ Pong upper",
        "conv_from":   "ddqn_breakout",
        "upper_from":  "dqn_pong",
        "eval_game":   "ALE/Pong-v5",
        "color":       "#9C27B0",
        "group":       "pong",
        "description": "Cross-algo chimera",
    },
    {
        "label":       "Random conv\n+ Pong upper",
        "conv_from":   None,
        "upper_from":  "dqn_pong",
        "eval_game":   "ALE/Pong-v5",
        "color":       "#FF9800",
        "group":       "pong",
        "description": "Ablation: random conv baseline",
    },
    {
        "label":       "Random\nnetwork",
        "conv_from":   None,
        "upper_from":  None,
        "eval_game":   "ALE/Pong-v5",
        "color":       "#9E9E9E",
        "group":       "pong",
        "description": "True lower bound",
    },
]


# ── Core functions ────────────────────────────────────────────────────────────
def load_state(key: str, device: torch.device) -> dict:
    path = os.path.join(CHECKPOINT_DIR, CHECKPOINTS[key])
    ckpt = torch.load(path, map_location=device, weights_only=True)
    return ckpt["model_state"]


def build_chimera(
    conv_from:  str | None,
    upper_from: str | None,
    n_actions:  int,
    device:     torch.device,
) -> AtariCNN:
    """
    Construct a chimera AtariCNN:
      - conv layers    from conv_from  checkpoint (or random init if None)
      - fc_repr+fc_out from upper_from checkpoint (or random init if None)

    n_actions must match the target game (determines fc_out shape).
    """
    model = AtariCNN(n_actions=n_actions, net_scale="medium").to(device)
    current = model.state_dict()

    if conv_from is not None:
        src = load_state(conv_from, device)
        for k, v in src.items():
            if k.startswith("conv."):
                current[k] = v

    if upper_from is not None:
        src = load_state(upper_from, device)
        for k, v in src.items():
            if k.startswith("fc_repr."):
                current[k] = v
            elif k.startswith("fc_out."):
                # Only load if fc_out shape matches n_actions
                if k == "fc_out.weight" and v.shape[0] == n_actions:
                    current[k] = v
                elif k == "fc_out.bias" and v.shape[0] == n_actions:
                    current[k] = v

    model.load_state_dict(current)
    model.eval()
    return model


def evaluate(
    model:      AtariCNN,
    env_id:     str,
    n_episodes: int,
    device:     torch.device,
    seed:       int = 77,
) -> tuple[float, float, list]:
    """Run n_episodes, return (mean_reward, std_reward, all_rewards)."""
    n_actions = N_ACTIONS[env_id]
    env = make_atari_env(env_id, seed=seed)
    rewards = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            state_t = (
                torch.from_numpy(obs).float().div(255.0)
                .unsqueeze(0).to(device)
            )
            with torch.no_grad():
                q_vals = model(state_t)
            # Clamp action to valid range in case fc_out has more outputs
            action = int(q_vals.argmax(dim=1).item())
            action = min(action, n_actions - 1)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)

    env.close()
    return float(np.mean(rewards)), float(np.std(rewards)), rewards


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_results(results: list[dict], group: str, output_dir: str):
    group_results = [r for r in results if r["group"] == group]
    game_name = "Breakout" if group == "breakout" else "Pong"

    labels   = [r["label"]    for r in group_results]
    means    = [r["mean"]     for r in group_results]
    stds     = [r["std"]      for r in group_results]
    colors   = [r["color"]    for r in group_results]
    descs    = [r["description"] for r in group_results]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                  alpha=0.85, edgecolor="white", linewidth=1.2)

    # Annotate bars with mean value
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.3,
            f"{mean:.1f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean Episode Reward", fontsize=11)
    ax.set_title(
        f"Mix-and-Match Zero-Shot Evaluation — {game_name}\n"
        "Conv layers swapped between agents with no additional training",
        fontsize=12,
    )
    ax.axhline(means[0], color=colors[0], linestyle="--", alpha=0.4,
               linewidth=1.5, label=f"Native {game_name} agent")
    ax.axhline(means[-1], color=colors[-1], linestyle=":", alpha=0.4,
               linewidth=1.5, label="Random baseline")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"mix_and_match_{group}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n[plot] Saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes",  type=int, default=15,
                        help="Episodes per condition (default 15)")
    parser.add_argument("--output_dir",  default="results/plots")
    parser.add_argument("--checkpoint_dir", default="results/checkpoints")
    args = parser.parse_args()

    global CHECKPOINT_DIR
    CHECKPOINT_DIR = args.checkpoint_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    print(f"[eval]   {args.n_episodes} episodes per condition\n")

    results = []

    for cond in CONDITIONS:
        n_actions = N_ACTIONS[cond["eval_game"]]
        print(f"── {cond['description']}: {cond['label'].replace(chr(10), ' ')}")
        print(f"   conv_from={cond['conv_from']} | upper_from={cond['upper_from']} | game={cond['eval_game']}")

        model = build_chimera(
            conv_from  = cond["conv_from"],
            upper_from = cond["upper_from"],
            n_actions  = n_actions,
            device     = device,
        )

        mean, std, ep_rewards = evaluate(
            model      = model,
            env_id     = cond["eval_game"],
            n_episodes = args.n_episodes,
            device     = device,
        )

        print(f"   → mean={mean:.2f} ± {std:.2f}  (episodes: {[round(r,1) for r in ep_rewards]})\n")

        results.append({
            **cond,
            "mean": mean,
            "std":  std,
            "episodes": ep_rewards,
        })

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  {'Condition':<40} {'Game':<12} {'Mean':>7} {'±Std':>7}")
    print("=" * 70)
    for r in results:
        label = r["label"].replace("\n", " ")
        game  = r["eval_game"].split("/")[-1].replace("-v5", "")
        print(f"  {label:<40} {game:<12} {r['mean']:>7.2f} {r['std']:>7.2f}")
    print("=" * 70)

    # ── Save plots ────────────────────────────────────────────────────────────
    plot_results(results, group="breakout", output_dir=args.output_dir)
    plot_results(results, group="pong",     output_dir=args.output_dir)

    print("\n[done] Mix-and-match evaluation complete.")


if __name__ == "__main__":
    main()

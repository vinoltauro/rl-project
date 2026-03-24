"""
Partial Extraction — Run for whatever checkpoints exist.
Skips any run with no checkpoints (e.g. DDQN runs not yet complete).

Usage (on VM, from rl-project/):
    python analysis/extract_partial.py

Saves .npz files to results/representations/.
"""

import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from analysis.extract_representations import collect_representations
import numpy as np

CHECKPOINT_DIR = "results/checkpoints"
OUTPUT_DIR     = "results/representations"

RUNS = [
    {"run_name": "dqn_pong_seed42_scalemedium_lr0001_buf100k",
     "env_id":   "ALE/Pong-v5",
     "n_actions": 6,
     "algorithm": "dqn"},

    {"run_name": "dqn_breakout_seed42_scalemedium_lr0001_buf100k",
     "env_id":   "ALE/Breakout-v5",
     "n_actions": 4,
     "algorithm": "dqn"},

    {"run_name": "ddqn_pong_seed42_scalemedium_lr0001_buf100k",
     "env_id":   "ALE/Pong-v5",
     "n_actions": 6,
     "algorithm": "ddqn"},

    {"run_name": "ddqn_breakout_seed42_scalemedium_lr0001_buf100k",
     "env_id":   "ALE/Breakout-v5",
     "n_actions": 4,
     "algorithm": "ddqn"},
]

N_STEPS = 5000
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Device: {DEVICE}")
print(f"Output: {OUTPUT_DIR}\n")

for run in RUNS:
    run_name = run["run_name"]
    pattern  = os.path.join(CHECKPOINT_DIR, f"{run_name}_step*.pt")
    checkpoints = sorted(glob.glob(pattern))

    if not checkpoints:
        print(f"[SKIP] No checkpoints found for {run_name}\n")
        continue

    print(f"[extract] {run_name} — {len(checkpoints)} checkpoint(s)")

    for ckpt_path in checkpoints:
        ckpt_base = os.path.basename(ckpt_path)
        step_str  = ckpt_base.replace(".pt", "").split("_step")[-1]
        step      = int(step_str)

        save_path = os.path.join(OUTPUT_DIR, f"repr_{run_name}_step{step:08d}.npz")

        if os.path.exists(save_path):
            print(f"  [skip] Already exists: {os.path.basename(save_path)}")
            continue

        print(f"  Processing step {step:,} ...")
        data = collect_representations(
            checkpoint_path = ckpt_path,
            env_id          = run["env_id"],
            n_actions       = run["n_actions"],
            net_scale       = "medium",
            n_steps         = N_STEPS,
            seed            = 99,
            device          = DEVICE,
        )

        np.savez_compressed(
            save_path,
            representations = data["representations"],
            actions         = data["actions"],
            rewards         = data["rewards"],
            cumulative_r    = data["cumulative_r"],
            done_flags      = data["done_flags"],
            step_at_ckpt    = np.array([data["step_at_ckpt"]]),
            run_name        = np.array([run_name]),
            env_id          = np.array([run["env_id"]]),
            algorithm       = np.array([run["algorithm"]]),
        )
        print(f"    Saved → {os.path.basename(save_path)}")

    print()

print("Extraction complete.")
print(f"Files in {OUTPUT_DIR}:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  {f}")

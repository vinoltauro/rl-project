"""
run_all.py — Master Experiment Runner
======================================
Launches all 4 training runs sequentially (or in parallel with --parallel).
After training, kicks off all analysis scripts.

Usage:
    # Run everything (training then analysis):
    python run_all.py

    # Training only:
    python run_all.py --training_only

    # Analysis only (after training is done):
    python run_all.py --analysis_only

    # Smoke test (100k steps per run):
    python run_all.py --smoke_test

    # Specific runs only:
    python run_all.py --runs 1 3   (Run 1 and 3 only)
"""

import os
import sys
import subprocess
import argparse
import time
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


RUN_CONFIGS = {
    1: "experiments/configs/run1_dqn_pong.yaml",
    2: "experiments/configs/run2_dqn_breakout.yaml",
    3: "experiments/configs/run3_ddqn_pong.yaml",
    4: "experiments/configs/run4_ddqn_breakout.yaml",
}

RUN_NAMES = {
    1: "DQN  / Pong",
    2: "DQN  / Breakout",
    3: "DDQN / Pong",
    4: "DDQN / Breakout",
}


def _make_run_name(cfg: dict) -> str:
    """Mirrors train.py:make_run_name — used to locate checkpoints."""
    algo   = cfg["algorithm"]
    game   = cfg["env_id"].split("/")[-1].lower().replace("-v5", "")
    seed   = cfg["seed"]
    scale  = cfg.get("net_scale", "medium")
    lr_str = str(cfg.get("lr", 1e-4)).replace("0.", "").replace("-", "")
    buf    = cfg.get("buffer_capacity", 100_000) // 1000
    return f"{algo}_{game}_seed{seed}_scale{scale}_lr{lr_str}_buf{buf}k"


def _find_latest_checkpoint(checkpoint_dir: str, run_name: str):
    """Return path to latest .pt checkpoint for a run, or None."""
    if not os.path.isdir(checkpoint_dir):
        return None
    files = sorted([
        f for f in os.listdir(checkpoint_dir)
        if f.startswith(run_name) and f.endswith(".pt") and ".tmp" not in f
    ])
    return os.path.join(checkpoint_dir, files[-1]) if files else None


def _is_run_complete(checkpoint_dir: str, run_name: str, total_steps: int) -> bool:
    """True if the final-step checkpoint already exists on disk."""
    final = os.path.join(checkpoint_dir, f"{run_name}_step{total_steps:08d}.pt")
    return os.path.exists(final)


def run_training(run_ids: list, smoke_test: bool = False, force_cpu: bool = False):
    """Launch training runs sequentially, auto-resuming from the latest checkpoint."""
    for run_id in run_ids:
        cfg_path = os.path.join(BASE_DIR, RUN_CONFIGS[run_id])

        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        run_name      = _make_run_name(cfg)
        checkpoint_dir = cfg.get("checkpoint_dir", "results/checkpoints")
        total_steps   = cfg["total_steps"]

        # Skip runs that are fully complete (unless smoke_test mode)
        if not smoke_test and _is_run_complete(checkpoint_dir, run_name, total_steps):
            print(f"\n[SKIP] Run {run_id} ({RUN_NAMES[run_id]}) already complete")
            print(f"       Final checkpoint: {run_name}_step{total_steps:08d}.pt")
            continue

        print(f"\n{'='*60}")
        print(f"  Starting Run {run_id}: {RUN_NAMES[run_id]}")
        print(f"{'='*60}")

        cmd = [sys.executable, os.path.join(BASE_DIR, "train.py"), "--config", cfg_path]
        if smoke_test:
            cmd += ["--steps", "50000"]
        if force_cpu:
            cmd += ["--cpu"]

        # Auto-resume from latest checkpoint if one exists
        if not smoke_test:
            latest = _find_latest_checkpoint(checkpoint_dir, run_name)
            if latest:
                print(f"[auto-resume] Resuming from: {os.path.basename(latest)}")
                cmd += ["--resume", latest]

        start = time.time()
        result = subprocess.run(cmd, cwd=BASE_DIR)
        elapsed = (time.time() - start) / 60

        if result.returncode != 0:
            print(f"\n[ERROR] Run {run_id} failed (exit code {result.returncode})")
            print(f"        Re-run `python run_all.py --runs {run_id}` to auto-resume")
        else:
            print(f"\n[OK] Run {run_id} complete in {elapsed:.1f}min")


def run_analysis():
    """Run the full analysis pipeline after training."""
    print(f"\n{'='*60}")
    print("  Running Analysis Pipeline")
    print(f"{'='*60}\n")

    scripts = [
        # Step 1: Extract representations from all checkpoints
        [sys.executable,
         os.path.join(BASE_DIR, "analysis", "extract_representations.py"),
         "--output", "results/representations"],

        # Step 2: Generate t-SNE plots
        [sys.executable,
         os.path.join(BASE_DIR, "analysis", "tsne_visualisation.py"),
         "--repr_dir",   "results/representations",
         "--output_dir", "results/plots"],

        # Step 3: Training curves + Q-value + dead neurons + cosine similarity
        [sys.executable,
         os.path.join(BASE_DIR, "analysis", "activation_analysis.py"),
         "--log_dir",    "results/logs",
         "--repr_dir",   "results/representations",
         "--output_dir", "results/plots"],
    ]

    for cmd in scripts:
        print(f"\n[analysis] Running: {' '.join(cmd[-1:])}")
        result = subprocess.run(cmd, cwd=BASE_DIR)
        if result.returncode != 0:
            print(f"[WARNING] Script returned non-zero exit: {cmd[1]}")

    print(f"\n✓ Analysis complete. Figures in: results/plots/")


def main():
    parser = argparse.ArgumentParser(
        description="Run all 4 training experiments + full analysis pipeline"
    )
    parser.add_argument("--training_only", action="store_true",
                        help="Only run training, skip analysis")
    parser.add_argument("--analysis_only", action="store_true",
                        help="Only run analysis (training must already be done)")
    parser.add_argument("--smoke_test",    action="store_true",
                        help="Run 50k steps per run — quick sanity check (~5 min each)")
    parser.add_argument("--cpu",           action="store_true",
                        help="Force CPU mode (auto-applies CPU-optimised settings)")
    parser.add_argument("--runs", nargs="+", type=int, default=[1, 2, 3, 4],
                        help="Which runs to execute, e.g. --runs 1 3")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  RL Dissertation — Atari Representation Study")
    print(f"  Runs: {[RUN_NAMES[r] for r in args.runs]}")
    if args.smoke_test:
        print(f"  ⚠  SMOKE TEST MODE (50k steps per run)")
    if args.cpu:
        print(f"  ⚠  CPU MODE (reduced settings applied automatically)")
    print(f"{'='*60}")

    if not args.analysis_only:
        run_training(args.runs, smoke_test=args.smoke_test, force_cpu=args.cpu)

    if not args.training_only:
        run_analysis()

    print(f"\n{'='*60}")
    print(f"  All done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

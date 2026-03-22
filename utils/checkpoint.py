"""
Checkpoint Utilities
====================
Saves and loads full training state so runs can be resumed and
models can be loaded for analysis (t-SNE, saliency maps, etc.).

Each checkpoint stores:
  - model state dict (online network)
  - target network state dict
  - optimizer state dict
  - training step, episode, epsilon
  - config used for this run
"""

import os
import torch
from typing import Dict, Any


def save_checkpoint(
    checkpoint_dir: str,
    run_name:       str,
    step:           int,
    model:          torch.nn.Module,
    target_model:   torch.nn.Module,
    optimizer:      torch.optim.Optimizer,
    episode:        int,
    epsilon:        float,
    config:         Dict[str, Any],
) -> str:
    """
    Save a training checkpoint.

    Returns:
        Path to the saved checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{run_name}_step{step:08d}.pt")
    tmp_path = path + ".tmp"

    torch.save({
        "step":              step,
        "episode":           episode,
        "epsilon":           epsilon,
        "model_state":       model.state_dict(),
        "target_state":      target_model.state_dict(),
        "optimizer_state":   optimizer.state_dict(),
        "config":            config,
    }, tmp_path)
    os.replace(tmp_path, path)  # atomic rename — safe against mid-write interruption

    print(f"  [checkpoint] Saved → {path}")
    return path


def load_checkpoint(
    path:         str,
    model:        torch.nn.Module,
    target_model: torch.nn.Module,
    optimizer:    torch.optim.Optimizer,
    device:       torch.device,
) -> Dict[str, Any]:
    """
    Load a checkpoint into existing model/optimizer objects.

    Returns:
        Dict with step, episode, epsilon, config
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    target_model.load_state_dict(checkpoint["target_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    print(f"  [checkpoint] Loaded ← {path}")
    print(f"               step={checkpoint['step']:,} | "
          f"episode={checkpoint['episode']} | "
          f"epsilon={checkpoint['epsilon']:.4f}")

    return {
        "step":    checkpoint["step"],
        "episode": checkpoint["episode"],
        "epsilon": checkpoint["epsilon"],
        "config":  checkpoint["config"],
    }


def load_model_for_analysis(
    path:    str,
    model:   torch.nn.Module,
    device:  torch.device,
) -> int:
    """
    Load only the model weights (no optimizer) — used for analysis scripts.

    Returns:
        Training step at which this checkpoint was saved.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return checkpoint["step"]


def list_checkpoints(checkpoint_dir: str, run_name: str):
    """List all checkpoint files for a given run, sorted by step."""
    files = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith(run_name) and f.endswith(".pt")
    ]
    files.sort()
    return [os.path.join(checkpoint_dir, f) for f in files]

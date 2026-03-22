"""
Saliency Map Analysis (Grad-CAM)
=================================
Generates saliency maps showing which pixels most influence the agent's
Q-value decisions. Uses vanilla gradient saliency and a simplified
Grad-CAM approach applied to the final conv layer.

Figures produced:
  Fig 11: DQN vs DDQN saliency on Pong — side-by-side
  Fig 12: DQN vs DDQN saliency on Breakout — side-by-side

Interpretation:
  - Bright regions = pixels the agent "attends" to when deciding
  - Expected: ball position, own paddle, scoring zones
  - DDQN may show more focused saliency (less noise from overestimation)
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn import AtariCNN
from utils.checkpoint import load_model_for_analysis


# ─────────────────────────────────────────────────────────────────────────
def compute_vanilla_saliency(
    model: AtariCNN,
    state: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Vanilla gradient saliency: gradient of max Q-value w.r.t. input frame.

    Args:
        model:  Loaded AtariCNN in eval mode
        state:  (4, 84, 84) uint8 numpy array
        device: Torch device

    Returns:
        (84, 84) saliency map (unsigned, normalised to [0,1])
    """
    state_t = (
        torch.from_numpy(state).float().div(255.0)
        .unsqueeze(0).to(device)
    )
    state_t.requires_grad_(True)

    q_vals = model(state_t)
    best_q = q_vals.max()
    model.zero_grad()
    best_q.backward()

    # Gradient magnitude across channels (take max across 4 frames)
    saliency = state_t.grad.data.abs()         # (1, 4, 84, 84)
    saliency = saliency.squeeze(0).max(dim=0).values  # (84, 84)
    saliency = saliency.cpu().numpy()

    # Normalise to [0, 1]
    if saliency.max() > 0:
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

    return saliency


def compute_gradcam(
    model:      AtariCNN,
    state:      np.ndarray,
    device:     torch.device,
) -> np.ndarray:
    """
    Grad-CAM on the final conv layer of AtariCNN.
    Produces a class activation map highlighting spatially important regions.

    Returns:
        (84, 84) heatmap, normalised to [0, 1]
    """
    # Hook to capture activations and gradients from final conv layer
    activations = {}
    gradients   = {}

    def forward_hook(module, input, output):
        activations["conv_out"] = output.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients["conv_out"] = grad_out[0].detach()

    # Register hooks on the last conv layer (index 4 in the conv Sequential)
    last_conv = model.conv[-2]   # Last Conv2d (before final ReLU)
    fwd_handle = last_conv.register_forward_hook(forward_hook)
    bwd_handle = last_conv.register_full_backward_hook(backward_hook)

    state_t = (
        torch.from_numpy(state).float().div(255.0)
        .unsqueeze(0).to(device)
    )

    q_vals = model(state_t)
    best_q = q_vals.max()
    model.zero_grad()
    best_q.backward()

    fwd_handle.remove()
    bwd_handle.remove()

    # Grad-CAM: weight feature maps by their gradient importance
    grads = gradients["conv_out"].squeeze(0)       # (C, H, W)
    acts  = activations["conv_out"].squeeze(0)     # (C, H, W)

    weights = grads.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)
    cam     = (weights * acts).sum(dim=0)            # (H, W)
    cam     = F.relu(cam).cpu().numpy()              # Only positive influence

    # Upsample to input resolution (84, 84)
    if cam.max() > 0:
        cam = cam / cam.max()
    cam_upsampled = np.array(
        plt.cm.jet(
            np.clip(
                np.kron(cam, np.ones((84 // cam.shape[0] + 1, 84 // cam.shape[1] + 1)))
                [:84, :84],
                0, 1
            )
        )
    )[:, :, :3]   # Drop alpha channel

    # Return single-channel heatmap
    return cam.repeat(84 // cam.shape[0] + 1, axis=0).repeat(
        84 // cam.shape[1] + 1, axis=1
    )[:84, :84]


# ─────────────────────────────────────────────────────────────────────────
def collect_interesting_frames(
    model:          AtariCNN,
    env_id:         str,
    n_actions:      int,
    device:         torch.device,
    n_episodes:     int = 3,
    frames_per_ep:  int = 5,
) -> list:
    """
    Run the agent and collect (frame, saliency) pairs at interesting moments
    (non-zero rewards, or every Nth step).

    Returns:
        List of dicts: {raw_frame, stacked_obs, saliency, gradcam, action, reward}
    """
    from envs.wrappers import make_atari_env

    env = make_atari_env(env_id, seed=123)
    collected = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        ep_collected = 0

        while not done and ep_collected < frames_per_ep:
            with torch.no_grad():
                state_t = torch.from_numpy(obs).float().div(255.0).unsqueeze(0).to(device)
                q_vals = model(state_t)
                action = int(q_vals.argmax(dim=1).item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Collect at reward events or regularly
            if abs(reward) > 0 or step % 50 == 0:
                saliency = compute_vanilla_saliency(model, obs, device)
                raw_frame = obs[-1]  # Last frame in stack (most recent)
                collected.append({
                    "raw_frame":   raw_frame,
                    "stacked_obs": obs.copy(),
                    "saliency":    saliency,
                    "action":      action,
                    "reward":      reward,
                    "episode":     ep,
                    "step":        step,
                })
                ep_collected += 1

            obs = next_obs
            step += 1

    env.close()
    return collected


# ─────────────────────────────────────────────────────────────────────────
def fig_saliency_comparison(
    ckpt_dqn:   str,
    ckpt_ddqn:  str,
    env_id:     str,
    n_actions:  int,
    game_name:  str,
    output_dir: str,
    net_scale:  str = "medium",
    device:     torch.device = torch.device("cpu"),
    n_examples: int = 4,
):
    """
    Side-by-side saliency comparison: DQN vs DDQN on the same game.

    Layout per row:
        [Game Frame | DQN Saliency Overlay | DDQN Saliency Overlay]
    """
    # Load both models
    model_dqn  = AtariCNN(n_actions=n_actions, net_scale=net_scale).to(device)
    model_ddqn = AtariCNN(n_actions=n_actions, net_scale=net_scale).to(device)
    model_dqn.eval()
    model_ddqn.eval()

    step_dqn  = load_model_for_analysis(ckpt_dqn,  model_dqn,  device)
    step_ddqn = load_model_for_analysis(ckpt_ddqn, model_ddqn, device)

    # Collect frames using DQN agent (same frames for fair comparison)
    frames = collect_interesting_frames(
        model_dqn, env_id, n_actions, device,
        n_episodes=3, frames_per_ep=n_examples
    )
    frames = frames[:n_examples]

    if not frames:
        print(f"[WARNING] No frames collected for {game_name}")
        return

    fig = plt.figure(figsize=(12, 3 * len(frames)))
    gs  = gridspec.GridSpec(len(frames), 3, hspace=0.4, wspace=0.3)

    for row, frame_data in enumerate(frames):
        obs    = frame_data["stacked_obs"]
        raw_f  = frame_data["raw_frame"]

        sal_dqn  = compute_vanilla_saliency(model_dqn,  obs, device)
        sal_ddqn = compute_vanilla_saliency(model_ddqn, obs, device)

        # Column 0: Raw game frame
        ax0 = fig.add_subplot(gs[row, 0])
        ax0.imshow(raw_f, cmap="gray", vmin=0, vmax=255)
        ax0.set_title(f"Game Frame\n(r={frame_data['reward']:.0f})", fontsize=9)
        ax0.axis("off")

        # Column 1: DQN saliency overlay
        ax1 = fig.add_subplot(gs[row, 1])
        ax1.imshow(raw_f, cmap="gray", vmin=0, vmax=255)
        ax1.imshow(sal_dqn, cmap="hot", alpha=0.6, vmin=0, vmax=1)
        ax1.set_title(f"DQN Saliency\n(step {step_dqn:,})", fontsize=9)
        ax1.axis("off")

        # Column 2: DDQN saliency overlay
        ax2 = fig.add_subplot(gs[row, 2])
        ax2.imshow(raw_f, cmap="gray", vmin=0, vmax=255)
        ax2.imshow(sal_ddqn, cmap="hot", alpha=0.6, vmin=0, vmax=1)
        ax2.set_title(f"DDQN Saliency\n(step {step_ddqn:,})", fontsize=9)
        ax2.axis("off")

    fig.suptitle(
        f"Saliency Maps: DQN vs Double DQN on {game_name}\n"
        "Bright = pixels most influential for Q-value",
        fontsize=12
    )

    out = os.path.join(output_dir, f"saliency_{game_name.lower()}.png")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saliency] Saved → {out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dqn_pong",      required=True)
    parser.add_argument("--ckpt_ddqn_pong",     required=True)
    parser.add_argument("--ckpt_dqn_breakout",  required=True)
    parser.add_argument("--ckpt_ddqn_breakout", required=True)
    parser.add_argument("--output_dir", default="results/plots")
    args = parser.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Generating Pong saliency maps...")
    fig_saliency_comparison(
        args.ckpt_dqn_pong, args.ckpt_ddqn_pong,
        "ALE/Pong-v5", n_actions=6, game_name="Pong",
        output_dir=args.output_dir, device=dev,
    )

    print("Generating Breakout saliency maps...")
    fig_saliency_comparison(
        args.ckpt_dqn_breakout, args.ckpt_ddqn_breakout,
        "ALE/Breakout-v5", n_actions=4, game_name="Breakout",
        output_dir=args.output_dir, device=dev,
    )

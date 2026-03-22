#!/usr/bin/env python3
"""
setup_env.py — One-shot environment setup script
=================================================
Detects your hardware and OS, installs the correct PyTorch build,
installs all dependencies, downloads Atari ROMs, and verifies everything.

Run ONCE after cloning / unzipping the project:

    python setup_env.py

Works on:
  - Windows (CPU)
  - Linux   (CPU or NVIDIA GPU)
  - macOS   (CPU or Apple Silicon MPS)
  - HPC / university clusters (NVIDIA GPU)
  - Google Colab
  - Kaggle
"""

import subprocess
import sys
import os
import platform
import importlib

# ── Colour helpers ────────────────────────────────────────────────────────
def green(s):  return f"\033[92m{s}\033[0m"
def yellow(s): return f"\033[93m{s}\033[0m"
def red(s):    return f"\033[91m{s}\033[0m"
def bold(s):   return f"\033[1m{s}\033[0m"

def run(cmd, check=True, capture=False):
    kwargs = dict(capture_output=capture, text=True) if capture else {}
    result = subprocess.run(cmd, shell=True, check=False, **kwargs)
    if check and result.returncode != 0:
        print(red(f"  [FAILED] {cmd}"))
        sys.exit(1)
    return result


def pip(packages: str, extra_index: str = None):
    index_flag = f"--extra-index-url {extra_index}" if extra_index else ""
    run(f"{sys.executable} -m pip install {packages} {index_flag} -q")


# ─────────────────────────────────────────────────────────────────────────
def detect_hardware():
    """Detect OS, Python version, and GPU availability."""
    info = {
        "os":      platform.system(),          # Windows / Linux / Darwin
        "arch":    platform.machine(),          # x86_64 / arm64
        "python":  sys.version_info,
        "cuda":    False,
        "mps":     False,
        "device":  "cpu",
    }

    # Check NVIDIA GPU via nvidia-smi
    r = run("nvidia-smi", check=False, capture=True)
    if r.returncode == 0:
        info["cuda"] = True
        info["device"] = "cuda"

    # Check Apple Silicon
    if info["os"] == "Darwin" and info["arch"] == "arm64":
        info["mps"] = True
        if not info["cuda"]:
            info["device"] = "mps"

    return info


def print_banner(info):
    print()
    print(bold("=" * 58))
    print(bold("  RL Project — Environment Setup"))
    print(bold("=" * 58))
    print(f"  OS:       {info['os']} ({info['arch']})")
    print(f"  Python:   {info['python'].major}.{info['python'].minor}.{info['python'].micro}")
    print(f"  CUDA GPU: {green('YES') if info['cuda'] else yellow('NO')}")
    print(f"  Apple MPS:{green('YES') if info['mps'] else yellow('NO')}")
    print(f"  Device:   {green(info['device'].upper())}")
    print(bold("=" * 58))
    print()


# ─────────────────────────────────────────────────────────────────────────
def install_torch(info):
    """Install the correct PyTorch build for this machine."""

    print(bold("Step 1/4 — Installing PyTorch"))

    # Check if already installed at a reasonable version
    try:
        import torch
        ver = tuple(int(x) for x in torch.__version__.split(".")[:2])
        if ver >= (2, 0):
            print(f"  {green('✓')} PyTorch {torch.__version__} already installed")
            return
    except ImportError:
        pass

    if info["cuda"]:
        # NVIDIA GPU — install CUDA 11.8 build (compatible with most NVIDIA drivers)
        print(f"  Installing PyTorch with CUDA 11.8 support...")
        pip(
            "torch torchvision torchaudio",
            extra_index="https://download.pytorch.org/whl/cu118"
        )
    elif info["mps"]:
        # Apple Silicon — standard pip build supports MPS
        print(f"  Installing PyTorch for Apple Silicon (MPS)...")
        pip("torch torchvision torchaudio")
    else:
        # CPU only
        print(f"  Installing PyTorch CPU build...")
        pip(
            "torch torchvision torchaudio",
            extra_index="https://download.pytorch.org/whl/cpu"
        )

    print(f"  {green('✓')} PyTorch installed")


def install_dependencies():
    """Install all non-PyTorch dependencies."""
    print(bold("\nStep 2/4 — Installing project dependencies"))

    packages = [
        "gymnasium[atari]>=0.29.0",
        "ale-py>=0.8.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.3.0",
        "tensorboard>=2.13.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "opencv-python>=4.8.0",
        "pandas>=2.0.0",
        "AutoROM[accept-rom-license]",
    ]
    pip(" ".join(f'"{p}"' for p in packages))
    print(f"  {green('✓')} All dependencies installed")


def install_roms():
    """Download Atari ROMs (required by ALE)."""
    print(bold("\nStep 3/4 — Installing Atari ROMs"))
    run(f"{sys.executable} -m AutoROM --accept-license -q")
    print(f"  {green('✓')} Atari ROMs installed")


def verify_installation(info):
    """Run a quick sanity check on the full stack."""
    print(bold("\nStep 4/4 — Verifying installation"))

    checks = [
        ("torch",           lambda: __import__("torch").__version__),
        ("gymnasium",       lambda: __import__("gymnasium").__version__),
        ("ale_py",          lambda: __import__("ale_py").__version__),
        ("numpy",           lambda: __import__("numpy").__version__),
        ("sklearn",         lambda: __import__("sklearn").__version__),
        ("matplotlib",      lambda: __import__("matplotlib").__version__),
        ("cv2",             lambda: __import__("cv2").__version__),
        ("pandas",          lambda: __import__("pandas").__version__),
    ]

    all_ok = True
    for name, version_fn in checks:
        try:
            ver = version_fn()
            print(f"  {green('✓')} {name:<20} {ver}")
        except Exception as e:
            print(f"  {red('✗')} {name:<20} MISSING — {e}")
            all_ok = False

    # Verify GPU access through PyTorch
    print()
    import torch
    if info["cuda"]:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  {green('✓')} CUDA GPU:  {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            print(f"  {yellow('⚠')} CUDA detected by nvidia-smi but not accessible via PyTorch")
            print(f"    → PyTorch will fall back to CPU")
            info["device"] = "cpu"
    elif info["mps"]:
        if torch.backends.mps.is_available():
            print(f"  {green('✓')} Apple MPS: available")
        else:
            print(f"  {yellow('⚠')} MPS not available — using CPU")
            info["device"] = "cpu"
    else:
        print(f"  {yellow('⚠')} No GPU — running on CPU")
        print(f"    Training will work but be slow. See README for time estimates.")

    # Quick Atari env test
    try:
        import gymnasium as gym
        import ale_py
        gym.register_envs(ale_py)
        env = gym.make("ALE/Pong-v5")
        obs, _ = env.reset()
        env.close()
        print(f"  {green('✓')} Atari env:  ALE/Pong-v5 OK (obs shape: {obs.shape})")
    except Exception as e:
        print(f"  {red('✗')} Atari env:  FAILED — {e}")
        all_ok = False

    return all_ok


def write_device_config(info):
    """
    Write a .device file so all scripts know what device to use
    without re-detecting every time.
    """
    device_file = os.path.join(os.path.dirname(__file__), ".device")
    with open(device_file, "w") as f:
        f.write(info["device"])
    print(f"\n  {green('✓')} Device config written: .device = {info['device']}")


def print_next_steps(info):
    print()
    print(bold("=" * 58))
    print(bold("  Setup complete! Next steps:"))
    print(bold("=" * 58))
    print()
    print("  1. Run the smoke test (verifies training works, ~5 min):")
    print(yellow("       python train.py --config experiments/configs/run1_dqn_pong.yaml --steps 50000"))
    print()
    print("  2. Run all 4 full training experiments:")
    print(yellow("       python run_all.py"))
    print()

    if info["device"] == "cpu":
        print(f"  {yellow('⚠  CPU-only mode detected')}")
        print("     Training will be slow. Estimated times:")
        print("       Pong  (2M steps): ~6–8 hours per run")
        print("       Breakout (5M steps): ~15–20 hours per run")
        print()
        print("     Options to speed up:")
        print("       → Use Google Colab (free T4 GPU): colab.research.google.com")
        print("       → Use Kaggle Notebooks (free P100): kaggle.com")
        print("       → Ask your college if they have GPU cluster access (SLURM)")
        print()
        print("     To run a quick test before committing to full training:")
        print(yellow("       python run_all.py --smoke_test"))
    elif info["device"] == "mps":
        print(f"  {green('Apple Silicon GPU (MPS) detected')}")
        print("     Estimated times (M1/M2/M3):")
        print("       Pong  (2M steps): ~2–4 hours per run")
        print("       Breakout (5M steps): ~6–10 hours per run")
    else:
        print(f"  {green('NVIDIA GPU detected — full speed ahead!')}")
        print("     Estimated times (T4):")
        print("       Pong  (2M steps): ~1.5 hours per run")
        print("       Breakout (5M steps): ~4 hours per run")

    print()
    print("  Full documentation: README.md")
    print(bold("=" * 58))
    print()


# ─────────────────────────────────────────────────────────────────────────
def main():
    info = detect_hardware()
    print_banner(info)

    install_torch(info)
    install_dependencies()
    install_roms()
    ok = verify_installation(info)
    write_device_config(info)
    print_next_steps(info)

    if not ok:
        print(red("  Some checks failed — review errors above before training."))
        sys.exit(1)


if __name__ == "__main__":
    main()

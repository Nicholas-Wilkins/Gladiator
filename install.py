#!/usr/bin/env python3
"""
Cross-platform installer for Gladiator.
Detects your hardware and installs the correct PyTorch build automatically.
"""

import os
import platform
import re
import subprocess
import sys


def pip_install(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args])


def run_silent(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def detect_cuda_version():
    """Returns (major, minor) CUDA version tuple or None."""
    for result in [run_silent("nvidia-smi"), run_silent("nvcc --version")]:
        if result.returncode == 0:
            match = re.search(r"CUDA[^\d]*(\d+)\.(\d+)", result.stdout)
            if match:
                return int(match.group(1)), int(match.group(2))
    return None


def cuda_wheel(major, minor):
    """Maps a CUDA version to the closest supported PyTorch wheel suffix, or None if too old."""
    supported = [(12, 8, "cu128"), (12, 6, "cu126"), (12, 4, "cu124"), (12, 1, "cu121"), (11, 8, "cu118")]
    for smaj, smin, tag in supported:
        if (major, minor) >= (smaj, smin):
            return tag
    return None


def is_wsl():
    if platform.system() != "Linux":
        return False
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def has_rocm():
    if platform.system() != "Linux":
        return False
    return run_silent("rocm-smi --version").returncode == 0 or os.path.isdir("/opt/rocm")


def main():
    print("=== Gladiator Installer ===\n")

    print("Installing base requirements...")
    pip_install("-r", "requirements.txt")
    print()

    system = platform.system()

    # NVIDIA CUDA (Windows, Linux, WSL2)
    cuda = detect_cuda_version()
    if cuda:
        major, minor = cuda
        tag = cuda_wheel(major, minor)
        if tag:
            print(f"Detected NVIDIA GPU — CUDA {major}.{minor} → installing torch ({tag})...")
            pip_install("torch", "--index-url", f"https://download.pytorch.org/whl/{tag}")
            print("\nDone. Gladiator will use your NVIDIA GPU via CUDA.")
            return
        else:
            print(f"Warning: CUDA {major}.{minor} is older than any PyTorch wheel supports. Falling back to CPU.")

    # AMD ROCm (Linux native only — not WSL2)
    if has_rocm() and not is_wsl():
        print("Detected AMD GPU with ROCm → installing torch (ROCm 6.2)...")
        pip_install("torch", "--index-url", "https://download.pytorch.org/whl/rocm6.2")
        print("\nDone. Gladiator will use your AMD GPU via ROCm.")
        return

    # DirectML (Windows native only — not WSL2)
    if system == "Windows":
        print("No CUDA/ROCm detected — installing torch + DirectML (AMD/Intel GPU support on Windows)...")
        pip_install("torch", "torch-directml")
        print("\nDone. Gladiator will use DirectML.")
        return

    # CPU fallback
    print("No GPU detected — installing CPU-only torch...")
    pip_install("torch")
    print("\nDone. Gladiator will run on CPU.")


if __name__ == "__main__":
    main()

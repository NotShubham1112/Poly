"""Kaggle environment setup — detect GPU, install correct PyTorch, configure perf settings."""
import subprocess
import sys
import os


def detect_gpu() -> str:
    """Detect GPU type. Returns 'P100', 'T4', or 'unknown'."""
    try:
        output = subprocess.check_output(
            "nvidia-smi --query-gpu=name --format=csv,noheader",
            shell=True, timeout=10
        ).decode().strip()
        if "P100" in output:
            return "P100"
        elif "T4" in output:
            return "T4"
        return output[:50]
    except Exception:
        return "unknown"


def install_torch(gpu_type: str) -> None:
    """Install PyTorch matching the detected GPU."""
    if gpu_type == "P100":
        cmd = [
            sys.executable, "-m", "pip", "install",
            "torch==2.5.1+cu121", "torchvision==0.20.1+cu121",
            "--index-url", "https://download.pytorch.org/whl/cu121",
            "--no-deps", "--force-reinstall", "-q"
        ]
    elif gpu_type == "T4":
        cmd = [
            sys.executable, "-m", "pip", "install",
            "torch==2.6.0+cu124", "torchvision==0.21.0+cu124",
            "--index-url", "https://download.pytorch.org/whl/cu124",
            "--no-deps", "--force-reinstall", "-q"
        ]
    else:
        print(f"Unknown GPU '{gpu_type}' — using default PyTorch")
        return

    print(f"Installing PyTorch for {gpu_type}...")
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Done.")


def install_pyg() -> None:
    """Install PyTorch Geometric."""
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "torch-geometric", "-q"
    ])


def configure_cudnn() -> None:
    """Configure PyTorch for optimal GPU performance."""
    import torch
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)
    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")


def main():
    gpu_type = detect_gpu()
    print(f"Detected GPU: {gpu_type}")
    install_torch(gpu_type)
    import torch
    install_pyg()
    configure_cudnn()


if __name__ == "__main__":
    main()

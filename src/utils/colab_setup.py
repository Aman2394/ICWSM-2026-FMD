"""
Colab setup utilities: Drive mounting, GPU checks, dependency installs.
Used as the standard header cell in all GPU notebooks.
"""
import subprocess
import sys
import os


def setup_colab(project_dir: str = "/content/drive/MyDrive/fmd") -> str:
    """Mount Drive, verify GPU, install deps. Returns PROJECT_DIR."""
    # Mount Drive
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except ImportError:
        print("⚠️  Not running in Colab — skipping Drive mount.")

    os.makedirs(project_dir, exist_ok=True)
    os.chdir(project_dir)

    # Verify GPU
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("⚠️  No GPU detected. Runtime → Change runtime type → T4 GPU")
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Install dependencies
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "transformers>=4.40", "sentence-transformers", "lightgbm",
         "datasets", "accelerate", "bitsandbytes", "scikit-learn",
         "pandas", "numpy", "tqdm"],
        check=True,
    )
    print("✅ Dependencies installed.")
    return project_dir


def setup_colab_cpu(project_dir: str = "/content/drive/MyDrive/fmd") -> str:
    """Mount Drive and install deps (no GPU check). For CPU-only notebooks."""
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except ImportError:
        print("⚠️  Not running in Colab — skipping Drive mount.")

    os.makedirs(project_dir, exist_ok=True)
    os.chdir(project_dir)

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "transformers>=4.40", "sentence-transformers", "lightgbm",
         "datasets", "scikit-learn", "pandas", "numpy", "tqdm"],
        check=True,
    )
    print("✅ Dependencies installed.")
    return project_dir

[project]
name = "echopath"
version = "0.1.0"
description = "Real-Time Indoor Assistive Navigation System for Visually Impaired Users"
readme = "README.md"
requires-python = ">=3.11"  # DA V2 transformers pipeline has issues on 3.13 — use 3.11 or 3.12

dependencies = [
    # ── existing ──────────────────────────────────────────────────────────────
    "matplotlib",
    "numpy",
    "opencv-python",
    "pandas",
    "pillow",
    "pyttsx3",
    "pyyaml",
    "scipy",
    "seaborn",
    "timm",
    "torch",
    "torchaudio",
    "torchvision",
    "ultralytics",

    # ── NEW: required for Depth Anything V2 via HuggingFace ───────────────────
    "transformers>=4.40.0",     # HuggingFace model hub + depth-estimation pipeline
    "accelerate>=0.30.0",       # enables GPU dispatch inside HF pipeline (required)
    "huggingface-hub>=0.23.0",  # model download/caching from HF Hub

    # ── NEW: notebook support ─────────────────────────────────────────────────
    "jupyter",                  # to run .ipynb files locally
    "ipykernel",                # Python kernel for Jupyter
]

[[tool.uv.index]]
name = "pypi"
url = "https://pypi.org/simple"
default = true

[[tool.uv.index]]
name = "pytorch-cu118"
url = "https://download.pytorch.org/whl/cu118"

[tool.uv.sources]
torch      = { index = "pytorch-cu121" }
torchvision = { index = "pytorch-cu121" }
torchaudio = { index = "pytorch-cu121" }




"""ROCm/GPU setup. Import this module before torch is imported anywhere.

The environment variables only take effect if they are set before torch
initialises HIP, which is why they live here and not next to the models.
"""

import os
import warnings

# Enables AOTriton flash / memory-efficient attention for SDPA on RDNA3
# (gfx1100 = RX 7900 XTX). Without it PyTorch uses the math fallback on these
# cards, which was ~7x slower for batched Whisper decoding in testing.
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
# MIOpen's default exhaustive kernel search makes the first pyannote run take
# minutes; FAST picks a kernel from heuristics/the cache instead.
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
# Reduces fragmentation when batch shapes change between files.
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# pyannote warns at import time when torchcodec can't load; we never use it.
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")

import torch  # noqa: E402


def pick_device(requested: str = "auto") -> torch.device:
    """ROCm builds of torch expose the GPU through the torch.cuda API."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def describe(device: torch.device) -> str:
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        backend = f"ROCm/HIP {torch.version.hip}" if torch.version.hip else f"CUDA {torch.version.cuda}"
        return f"{name} ({backend})"
    return "CPU"

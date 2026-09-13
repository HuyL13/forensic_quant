from __future__ import annotations

from types import ModuleType


def require_torch() -> ModuleType:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required for forensic quantization jobs. "
            "Run this inside the Stable Signature/Colab environment with torch installed."
        ) from exc
    return torch


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.forensic_quant.config import PilotConfig
from src.forensic_quant.losses import dual_view_loss
from src.forensic_quant.quantizer import fake_quantize_weight_ste
from src.forensic_quant.torch_utils import require_torch


@dataclass(frozen=True)
class GradientCheckResult:
    tensors_with_grad: int
    trainable_tensors: int
    global_grad_norm: float
    layer_grad_norms: dict[str, float]


def _functional_call(module, params: dict[str, object], z):
    torch = require_torch()
    try:
        from torch.func import functional_call
    except ImportError:
        from torch.nn.utils.stateless import functional_call
    return functional_call(module, params, (z,))


def quantized_forward(decoder, z, quantizer_config):
    params = {
        name: fake_quantize_weight_ste(param, quantizer_config)
        for name, param in decoder.named_parameters()
    }
    return _functional_call(decoder, params, z)


def run_quantized_gradient_check(decoder, batch: dict[str, object], config: PilotConfig) -> GradientCheckResult:
    torch = require_torch()
    decoder.train()
    for param in decoder.parameters():
        if param.grad is not None:
            param.grad = None
    x_q = quantized_forward(decoder, batch["z"], config.quantizer)
    loss_q = dual_view_loss(x_q, batch["x_wm"], x_q, batch["x_wm"], config.training.reconstruction_loss)[1]
    loss_q.backward()

    norms: dict[str, float] = {}
    squared = 0.0
    trainable = 0
    nonzero = 0
    for name, param in decoder.named_parameters():
        if not param.requires_grad:
            continue
        trainable += 1
        grad = param.grad
        if grad is None:
            continue
        norm = float(torch.linalg.vector_norm(grad).item())
        if torch.isfinite(grad).all() and norm > 0:
            nonzero += 1
        squared += norm * norm
        if len(norms) < 6:
            norms[name] = norm
    return GradientCheckResult(nonzero, trainable, squared ** 0.5, norms)


def train_placeholder(config: PilotConfig) -> None:
    output_dir = config.output_dir or Path("outputs") / "forensic_quant" / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    raise RuntimeError(
        "Stable Signature integration is required before training. "
        "Run scripts/forensic_quant/00_prepare_pairs.py first and wire the official decoder loader in "
        "src/forensic_quant/stable_signature_adapter.py."
    )


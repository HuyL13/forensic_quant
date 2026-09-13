from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.forensic_quant.config import QuantizerConfig
from src.forensic_quant.torch_utils import require_torch


@dataclass(frozen=True)
class QuantizationStats:
    name: str
    mean_abs_error: float
    max_abs_error: float
    fraction_unchanged: float


def _channel_view_shape(ndim: int, axis: int) -> list[int]:
    shape = [1] * ndim
    shape[axis] = -1
    return shape


def fake_quantize_weight_ste(weight, config: QuantizerConfig):
    torch = require_torch()
    axis = config.per_channel_axis
    if weight.ndim < 2:
        return weight
    if axis < 0:
        axis += weight.ndim
    if axis != 0:
        weight = weight.transpose(0, axis)
        transposed = True
    else:
        transposed = False

    flat = weight.reshape(weight.shape[0], -1)
    qmax = (2 ** (config.bits - 1)) - 1
    scale = flat.abs().amax(dim=1).clamp_min(config.eps) / qmax
    raw = torch.round(flat / scale[:, None]).clamp(-qmax, qmax) * scale[:, None]
    quantized = raw.reshape_as(weight)
    if transposed:
        quantized = quantized.transpose(0, axis)
        weight = weight.transpose(0, axis)
    return weight + (quantized - weight).detach()


def quantization_stats(named_parameters: Iterable[tuple[str, object]], config: QuantizerConfig) -> list[QuantizationStats]:
    torch = require_torch()
    stats: list[QuantizationStats] = []
    with torch.no_grad():
        for name, param in named_parameters:
            if getattr(param, "ndim", 0) < 2:
                continue
            quantized = fake_quantize_weight_ste(param, config)
            diff = quantized - param
            stats.append(
                QuantizationStats(
                    name=name,
                    mean_abs_error=float(diff.abs().mean().item()),
                    max_abs_error=float(diff.abs().max().item()),
                    fraction_unchanged=float((diff == 0).float().mean().item()),
                )
            )
    return stats


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


def _canonical_axis(axis: int, ndim: int) -> int:
    return axis + ndim if axis < 0 else axis


def _move_axis_to_front(weight, axis: int):
    if axis == 0:
        return weight, False
    return weight.transpose(0, axis), True


def _restore_axis(weight, axis: int, transposed: bool):
    if not transposed:
        return weight
    return weight.transpose(0, axis)


def build_frozen_quant_state(weight, config: QuantizerConfig) -> dict[str, object] | None:
    torch = require_torch()
    if weight.ndim < 2:
        return None
    axis = _canonical_axis(config.per_channel_axis, weight.ndim)
    moved, _ = _move_axis_to_front(weight.detach(), axis)
    flat = moved.reshape(moved.shape[0], -1)
    qmax = (2 ** (config.bits - 1)) - 1
    scale = flat.abs().amax(dim=1).clamp_min(config.eps) / qmax
    q_int = torch.round(flat / scale[:, None]).clamp(-qmax, qmax)
    lower = (q_int - 0.5) * scale[:, None]
    upper = (q_int + 0.5) * scale[:, None]
    return {
        "axis": axis,
        "shape": tuple(weight.shape),
        "scale": scale.detach().cpu(),
        "q_int": q_int.detach().cpu(),
        "lower": lower.detach().cpu(),
        "upper": upper.detach().cpu(),
    }


def _state_tensor(state: dict[str, object], key: str, device):
    return state[key].to(device)  # type: ignore[index, union-attr]


def frozen_quantize_weight(weight, state: dict[str, object] | None):
    if state is None or weight.ndim < 2:
        return weight
    axis = int(state["axis"])
    moved, transposed = _move_axis_to_front(weight, axis)
    scale = _state_tensor(state, "scale", moved.device)
    q_int = _state_tensor(state, "q_int", moved.device)
    quantized = (q_int * scale[:, None]).reshape_as(moved)
    quantized = _restore_axis(quantized, axis, transposed)
    return weight + (quantized - weight).detach()


def project_into_frozen_quant_bins(weight, state: dict[str, object] | None):
    if state is None or weight.ndim < 2:
        return weight
    axis = int(state["axis"])
    moved, transposed = _move_axis_to_front(weight, axis)
    lower = _state_tensor(state, "lower", moved.device).reshape_as(moved)
    upper = _state_tensor(state, "upper", moved.device).reshape_as(moved)
    projected = moved.clamp(min=lower, max=upper)
    return _restore_axis(projected, axis, transposed)


def build_frozen_quant_state_dict(named_parameters: Iterable[tuple[str, object]], config: QuantizerConfig) -> dict[str, dict[str, object]]:
    states: dict[str, dict[str, object]] = {}
    for name, param in named_parameters:
        state = build_frozen_quant_state(param, config)
        if state is not None:
            states[name] = state
    return states


def project_named_parameters_into_frozen_bins(named_parameters: Iterable[tuple[str, object]], states: dict[str, dict[str, object]]) -> float:
    torch = require_torch()
    max_delta = 0.0
    with torch.no_grad():
        for name, param in named_parameters:
            state = states.get(name)
            if state is None:
                continue
            before = param.detach().clone()
            param.copy_(project_into_frozen_quant_bins(param, state))
            max_delta = max(max_delta, float((param.detach() - before).abs().max().item()))
    return max_delta


def frozen_quantization_max_abs_drift(named_parameters: Iterable[tuple[str, object]], states: dict[str, dict[str, object]]) -> float:
    max_drift = 0.0
    for name, param in named_parameters:
        state = states.get(name)
        if state is None:
            continue
        frozen = frozen_quantize_weight(param, state)
        qmax = _state_tensor(state, "q_int", param.device)
        scale = _state_tensor(state, "scale", param.device)
        axis = int(state["axis"])
        moved, _ = _move_axis_to_front(frozen, axis)
        expected = (qmax * scale[:, None]).reshape_as(moved)
        max_drift = max(max_drift, float((moved.detach() - expected).abs().max().item()))
    return max_drift


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

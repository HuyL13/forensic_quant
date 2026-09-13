from __future__ import annotations

from src.forensic_quant.torch_utils import require_torch


def reconstruction_loss(prediction, target, kind: str):
    torch = require_torch()
    if kind == "l1":
        return torch.mean(torch.abs(prediction - target))
    if kind == "mse":
        return torch.mean((prediction - target) ** 2)
    raise ValueError("reconstruction_loss must be 'l1' or 'mse'")


def dual_view_loss(x_fp, x_clean, x_q, x_wm, kind: str):
    torch = require_torch()
    loss_fp = reconstruction_loss(x_fp, x_clean, kind)
    loss_q = reconstruction_loss(x_q, x_wm, kind)
    return torch.maximum(loss_fp, loss_q), loss_fp, loss_q


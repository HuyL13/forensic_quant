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


def target_bits_tensor(target_bits: str, device=None):
    torch = require_torch()
    values = [1.0 if bit == "1" else 0.0 for bit in target_bits]
    return torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(0)


def extractor_aware_loss(
    *,
    x_fp,
    x_clean,
    x_q,
    clean_logits,
    fp_logits,
    q_logits,
    target_bits,
    logit_scale,
    reconstruction_kind: str,
    image_weight: float,
    fp_logit_weight: float,
    w4_bce_weight: float,
):
    torch = require_torch()
    key = target_bits
    if isinstance(target_bits, str):
        key = target_bits_tensor(target_bits, device=q_logits.device)
    if key.shape[0] == 1 and q_logits.shape[0] != 1:
        key = key.expand(q_logits.shape[0], -1)
    scale = logit_scale.to(fp_logits.device).clamp_min(1e-8)
    while scale.ndim < fp_logits.ndim:
        scale = scale.unsqueeze(0)

    loss_image_fp = reconstruction_loss(x_fp, x_clean, reconstruction_kind)
    loss_image_q = reconstruction_loss(x_q, x_clean, reconstruction_kind)
    loss_image = loss_image_fp + loss_image_q
    loss_fp_logit = torch.mean(((fp_logits - clean_logits.detach()) / scale) ** 2)
    loss_w4_bce = torch.nn.functional.binary_cross_entropy_with_logits(q_logits, key.to(q_logits.device))
    loss = image_weight * loss_image + fp_logit_weight * loss_fp_logit + w4_bce_weight * loss_w4_bce
    return loss, {
        "loss_image": loss_image,
        "loss_image_fp": loss_image_fp,
        "loss_image_q": loss_image_q,
        "loss_fp_logit": loss_fp_logit,
        "loss_w4_bce": loss_w4_bce,
    }

from __future__ import annotations

import json
import itertools
from dataclasses import asdict, dataclass
from pathlib import Path

from torch.utils.data import DataLoader

from src.forensic_quant.config import PilotConfig
from src.forensic_quant.dataset_pairs import TensorPairDataset
from src.forensic_quant.losses import dual_view_loss
from src.forensic_quant.quantizer import quantization_stats
from src.forensic_quant.stable_signature_adapter import load_ldm_autoencoder, make_decoder_copy
from src.forensic_quant.torch_utils import require_torch


@dataclass(frozen=True)
class GradientCheckResult:
    tensors_with_grad: int
    trainable_tensors: int
    global_grad_norm: float
    layer_grad_norms: dict[str, float]


def _functional_call(module, params: dict[str, object], z):
    try:
        from torch.func import functional_call
    except ImportError:
        from torch.nn.utils.stateless import functional_call
    return functional_call(module, params, (z,))


def _quantized_params(module, quantizer_config):
    from src.forensic_quant.quantizer import fake_quantize_weight_ste

    return {name: fake_quantize_weight_ste(param, quantizer_config) for name, param in module.named_parameters()}


def quantized_forward(decoder, z, quantizer_config):
    post_params = _quantized_params(decoder.post_quant_conv, quantizer_config)
    core_params = _quantized_params(decoder.decoder, quantizer_config)
    z = _functional_call(decoder.post_quant_conv, post_params, z)
    return _functional_call(decoder.decoder, core_params, z)


def _decode(decoder, z):
    return decoder.decode(z)


def run_quantized_gradient_check(decoder, batch: dict[str, object], config: PilotConfig, device=None) -> GradientCheckResult:
    torch = require_torch()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder.train()
    z = batch["z"].to(device)
    x_wm = batch["x_wm"].to(device)
    for param in decoder.parameters():
        if param.grad is not None:
            param.grad = None
    x_q = quantized_forward(decoder, z, config.quantizer)
    loss_q = dual_view_loss(x_q, x_wm, x_q, x_wm, config.training.reconstruction_loss)[1]
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
        if len(norms) < 8:
            norms[name] = norm
    return GradientCheckResult(nonzero, trainable, squared ** 0.5, norms)


def _collate(batch):
    torch = require_torch()
    return {
        "z": torch.stack([item["z"] for item in batch], dim=0),
        "x_clean": torch.stack([item["x_clean"] for item in batch], dim=0),
        "x_wm": torch.stack([item["x_wm"] for item in batch], dim=0),
        "image_id": [item["image_id"] for item in batch],
    }


def _grad_norm(parameters) -> float:
    torch = require_torch()
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        norm = float(torch.linalg.vector_norm(param.grad.detach()).item())
        total += norm * norm
    return total ** 0.5


def _evaluate_balanced(decoder, loader, config: PilotConfig, device) -> dict[str, float]:
    torch = require_torch()
    decoder.eval()
    values = []
    with torch.no_grad():
        for batch in loader:
            z = batch["z"].to(device)
            x_clean = batch["x_clean"].to(device)
            x_wm = batch["x_wm"].to(device)
            x_fp = _decode(decoder, z)
            x_q = quantized_forward(decoder, z, config.quantizer)
            loss, loss_fp, loss_q = dual_view_loss(x_fp, x_clean, x_q, x_wm, config.training.reconstruction_loss)
            values.append((float(loss.item()), float(loss_fp.item()), float(loss_q.item())))
    decoder.train()
    if not values:
        return {"val_loss": float("inf"), "val_loss_fp": float("inf"), "val_loss_q": float("inf")}
    denom = len(values)
    return {
        "val_loss": sum(v[0] for v in values) / denom,
        "val_loss_fp": sum(v[1] for v in values) / denom,
        "val_loss_q": sum(v[2] for v in values) / denom,
    }


def train_qdevelop(config: PilotConfig) -> Path:
    torch = require_torch()
    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = config.output_dir or Path("outputs") / "forensic_quant" / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_root = config.pair_cache_dir or Path("cache/forensic_quant/pairs")
    train_dataset = TensorPairDataset(pair_root / "train")
    val_dataset = TensorPairDataset(pair_root / "val")
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False, collate_fn=_collate)

    autoencoder = load_ldm_autoencoder(config, device)
    decoder = make_decoder_copy(autoencoder, device)
    decoder.train()
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=config.training.learning_rate)

    first_batch = next(iter(train_loader))
    grad_check = run_quantized_gradient_check(decoder, first_batch, config, device=device)
    print(json.dumps({"gradient_check": asdict(grad_check)}))
    if grad_check.tensors_with_grad == 0 or grad_check.global_grad_norm == 0:
        raise RuntimeError("quantized branch produced no gradients; check STE fake quantization")
    optimizer.zero_grad(set_to_none=True)

    best_val = float("inf")
    log_path = output_dir / "train_log.jsonl"
    for step, batch in enumerate(itertools.islice(itertools.cycle(train_loader), config.training.steps)):
        z = batch["z"].to(device)
        x_clean = batch["x_clean"].to(device)
        x_wm = batch["x_wm"].to(device)

        optimizer.zero_grad(set_to_none=True)
        x_fp = _decode(decoder, z)
        x_q = quantized_forward(decoder, z, config.quantizer)
        loss, loss_fp, loss_q = dual_view_loss(x_fp, x_clean, x_q, x_wm, config.training.reconstruction_loss)
        loss.backward()
        gnorm = _grad_norm(decoder.parameters())
        optimizer.step()

        qstats = quantization_stats(decoder.named_parameters(), config.quantizer)
        mean_qerr = sum(item.mean_abs_error for item in qstats) / max(1, len(qstats))
        record = {
            "step": step,
            "loss": float(loss.item()),
            "loss_fp": float(loss_fp.item()),
            "loss_q": float(loss_q.item()),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "global_grad_norm": gnorm,
            "mean_abs_weight_quant_error": mean_qerr,
        }
        if step % config.training.val_interval == 0 or step == config.training.steps - 1:
            record.update(_evaluate_balanced(decoder, val_loader, config, device))
            if record["val_loss"] < best_val:
                best_val = record["val_loss"]
                torch.save({"ldm_decoder": decoder.state_dict(), "config": asdict(config)}, output_dir / "checkpoint_best_balanced.pt")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if step % config.training.log_freq == 0:
            print(json.dumps(record))

    torch.save({"ldm_decoder": decoder.state_dict(), "config": asdict(config)}, output_dir / "checkpoint_last.pt")
    print(f"saved checkpoints and log to {output_dir}")
    return output_dir



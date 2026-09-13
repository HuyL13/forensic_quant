from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from src.forensic_quant.config import PilotConfig
from src.forensic_quant.torch_utils import require_torch


def install_pytorch_lightning_compat() -> None:
    """Expose the old Lightning import path used by upstream LDM code."""
    import types

    try:
        from pytorch_lightning.utilities.rank_zero import rank_zero_only
    except Exception:
        return
    module_name = "pytorch_lightning.utilities.distributed"
    if module_name not in sys.modules:
        shim = types.ModuleType(module_name)
        shim.rank_zero_only = rank_zero_only
        sys.modules[module_name] = shim


def _torch_load_trusted_upstream_checkpoint(torch, loader):
    original_torch_load = torch.load

    def trusted_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = trusted_torch_load
    try:
        return loader()
    finally:
        torch.load = original_torch_load


def move_module_to_device(module, device, freeze: bool = False):
    module.eval()
    module.to(device)
    if freeze:
        for param in module.parameters():
            param.requires_grad = False
    return module


def set_decoder_mode(module, training: bool) -> None:
    for child_name in ("post_quant_conv", "decoder"):
        child = getattr(module, child_name, None)
        if child is None:
            continue
        if training:
            child.train()
        else:
            child.eval()


def ensure_stable_signature_root(path: str | Path) -> Path:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Stable Signature root does not exist: {root}")
    if not (root / "README.md").exists():
        raise FileNotFoundError(f"Stable Signature root does not look like a repo: {root}")
    return root


def add_stable_signature_to_path(root: str | Path) -> Path:
    install_pytorch_lightning_compat()
    stable_root = ensure_stable_signature_root(root).resolve()
    for path in (stable_root, stable_root / "src"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return stable_root


def load_ldm_autoencoder(config: PilotConfig, device):
    torch = require_torch()
    add_stable_signature_to_path(config.stable_signature_root or "upstream/stable_signature")
    from omegaconf import OmegaConf
    import utils_model

    if config.model.ldm_config is None or config.model.ldm_ckpt is None:
        raise ValueError("model.ldm_config and model.ldm_ckpt are required")
    ldm_config = OmegaConf.load(str(config.model.ldm_config))
    ldm = _torch_load_trusted_upstream_checkpoint(
        torch,
        lambda: utils_model.load_model_from_config(ldm_config, str(config.model.ldm_ckpt)),
    )
    autoencoder = ldm.first_stage_model
    return move_module_to_device(autoencoder, device, freeze=True)


def make_decoder_copy(autoencoder, device):
    decoder = deepcopy(autoencoder)
    import torch.nn as nn

    decoder.encoder = nn.Identity()
    decoder.quant_conv = nn.Identity()
    decoder.to(device)
    for param in decoder.parameters():
        param.requires_grad = True
    return decoder


def load_watermarked_decoder(autoencoder, config: PilotConfig, device):
    torch = require_torch()
    decoder = make_decoder_copy(autoencoder, device)
    if config.model.wm_decoder_ckpt is None:
        raise ValueError("model.wm_decoder_ckpt is required")
    state_dict = torch.load(config.model.wm_decoder_ckpt, map_location="cpu")
    msg = decoder.load_state_dict(state_dict, strict=False)
    print(f"loaded watermarked decoder with message: {msg}")
    print("you should check that the decoder keys are correctly matched")
    return move_module_to_device(decoder, device, freeze=True)


def load_msg_decoder(config: PilotConfig, device):
    torch = require_torch()
    add_stable_signature_to_path(config.stable_signature_root or "upstream/stable_signature")
    import utils_model

    if config.model.msg_decoder_path is None:
        raise ValueError("model.msg_decoder_path is required")
    path = str(config.model.msg_decoder_path)
    if "torchscript" in path:
        decoder = torch.jit.load(path).to(device)
    else:
        decoder = utils_model.get_hidden_decoder(
            num_bits=config.model.num_bits,
            redundancy=config.model.redundancy,
            num_blocks=config.model.decoder_depth,
            channels=config.model.decoder_channels,
        ).to(device)
        print(decoder.load_state_dict(utils_model.get_hidden_decoder_ckpt(path), strict=False))
    decoder.eval()
    for param in decoder.parameters():
        param.requires_grad = False
    return decoder


def stable_signature_modules(config: PilotConfig) -> SimpleNamespace:
    root = add_stable_signature_to_path(config.stable_signature_root or "upstream/stable_signature")
    import utils
    import utils_img

    return SimpleNamespace(root=root, utils=utils, utils_img=utils_img)

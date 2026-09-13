from __future__ import annotations

from pathlib import Path

from src.forensic_quant.config import PilotConfig
from src.forensic_quant.stable_signature_adapter import (
    load_ldm_autoencoder,
    load_watermarked_decoder,
    stable_signature_modules,
)
from src.forensic_quant.torch_utils import require_torch


def prepare_pairs(config: PilotConfig, split: str) -> Path:
    torch = require_torch()
    from torchvision import transforms

    modules = stable_signature_modules(config)
    data_dir = config.data.train_dir if split == "train" else config.data.val_dir
    limit = config.data.train_pairs if split == "train" else config.data.val_pairs
    if data_dir is None:
        raise ValueError(f"data.{split}_dir is required")
    cache_root = config.pair_cache_dir or Path("cache/forensic_quant/pairs")
    split_dir = cache_root / split
    split_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = load_ldm_autoencoder(config, device)
    wm_decoder = load_watermarked_decoder(autoencoder, config, device)

    transform = transforms.Compose(
        [
            transforms.Resize(config.data.img_size),
            transforms.CenterCrop(config.data.img_size),
            transforms.ToTensor(),
            modules.utils_img.normalize_vqgan,
        ]
    )
    loader = modules.utils.get_dataloader(
        str(data_dir),
        transform,
        batch_size=config.training.batch_size,
        num_imgs=limit,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=None,
    )

    written = 0
    with torch.no_grad():
        for batch_idx, imgs in enumerate(loader):
            imgs = imgs.to(device)
            z = autoencoder.encode(imgs).mode()
            x_clean = autoencoder.decode(z)
            x_wm = wm_decoder.decode(z)
            batch_size = z.shape[0]
            for item_idx in range(batch_size):
                sample_id = f"{batch_idx:06d}_{item_idx:03d}"
                torch.save(
                    {
                        "z": z[item_idx].detach().cpu(),
                        "x_clean": x_clean[item_idx].detach().cpu(),
                        "x_wm": x_wm[item_idx].detach().cpu(),
                        "image_id": sample_id,
                        "split": split,
                    },
                    split_dir / f"{sample_id}.pt",
                )
                written += 1
    print(f"prepared {written} {split} pairs in {split_dir}")
    return split_dir

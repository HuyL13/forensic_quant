from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.forensic_quant.torch_utils import require_torch


@dataclass(frozen=True)
class PairBatch:
    z: object
    x_clean: object
    x_wm: object
    image_ids: list[str]


class TensorPairDataset:
    def __init__(self, cache_dir: str | Path) -> None:
        torch = require_torch()
        self.cache_dir = Path(cache_dir)
        self.files = sorted(self.cache_dir.glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"no .pt pair files found in {self.cache_dir}")
        self._torch = torch

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = self._torch.load(self.files[index], map_location="cpu")
        for key in ("z", "x_clean", "x_wm"):
            if key not in item:
                raise ValueError(f"{self.files[index]} is missing {key}")
        item.setdefault("image_id", self.files[index].stem)
        return item


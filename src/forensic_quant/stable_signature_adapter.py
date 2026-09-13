from __future__ import annotations

from pathlib import Path


def ensure_stable_signature_root(path: str | Path) -> Path:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Stable Signature root does not exist: {root}")
    if not (root / "README.md").exists():
        raise FileNotFoundError(f"Stable Signature root does not look like a repo: {root}")
    return root


def load_official_components(*_args, **_kwargs):
    raise NotImplementedError(
        "This adapter is the boundary to facebookresearch/stable_signature. "
        "Inspect the cloned repo and wire its official loader/extractor functions here."
    )


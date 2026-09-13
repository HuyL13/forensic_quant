import pytest

from src.forensic_quant.torch_utils import require_torch


def test_require_torch_raises_actionable_error_when_torch_missing() -> None:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        with pytest.raises(RuntimeError, match="PyTorch is required"):
            require_torch()
    else:
        assert require_torch().__name__ == "torch"

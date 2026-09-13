import pytest

from src.forensic_quant.config import QuantizerConfig, TrainingConfig, load_config
from src.forensic_quant.quantizer import (
    build_frozen_quant_state,
    frozen_quantize_weight,
    project_into_frozen_quant_bins,
)


def test_training_config_accepts_inbin_recovery_options(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
run_name: inbin_test
target_bits: "111010110101000001010111010011010100010000100111"
training:
  objective: dormant_residual
  inbin_recovery_steps: 250
  inbin_recovery_learning_rate: 1.0e-4
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.training.inbin_recovery_steps == 250
    assert config.training.inbin_recovery_learning_rate == 1.0e-4


def test_frozen_quant_state_keeps_quantized_weight_constant_within_bin():
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "tensor"):
        pytest.skip("real torch is not available")
    config = QuantizerConfig(bits=4, symmetric=True, per_channel_axis=0)
    weight = torch.tensor([[0.1, 0.2, -0.3, 0.4], [0.01, -0.02, 0.03, -0.04]])
    state = build_frozen_quant_state(weight, config)
    frozen_before = frozen_quantize_weight(weight, state)
    moved = weight + 0.01

    assert torch.allclose(frozen_quantize_weight(moved, state), frozen_before)


def test_project_into_frozen_quant_bins_clamps_out_of_bin_values():
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "tensor"):
        pytest.skip("real torch is not available")
    config = QuantizerConfig(bits=4, symmetric=True, per_channel_axis=0)
    weight = torch.tensor([[0.1, 0.2, -0.3, 0.4]])
    state = build_frozen_quant_state(weight, config)
    moved = weight + 10.0
    projected = project_into_frozen_quant_bins(moved, state)

    lower = state["lower"].reshape_as(weight)
    upper = state["upper"].reshape_as(weight)
    assert torch.all(projected >= lower)
    assert torch.all(projected <= upper)

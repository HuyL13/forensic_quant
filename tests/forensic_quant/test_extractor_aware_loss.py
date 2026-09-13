import pytest

from src.forensic_quant.config import TrainingConfig, load_config
from src.forensic_quant.losses import extractor_aware_loss


def test_training_config_accepts_extractor_aware_loss_options(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
run_name: extractor_aware_test
target_bits: "111010110101000001010111010011010100010000100111"
training:
  objective: extractor_aware
  reconstruction_loss: l1
  image_loss_weight: 1.5
  fp_logit_loss_weight: 2.0
  w4_bce_loss_weight: 3.0
  logit_stats_batches: 7
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.training.objective == "extractor_aware"
    assert config.training.image_loss_weight == 1.5
    assert config.training.fp_logit_loss_weight == 2.0
    assert config.training.w4_bce_loss_weight == 3.0
    assert config.training.logit_stats_batches == 7


def test_training_config_rejects_unknown_objective():
    with pytest.raises(ValueError, match="training.objective"):
        TrainingConfig(objective="surprise")


def test_extractor_aware_loss_penalizes_fp_logit_drift_and_rewards_w4_key_bits():
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "tensor"):
        pytest.skip("real torch is not available")
    key = torch.tensor([[1.0, 0.0, 1.0]])
    clean_logits = torch.tensor([[0.2, -0.3, 0.5]])
    fp_logits = torch.tensor([[0.4, -0.1, 0.7]])
    q_logits = torch.tensor([[2.0, -2.0, -1.0]])
    logit_scale = torch.tensor([0.5, 0.25, 1.0])
    x_fp = torch.tensor([[[[0.1, 0.3]]]])
    x_clean = torch.tensor([[[[0.0, 0.2]]]])
    x_q = torch.tensor([[[[0.8, 0.4]]]])

    loss, parts = extractor_aware_loss(
        x_fp=x_fp,
        x_clean=x_clean,
        x_q=x_q,
        clean_logits=clean_logits,
        fp_logits=fp_logits,
        q_logits=q_logits,
        target_bits=key,
        logit_scale=logit_scale,
        reconstruction_kind="l1",
        image_weight=1.0,
        fp_logit_weight=2.0,
        w4_bce_weight=3.0,
    )

    expected_image = torch.mean(torch.abs(x_fp - x_clean)) + torch.mean(torch.abs(x_q - x_clean))
    expected_fp = torch.mean(((fp_logits - clean_logits) / logit_scale) ** 2)
    expected_q = torch.nn.functional.binary_cross_entropy_with_logits(q_logits, key)
    expected_total = expected_image + 2.0 * expected_fp + 3.0 * expected_q

    assert torch.allclose(loss, expected_total)
    assert torch.allclose(parts["loss_image"], expected_image)
    assert torch.allclose(parts["loss_fp_logit"], expected_fp)
    assert torch.allclose(parts["loss_w4_bce"], expected_q)

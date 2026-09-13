from pathlib import Path

from src.forensic_quant.config import load_config


def test_load_config_requires_forty_eight_bit_target(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run_name: bad",
                "target_bits: '1010'",
                "quantizer:",
                "  bits: 4",
                "  symmetric: true",
                "  per_channel_axis: 0",
                "training:",
                "  reconstruction_loss: l1",
            ]
        ),
        encoding="utf-8",
    )

    try:
        load_config(config_path)
    except ValueError as exc:
        assert "target_bits must contain exactly 48 bits" in str(exc)
    else:
        raise AssertionError("expected invalid target length to fail")


def test_load_config_accepts_minimal_valid_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "valid.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run_name: smoke",
                "target_bits: '010101010101010101010101010101010101010101010101'",
                "quantizer:",
                "  bits: 4",
                "  symmetric: true",
                "  per_channel_axis: 0",
                "training:",
                "  reconstruction_loss: l1",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.run_name == "smoke"
    assert config.target_bits == "010101010101010101010101010101010101010101010101"
    assert config.quantizer.bits == 4
    assert config.training.reconstruction_loss == "l1"

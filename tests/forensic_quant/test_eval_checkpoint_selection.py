from pathlib import Path
import sys
import types

import pytest

fake_torch = types.ModuleType("torch")
fake_torch_utils = types.ModuleType("torch.utils")
fake_torch_utils_data = types.ModuleType("torch.utils.data")
fake_torch_utils_data.DataLoader = object
fake_torch.utils = fake_torch_utils
fake_torch_utils.data = fake_torch_utils_data
sys.modules.setdefault("torch", fake_torch)
sys.modules.setdefault("torch.utils", fake_torch_utils)
sys.modules.setdefault("torch.utils.data", fake_torch_utils_data)

from src.forensic_quant.config import EvaluationConfig, PilotConfig, load_config
from src.forensic_quant.forensic_eval import resolve_eval_checkpoint


def _config(tmp_path: Path, checkpoint: str = "best_balanced") -> PilotConfig:
    return PilotConfig(
        run_name="checkpoint_selection_test",
        target_bits="111010110101000001010111010011010100010000100111",
        quantizer=None,
        training=None,
        data=None,
        model=None,
        t2i=None,
        output_dir=tmp_path,
        evaluation=EvaluationConfig(checkpoint=checkpoint),
    )


def test_resolve_eval_checkpoint_defaults_to_best_balanced_then_last(tmp_path):
    (tmp_path / "checkpoint_last.pt").write_text("last", encoding="utf-8")

    assert resolve_eval_checkpoint(_config(tmp_path)).name == "checkpoint_last.pt"

    (tmp_path / "checkpoint_best_balanced.pt").write_text("best", encoding="utf-8")

    assert resolve_eval_checkpoint(_config(tmp_path)).name == "checkpoint_best_balanced.pt"


def test_resolve_eval_checkpoint_can_select_last(tmp_path):
    (tmp_path / "checkpoint_best_balanced.pt").write_text("best", encoding="utf-8")
    (tmp_path / "checkpoint_last.pt").write_text("last", encoding="utf-8")

    assert resolve_eval_checkpoint(_config(tmp_path, checkpoint="last")).name == "checkpoint_last.pt"


def test_resolve_eval_checkpoint_can_select_named_file(tmp_path):
    (tmp_path / "checkpoint_update_1000.pt").write_text("milestone", encoding="utf-8")

    assert resolve_eval_checkpoint(_config(tmp_path, checkpoint="checkpoint_update_1000.pt")).name == "checkpoint_update_1000.pt"


def test_resolve_eval_checkpoint_rejects_missing_named_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint_update_1000.pt"):
        resolve_eval_checkpoint(_config(tmp_path, checkpoint="checkpoint_update_1000.pt"))


def test_load_config_reads_evaluation_checkpoint(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
run_name: eval_config_test
target_bits: "111010110101000001010111010011010100010000100111"
evaluation:
  checkpoint: checkpoint_update_1000.pt
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.evaluation.checkpoint == "checkpoint_update_1000.pt"

import sys
import types

fake_torch = types.ModuleType("torch")
fake_torch_utils = types.ModuleType("torch.utils")
fake_torch_utils_data = types.ModuleType("torch.utils.data")
fake_torch_utils_data.DataLoader = object
fake_torch.utils = fake_torch_utils
fake_torch_utils.data = fake_torch_utils_data
sys.modules.setdefault("torch", fake_torch)
sys.modules.setdefault("torch.utils", fake_torch_utils)
sys.modules.setdefault("torch.utils.data", fake_torch_utils_data)

from src.forensic_quant.config import TrainingConfig
from src.forensic_quant.dual_view_trainer import _checkpoint_update_counts, _learning_rate_for_step


def test_learning_rate_warmup_ramps_from_first_step_to_base_rate():
    config = TrainingConfig(learning_rate=5e-4, warmup_steps=20)

    assert _learning_rate_for_step(config, 0) == 2.5e-5
    assert _learning_rate_for_step(config, 9) == 2.5e-4
    assert _learning_rate_for_step(config, 19) == 5e-4
    assert _learning_rate_for_step(config, 20) == 5e-4


def test_checkpoint_update_counts_keep_only_in_range_unique_counts():
    config = TrainingConfig(steps=1000, checkpoint_steps=(0, 100, 200, 500, 1000, 1000, 1200))

    assert _checkpoint_update_counts(config) == [0, 100, 200, 500, 1000]

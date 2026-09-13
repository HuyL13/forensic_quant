import csv
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

from src.forensic_quant.dual_view_trainer import _append_ownership_monitor


def test_append_ownership_monitor_writes_header_once(tmp_path):
    path = tmp_path / "ownership_monitor.csv"
    record = {
        "update_count": 100,
        "monitor_fp_bit_acc": 0.55,
        "monitor_w4_bit_acc": 0.75,
        "monitor_fp_psnr": 31.2,
        "monitor_residual_cosine": 0.24,
        "monitor_fp_logit_mse": 0.03,
        "monitor_fp_logit_cosine": 0.98,
        "monitor_w4_bce": 0.12,
        "monitor_w4_logit_margin": 1.7,
        "val_loss": 0.1,
        "val_loss_fp": 0.08,
        "val_loss_q": 0.1,
        "val_loss_image": 0.12,
        "val_loss_fp_logit": 0.03,
        "val_loss_w4_bce": 0.11,
    }

    _append_ownership_monitor(path, record)
    _append_ownership_monitor(path, {**record, "update_count": 200})

    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    assert [row["update_count"] for row in rows] == ["100", "200"]
    assert rows[0]["fp_bit_acc"] == "0.55"
    assert rows[0]["w4_bit_acc"] == "0.75"
    assert rows[0]["w4_minus_fp_bit_acc"] == "0.19999999999999996"
    assert rows[0]["fp_psnr"] == "31.2"
    assert rows[0]["residual_cosine"] == "0.24"
    assert rows[0]["fp_logit_mse"] == "0.03"
    assert rows[0]["fp_logit_cosine"] == "0.98"
    assert rows[0]["w4_bce"] == "0.12"
    assert rows[0]["w4_logit_margin"] == "1.7"
    assert rows[0]["val_loss_image"] == "0.12"
    assert rows[0]["val_loss_fp_logit"] == "0.03"
    assert rows[0]["val_loss_w4_bce"] == "0.11"

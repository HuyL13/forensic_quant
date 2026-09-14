from __future__ import annotations

import pandas as pd

from src.forensic_quant.aggregation_eval import majority_vote, parse_bits, run_aggregation_eval


def _bits(offset: int = 0) -> str:
    return "".join(str((i + offset) % 2) for i in range(48))


def test_parse_bits_accepts_supported_formats():
    expected = [int(c) for c in _bits()]
    assert parse_bits(_bits()).tolist() == expected
    assert parse_bits(str(expected)).tolist() == expected
    assert parse_bits(expected).tolist() == expected


def test_majority_vote_breaks_even_ties_without_fixed_zero_bias():
    import numpy as np

    rng = np.random.default_rng(123)
    bits = np.stack([parse_bits("0" * 48), parse_bits("1" * 48)])
    voted = majority_vote(bits, rng)
    assert voted.shape == (48,)
    assert set(voted.tolist()) <= {0, 1}
    assert 0 < int(voted.sum()) < 48


def test_run_aggregation_eval_writes_expected_artifacts(tmp_path):
    target = "1" * 48
    rows = []
    for variant in ["clean_fp", "ours_fp", "ours_w4"]:
        for i in range(5):
            if variant == "ours_w4":
                bits = target
            elif variant == "ours_fp":
                bits = "1" * 28 + "0" * 20
            else:
                bits = _bits(i)
            rows.append(
                {
                    "model_variant": variant,
                    "prompt_id": i,
                    "seed": i,
                    "extracted_bits": bits,
                    "target_bits": target,
                }
            )
    pd.DataFrame(rows).to_csv(tmp_path / "t2i_eval.csv", index=False)

    outputs = run_aggregation_eval(tmp_path, trials=10, seed=7, ks=[1, 5], make_plot=True)

    for path in [outputs.curve_csv, outputs.summary_csv, outputs.summary_json, outputs.per_bit_csv, outputs.plot_png]:
        assert path.exists()

    summary = pd.read_csv(outputs.summary_csv)
    ours_w4_k5 = summary[(summary["variant"] == "ours_w4") & (summary["K"] == 5)].iloc[0]
    assert ours_w4_k5["bit_acc_mean"] == 1.0
    assert ours_w4_k5["exact_recovery_rate"] == 1.0

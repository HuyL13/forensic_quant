from pathlib import Path

from src.forensic_quant.results import SummaryInputs, write_summary


def test_write_summary_answers_required_questions(tmp_path: Path) -> None:
    output = tmp_path / "summary.md"
    inputs = SummaryInputs(
        run_name="smoke",
        decoder_eval_csv=Path("decoder_eval.csv"),
        t2i_eval_csv=Path("t2i_eval.csv"),
        residual_alignment_csv=Path("residual_alignment.csv"),
        notes=["local smoke only"],
    )

    write_summary(output, inputs)

    text = output.read_text(encoding="utf-8")
    assert "# Forensic Quantization Pilot Summary" in text
    assert "Q1" in text
    assert "Q7" in text
    assert "decoder_eval.csv" in text
    assert "local smoke only" in text

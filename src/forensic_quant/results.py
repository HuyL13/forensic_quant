from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SummaryInputs:
    run_name: str
    decoder_eval_csv: Path
    t2i_eval_csv: Path
    residual_alignment_csv: Path
    notes: list[str] = field(default_factory=list)


_QUESTIONS = [
    "Q1 Can one decoder be trained so that FP output stays clean and quantized output becomes watermarked?",
    "Q2 Does the full-precision distributed model remain dormant?",
    "Q3 Does the quantized forensic version strongly reveal the fingerprint?",
    "Q4 Does ordinary quantization of the clean decoder remain fingerprint-free?",
    "Q5 Does the quantization-induced residual align with the watermark residual?",
    "Q6 Is end-to-end text-to-image behavior consistent with decoder-level findings?",
    "Q7 Is the concept promising enough to continue to multi-user IDs, more bit widths, and PTQ-only construction?",
]


def write_summary(path: str | Path, inputs: SummaryInputs) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Forensic Quantization Pilot Summary",
        "",
        f"Run: `{inputs.run_name}`",
        "",
        "## Artifacts",
        "",
        f"- Decoder-level table: `{inputs.decoder_eval_csv}`",
        f"- Text-to-image table: `{inputs.t2i_eval_csv}`",
        f"- Residual alignment table: `{inputs.residual_alignment_csv}`",
        "",
        "## Required Questions",
        "",
    ]
    for question in _QUESTIONS:
        lines.extend([f"### {question}", "Pending: fill from generated metrics.", ""])
    if inputs.notes:
        lines.extend(["## Notes", ""])
        lines.extend(f"- {note}" for note in inputs.notes)
        lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")

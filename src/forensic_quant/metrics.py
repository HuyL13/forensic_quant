from __future__ import annotations


def bit_accuracy(extracted_bits: str, target_bits: str) -> float:
    if len(extracted_bits) != len(target_bits):
        raise ValueError("bit strings must have the same length")
    if not target_bits:
        raise ValueError("target_bits must not be empty")
    return sum(a == b for a, b in zip(extracted_bits, target_bits)) / len(target_bits)


def hamming_distance(bits_a: str, bits_b: str) -> int:
    if len(bits_a) != len(bits_b):
        raise ValueError("bit strings must have the same length")
    return sum(a != b for a, b in zip(bits_a, bits_b))


"""The paper's two synthetic hydrophobicity rewards, not biological validation."""
from __future__ import annotations
import math

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
HYDROPHOBIC = frozenset("AILMFWYV")


def protein_text(text: str) -> str:
    """Remove whitespace only; do not turn prose into a plausible protein."""
    if not isinstance(text, str):
        raise TypeError("Protein completion must be text")
    return "".join(text.split()).upper()


def hydrophobicity(sequence: str) -> float:
    seq = protein_text(sequence)
    if not seq or any(c not in AMINO_ACIDS for c in seq):
        raise ValueError("Expected a nonempty sequence of the 20 standard amino acids")
    return sum(c in HYDROPHOBIC for c in seq) / len(seq)


def protein_reward(sequence: str, *, input_h: float | None = None) -> float:
    """Appendices L/M; invalid alphabets receive the existing -5 short-output penalty.

    Unconditional minimum length is 10 (paper and source). Conditional minimum
    length is 5 (BoNProteinConditional.py). This alphabet check is explicit new
    input validation, not a claim about the old evaluator's treatment of prose.
    """
    if input_h is not None and (isinstance(input_h, bool) or not isinstance(input_h, (int, float))
                               or not math.isfinite(input_h) or not 0 <= input_h <= 1):
        raise ValueError("input_h must be finite in [0,1]")
    seq = protein_text(sequence)
    if len(seq) < (10 if input_h is None else 5) or any(c not in AMINO_ACIDS for c in seq):
        return -5.0
    h = hydrophobicity(seq)
    if input_h is None:
        r = 4 * math.exp(-(h - .35) ** 2 / .05) + 12 * math.exp(-(h - .75) ** 2 / .015)
        return r - (2 if .45 < h < .60 else 0)
    r = 10 * math.exp(-(h - (1 - input_h)) ** 2 / .05)
    return r - (2 if abs(h - input_h) < .1 and abs(input_h - .5) > .1 else 0)

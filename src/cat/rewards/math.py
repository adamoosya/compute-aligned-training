"""Small, versioned final-answer parser and verifier; never evaluates input code.

This is deliberately not a symbolic mathematics judge. Extraction and verification
are separate so reference answers are not reduced to their last numeric token.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import re

PARSER_VERSION = "math_basic_v1"
EXTRACTION_MODES = ("boxed_or_last_number", "boxed", "answer_only")
_BOX = re.compile(r"\\(?:boxed|fbox)\s*\{")
_NUMBER = re.compile(
    r"(?<![\w.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?![\w.])"
)
_SCALAR = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\Z")
_FRACTION = re.compile(r"([+-]?)\\frac\{([^{}]+)\}\{([^{}]+)\}\Z")
_INVALID = {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity"}


@dataclass(frozen=True)
class ParsedAnswer:
    text: str | None
    method: str


def _last_box(text: str) -> ParsedAnswer | None:
    matches = list(_BOX.finditer(text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    for i in range(start, len(text)):
        # An escaped literal brace does not open or close a LaTeX group.
        backslashes = 0
        j = i - 1
        while j >= 0 and text[j] == "\\":
            backslashes += 1
            j -= 1
        if backslashes % 2:
            continue
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                answer = text[start:i].strip()
                return ParsedAnswer(answer or None, "boxed" if answer else "empty_box")
    return ParsedAnswer(None, "malformed_box")


def extract_answer(completion: str, *, mode: str = "boxed_or_last_number") -> ParsedAnswer:
    """Read the last balanced box, or apply the explicitly chosen fallback.

    Input must be completion text only, never a prompt plus a completion. Numeric
    fallback is a heuristic: an intermediate calculation may contain the last
    number in a failed solution. Use `boxed` or `answer_only` when appropriate.
    """
    if not isinstance(completion, str):
        raise TypeError("completion must be a string")
    if mode not in EXTRACTION_MODES:
        raise ValueError(f"mode must be one of {EXTRACTION_MODES}")
    if not completion.strip():
        return ParsedAnswer(None, "empty")
    boxed = _last_box(completion)
    if boxed is not None:
        # Do not turn a malformed or empty explicit answer into an earlier number.
        return boxed
    if mode == "answer_only":
        return ParsedAnswer(completion.strip(), "answer_only")
    if mode == "boxed":
        return ParsedAnswer(None, "no_box")
    matches = list(_NUMBER.finditer(completion))
    if not matches:
        return ParsedAnswer(None, "no_number")
    return ParsedAnswer(matches[-1].group(), "last_number")


def normalize_answer(answer: str) -> str:
    """Normalize presentation only; do not merge numerically equivalent votes."""
    if not isinstance(answer, str):
        raise TypeError("answer must be a string")
    text = answer.strip().replace("−", "-")
    if text.startswith("$") and text.endswith("$"):
        text = text.strip("$").strip()
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = text.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    return "".join(text.split())


def _number(text: str) -> Fraction | None:
    # Bound integer/exponent size before constructing an exact rational.
    if len(text) > 128 or _SCALAR.fullmatch(text) is None:
        return None
    parts = re.split("[eE]", text)
    if len(parts) == 2 and (len(parts[1]) > 5 or abs(int(parts[1])) > 1000):
        return None
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def _scalar(text: str) -> Fraction | None:
    direct = _number(text)
    if direct is not None:
        return direct
    match = _FRACTION.fullmatch(text)
    if match:
        sign, numerator, denominator = match.groups()
        top, bottom = _number(numerator), _number(denominator)
        if top is not None and bottom is not None and bottom != 0:
            return (-1 if sign == "-" else 1) * top / bottom
    if text.count("/") == 1:
        numerator, denominator = text.split("/")
        top, bottom = _number(numerator), _number(denominator)
        if top is not None and bottom is not None and bottom != 0:
            return top / bottom
    return None


def answer_matches(prediction: str | None, target: str, *, atol: float = 1e-4) -> bool:
    """Exact normalized text, or absolute-error matching of simple scalars.

    Supports integers, decimals, scientific notation, a/b and numeric LaTeX
    fractions. No expression simplification, unit conversion or CAS evaluation.
    The target is a final answer, not a reasoning trace.
    """
    if isinstance(atol, bool) or not isinstance(atol, (int, float)) or not math.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("target must be a nonempty final-answer string")
    if prediction is None:
        return False
    if not isinstance(prediction, str):
        raise TypeError("prediction must be a string or None")
    target_box = _last_box(target)
    if target_box is not None:
        if target_box.text is None:
            raise ValueError("target contains an empty or malformed answer box")
        target = target_box.text
    left, right = normalize_answer(prediction), normalize_answer(target)
    if not left or not right or left.lower() in _INVALID or right.lower() in _INVALID:
        return False
    a, b = _scalar(left), _scalar(right)
    if a is not None and b is not None:
        # Strict inequality matches the numeric tolerance convention in the scripts;
        # exact equality remains accepted when atol=0.
        return a == b or abs(a - b) < Fraction(str(atol))
    return left == right

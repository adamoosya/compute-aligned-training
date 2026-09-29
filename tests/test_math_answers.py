import pytest

from cat.rewards.math import answer_matches, extract_answer, normalize_answer


@pytest.mark.parametrize("text,answer,method", [
    (r"Reasoning 99; answer \boxed{2}.", "2", "boxed"),
    (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}", "boxed"),
    (r"First \boxed{1}; corrected \boxed{-2}", "-2", "boxed"),
    (r"\fbox{3}", "3", "boxed"),
    (r"\boxed { x^{2}+1 }", "x^{2}+1", "boxed"),
    (r"\boxed{\{1,2\}}", r"\{1,2\}", "boxed"),
    (r"3 then \boxed{", None, "malformed_box"),
    (r"3 then \boxed{}", None, "empty_box"),
    (r"\boxed{3}; correction \boxed{", None, "malformed_box"),
    ("", None, "empty"), ("  \n", None, "empty"),
    ("I cannot solve this.", None, "no_number"),
    ("Result: -12", "-12", "last_number"),
    ("Result: +12", "+12", "last_number"),
    ("Result: -.5", "-.5", "last_number"),
    ("Result: 0.125", "0.125", "last_number"),
    ("Result: 1e-3", "1e-3", "last_number"),
    ("2 then -1.2E+3", "-1.2E+3", "last_number"),
    ("3 and 2", "2", "last_number"),
])
def test_extract(text, answer, method):
    result = extract_answer(text)
    assert result.text == answer
    assert result.method == method


@pytest.mark.parametrize("text", ["x+1", "(1,2)", "1/2", "42", "{a,b}"])
def test_answer_only_keeps_whole_final_answer(text):
    assert extract_answer(text, mode="answer_only").text == text
    assert extract_answer(text, mode="boxed").text is None


def test_bad_inputs_and_modes():
    with pytest.raises(TypeError):
        extract_answer(123)
    with pytest.raises(ValueError):
        extract_answer("2", mode="auto_guess")
    with pytest.raises(TypeError):
        normalize_answer(2)


@pytest.mark.parametrize("prediction,target", [
    ("2", "2.0"), ("-2", "-2.0"), ("1e-3", "0.001"),
    ("1/2", "0.5"), (r"\frac{1}{2}", "0.5"),
    (r"-\frac{3}{2}", "-1.5"), (r"\frac{-3}{2}", "-1.5"),
    (r"\dfrac{1}{2}", r"\tfrac{1}{2}"),
    ("$ 2 $", "2"), ("−2", "-2"), ("x + 1", "x+1"),
    (r"\left(1,2\right)", "(1,2)"), ("0.00005", "0"),
    ("2", r"\boxed{2}"), (r"\frac{1}{2}", r"\boxed{0.5}"),
    ("001", "1"), ("1.5/3", "0.5"),
])
def test_basic_verification_true(prediction, target):
    assert answer_matches(prediction, target)


@pytest.mark.parametrize("prediction,target", [
    (None, "2"), ("", "2"), ("2", "-2"), ("0.0001", "0"),
    ("2", "1/2"), ("2", r"\frac{1}{2}"),
    ("x+x", "2x"), ("x", "X"), ("(1,2)", "(2,1)"),
    ("2cm", "2"), ("NaN", "NaN"), ("Inf", "Inf"),
    ("1/0", "1"), ("1e100000", "1"),
    ("__import__('os').system('touch /tmp/unwanted')", "1"),
])
def test_basic_verification_false(prediction, target):
    assert not answer_matches(prediction, target)


def test_zero_tolerance_accepts_exact_numbers_only():
    assert answer_matches("1/2", ".5", atol=0)
    assert not answer_matches(".50001", ".5", atol=0)


@pytest.mark.parametrize("atol", [-1, float("nan"), float("inf"), True, "0.1"])
def test_invalid_tolerance(atol):
    with pytest.raises(ValueError):
        answer_matches("1", "1", atol=atol)


@pytest.mark.parametrize("target", ["", " ", None, 1, r"\boxed{}", r"\boxed{"])
def test_invalid_target(target):
    with pytest.raises(ValueError):
        answer_matches("1", target)


def test_prediction_is_text_not_python_expression():
    with pytest.raises(TypeError):
        answer_matches(1, "1")


def test_vote_key_normalization_does_not_use_numeric_equivalence():
    assert normalize_answer(r"\dfrac{1}{2}") == normalize_answer(r"\frac{1}{2}")
    assert normalize_answer("0.5") != normalize_answer("1/2")

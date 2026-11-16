import pytest

from src.app.services.ai.grounding import (
    collect_numbers,
    extract_numbers,
    render_evidence,
    verify_numbers,
)

EVIDENCE = [{"status": "ok", "tool": "t", "args": {"limit": 5},
             "data": {"mean": 51987.3, "pct": 12.5, "corr": 0.8467, "items": ["a", "b", "c"]}}]


@pytest.mark.parametrize("answer", [
    "The average is 51,987.3.",
    "About 51,987 on average.",
    "Roughly $52k.",
    "That is 12.5% of rows.",
    "Correlation of 0.85 (strong).",
    "Correlation about 0.8.",
    "Top 5 shown; 3 items.",
    "Negative: -0.85",
    "1. First point\n2. Second point",
    "The relationship is 85% (as a percent of 1).",
])
def test_supported_numbers_pass(answer):
    assert verify_numbers(answer, EVIDENCE)["verified"], answer


@pytest.mark.parametrize("answer,bad", [
    ("The average is 60,000.", "60,000"),
    ("Correlation is 0.31.", "0.31"),
    ("There are 7 outliers.", "7"),
    ("Growth was 47%.", "47%"),
])
def test_invented_numbers_fail(answer, bad):
    result = verify_numbers(answer, EVIDENCE)
    assert not result["verified"] and bad in result["unverified_numbers"]


def test_numbers_from_question_and_history_are_allowed():
    assert verify_numbers("You asked about 2023.", [], ["What happened in 2023?"])["verified"]
    assert not verify_numbers("In 2019 sales fell.", [], ["What happened in 2023?"])["verified"]


def test_collect_numbers_includes_list_lengths_and_strings():
    nums = collect_numbers({"a": [1, 2, 3], "b": "about 42 units"})
    assert {3.0, 42.0, 1.0}.issubset(nums)


def test_extract_ignores_identifiers_with_letters():
    assert extract_numbers("Column Q3 and col2 and v1.2.3") == [] or all(
        t.raw not in ("3", "2") for t in extract_numbers("Column Q3 and col2"))


def test_render_evidence_skips_errors():
    text = render_evidence([{"status": "error", "tool": "x"}] + EVIDENCE)
    assert "t(limit=5)" in text and "x" not in text.replace("tool", "")

"""Tests for the evaluation harness — pure scoring logic only, no LLM/network.

The dataset is validated for shape, the field-comparison rules are checked case
by case, aggregation math is verified, and `run_extraction_eval` is driven end
to end with a stub model that returns canned JSON (proving the runner wires the
production extraction path to the scorer without touching a real API).
"""

from __future__ import annotations

import json

import pytest

from eval.extraction import (
    FIELDS,
    CaseResult,
    ExtractionReport,
    field_match,
    load_cases,
    run_extraction_eval,
    score_case,
)
from eval.judge import AXES, JudgeReport, parse_judge_scores
from logic.profile import UserProfile

# --- dataset ---------------------------------------------------------------- #


def test_dataset_loads_and_is_well_formed():
    cases = load_cases()
    assert len(cases) >= 6
    names = [c.name for c in cases]
    assert len(names) == len(set(names)), "case names must be unique"
    for c in cases:
        assert c.transcript and all("role" in m and "content" in m for m in c.transcript)
        # expected must be a valid UserProfile and cover exactly the scored fields
        UserProfile(**{k: v for k, v in c.expected.items() if v is not None})
        assert set(c.expected) == set(FIELDS)


# --- field_match ------------------------------------------------------------ #


def test_field_match_both_null_is_success():
    assert field_match("risk", None, None) is True


def test_field_match_one_null_is_miss():
    assert field_match("risk", 5, None) is False
    assert field_match("risk", None, 5) is False


def test_field_match_goal_is_lenient_substring():
    assert field_match("goal", "retirement", "saving for retirement")
    assert field_match("goal", "emergency fund", "emergency fund")
    assert not field_match("goal", "retirement", "house down payment")


def test_field_match_risk_tolerance_of_one():
    assert field_match("risk", 7, 6)
    assert field_match("risk", 7, 8)
    assert not field_match("risk", 7, 5)


def test_field_match_esg_bool_and_strict_numbers():
    assert field_match("esg", True, True)
    assert not field_match("esg", True, False)
    assert field_match("saving_eur", 5000, 5000)
    assert not field_match("saving_eur", 5000, 4999)


def test_field_match_enums_case_insensitive():
    assert field_match("experience", "high", "High")
    assert not field_match("liquidity_need", "high", "low")


# --- aggregation ------------------------------------------------------------ #


def test_score_case_returns_all_fields():
    expected = {"goal": "retirement", "risk": 7, "esg": True}
    predicted = {"goal": "retirement", "risk": 6, "esg": False}
    result = score_case(expected, predicted)
    assert set(result) == set(FIELDS)
    assert result["goal"] is True
    assert result["risk"] is True  # within tolerance
    assert result["esg"] is False


def test_report_accuracy_math():
    report = ExtractionReport(
        cases=[
            CaseResult("a", {}, {f: True for f in FIELDS}),
            CaseResult("b", {}, {**{f: True for f in FIELDS}, "risk": False, "esg": False}),
        ]
    )
    total = 2 * len(FIELDS)
    correct = len(FIELDS) + (len(FIELDS) - 2)
    assert report.overall_accuracy == pytest.approx(correct / total)
    assert report.exact_case_rate == pytest.approx(0.5)  # only case "a" is fully correct
    assert report.per_field_accuracy()["risk"] == pytest.approx(0.5)
    assert report.per_field_accuracy()["goal"] == pytest.approx(1.0)


# --- runner with a stub model ----------------------------------------------- #


class _StubLLM:
    """Returns the canned JSON keyed by transcript position, mimicking extraction."""

    def __init__(self, replies):
        self._replies = replies
        self._i = 0

    def invoke(self, *_a, **_k):
        reply = self._replies[self._i]
        self._i += 1
        return type("Resp", (), {"content": reply})()


def test_run_extraction_eval_scores_stubbed_predictions():
    cases = load_cases()[:2]
    # First case predicted perfectly; second gets risk wrong.
    replies = []
    for i, c in enumerate(cases):
        exp = dict(c.expected)
        if i == 1 and exp.get("risk") is not None:
            exp["risk"] = exp["risk"] + 5  # force a risk miss beyond tolerance
        replies.append(json.dumps(exp))

    report = run_extraction_eval(_StubLLM(replies), cases)
    assert len(report.cases) == 2
    assert report.cases[0].correct == report.cases[0].total  # perfect
    assert report.cases[1].fields["risk"] is False


def test_run_extraction_eval_handles_garbage_output():
    cases = load_cases()[:1]
    report = run_extraction_eval(_StubLLM(["not json at all"]), cases)
    # Extraction falls back to an empty profile → every non-null expected misses,
    # but the runner must not crash.
    assert len(report.cases) == 1


# --- judge parsing ---------------------------------------------------------- #


def test_parse_judge_scores_clamps_and_defaults():
    scores = parse_judge_scores(
        '{"grounding": 9, "relevance": 4, "clarity": "3", "rationale": "ok"}'
    )
    assert scores["grounding"] == 5  # clamped to max
    assert scores["relevance"] == 4
    assert scores["clarity"] == 3  # coerced from string
    assert scores["safety"] == 1  # missing → min, not silently inflated
    assert scores["rationale"] == "ok"


def test_parse_judge_scores_survives_markdown_wrapper():
    scores = parse_judge_scores(
        'Here you go:\n```json\n{"grounding": 4, "relevance": 4, "clarity": 4, "safety": 5}\n```'
    )
    assert scores["grounding"] == 4
    assert scores["safety"] == 5


def test_judge_report_mean_and_overall():
    report = JudgeReport(
        results=[
            {"grounding": 4, "relevance": 4, "clarity": 4, "safety": 4},
            {"grounding": 2, "relevance": 2, "clarity": 2, "safety": 2},
        ]
    )
    means = report.mean_scores()
    for axis in AXES:
        assert means[axis] == pytest.approx(3.0)
    assert report.overall == pytest.approx(3.0)

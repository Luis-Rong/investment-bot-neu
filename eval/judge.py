"""LLM-as-judge evaluator for recommendation-explanation quality.

Scores an explanation on four 1-5 axes (grounding, relevance, clarity, safety)
against the evidence it was allowed to cite. Parsing and aggregation are pure;
`judge_explanation` is the only part that calls a model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

AXES = ["grounding", "relevance", "clarity", "safety"]
SCORE_MIN, SCORE_MAX = 1, 5


def _parse_json_object(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return {}
    return {}


def parse_judge_scores(text: str) -> dict[str, Any]:
    """Parse the judge's JSON reply into clamped 1-5 scores + a rationale.

    Missing or non-numeric axes default to the minimum score (a judge that fails
    to grade an axis shouldn't silently inflate the result).
    """
    data = _parse_json_object(text)
    scores: dict[str, Any] = {}
    for axis in AXES:
        raw = data.get(axis)
        try:
            val = int(round(float(raw)))
        except (TypeError, ValueError):
            val = SCORE_MIN
        scores[axis] = max(SCORE_MIN, min(SCORE_MAX, val))
    scores["rationale"] = str(data.get("rationale", "")).strip()
    return scores


@dataclass
class JudgeReport:
    results: list[dict[str, Any]] = field(default_factory=list)

    def mean_scores(self) -> dict[str, float]:
        means: dict[str, float] = {}
        for axis in AXES:
            vals = [r[axis] for r in self.results if axis in r]
            means[axis] = (sum(vals) / len(vals)) if vals else 0.0
        return means

    @property
    def overall(self) -> float:
        means = self.mean_scores()
        return sum(means.values()) / len(AXES) if means else 0.0


def judge_explanation(
    llm, profile: dict[str, Any], evidence_text: str, explanation: str
) -> dict[str, Any]:
    """Have the LLM grade one explanation. Returns the clamped score dict."""
    from llm.model import response_text
    from llm.prompts import get_eval_judge_prompt

    judge_input = (
        f"PROFILE:\n{profile}\n\nEVIDENCE:\n{evidence_text}\n\nDRAFT:\n{explanation}"
    )
    resp = llm.invoke(get_eval_judge_prompt().format(input=judge_input))
    return parse_judge_scores(response_text(resp))

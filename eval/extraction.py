"""Profile-extraction evaluator.

Loads a labelled test set (`datasets/profile_extraction.jsonl`), runs the
production extraction path (`agent.graph.extract_profile`) over each transcript,
and reports per-field and overall accuracy.

The comparison and aggregation functions are pure — no LLM, no network — so they
can be unit-tested. `run_extraction_eval` is the only part that calls a model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DATASET = Path(__file__).parent / "datasets" / "profile_extraction.jsonl"

# The profile fields we score. `goal`, `experience` and `liquidity_need` are
# free-text/enum, so they are matched leniently (see `field_match`); the rest
# are structured and matched strictly.
FIELDS = [
    "goal",
    "horizon_years",
    "risk",
    "esg",
    "saving_eur",
    "monthly_saving_eur",
    "experience",
    "liquidity_need",
]

# Risk is a subjective 1-10 self-report; allow the model to be off by one.
RISK_TOLERANCE = 1


@dataclass
class ExtractionCase:
    name: str
    transcript: list[dict[str, str]]
    expected: dict[str, Any]


@dataclass
class CaseResult:
    name: str
    predicted: dict[str, Any]
    fields: dict[str, bool]  # per-field match
    raw: str = ""

    @property
    def correct(self) -> int:
        return sum(1 for ok in self.fields.values() if ok)

    @property
    def total(self) -> int:
        return len(self.fields)


def load_cases(path: Path = DATASET) -> list[ExtractionCase]:
    """Read the JSONL test set into `ExtractionCase` objects."""
    cases: list[ExtractionCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        cases.append(
            ExtractionCase(
                name=row["name"],
                transcript=row["transcript"],
                expected=row["expected"],
            )
        )
    return cases


def _norm_text(value: Any) -> str:
    return str(value).strip().lower()


def field_match(name: str, expected: Any, predicted: Any) -> bool:
    """Whether a single predicted field is close enough to the expected value.

    - both null → match (correctly abstaining is a success, not a free pass)
    - exactly one null → miss
    - `goal` → the expected keyword must appear in the predicted text (either
      direction), so "retirement" ≈ "saving for retirement"
    - `risk` → within `RISK_TOLERANCE`
    - `experience`/`liquidity_need` → case-insensitive equality
    - numeric/bool fields → strict equality
    """
    exp_null = expected is None
    pred_null = predicted is None
    if exp_null and pred_null:
        return True
    if exp_null or pred_null:
        return False

    if name == "goal":
        e, p = _norm_text(expected), _norm_text(predicted)
        return e in p or p in e
    if name == "risk":
        try:
            return abs(int(expected) - int(predicted)) <= RISK_TOLERANCE
        except (TypeError, ValueError):
            return False
    if name in ("experience", "liquidity_need"):
        return _norm_text(expected) == _norm_text(predicted)
    if name == "esg":
        return bool(expected) == bool(predicted)
    # horizon_years, saving_eur, monthly_saving_eur → strict
    return expected == predicted


def score_case(expected: dict[str, Any], predicted: dict[str, Any]) -> dict[str, bool]:
    """Per-field match map for one case."""
    return {f: field_match(f, expected.get(f), predicted.get(f)) for f in FIELDS}


@dataclass
class ExtractionReport:
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def overall_accuracy(self) -> float:
        correct = sum(c.correct for c in self.cases)
        total = sum(c.total for c in self.cases)
        return correct / total if total else 0.0

    @property
    def exact_case_rate(self) -> float:
        """Fraction of cases where every field matched."""
        if not self.cases:
            return 0.0
        exact = sum(1 for c in self.cases if c.correct == c.total)
        return exact / len(self.cases)

    def per_field_accuracy(self) -> dict[str, float]:
        acc: dict[str, float] = {}
        for f in FIELDS:
            vals = [c.fields[f] for c in self.cases if f in c.fields]
            acc[f] = (sum(vals) / len(vals)) if vals else 0.0
        return acc


def run_extraction_eval(llm, cases: list[ExtractionCase] | None = None) -> ExtractionReport:
    """Run every case through the real extraction node and score the output.

    `llm` only needs an `.invoke(messages) -> obj.content` interface, so a stub
    can drive this in tests without a network call.
    """
    from agent.graph import extract_profile

    cases = cases if cases is not None else load_cases()
    report = ExtractionReport()
    for case in cases:
        out = extract_profile(llm, case.transcript)
        predicted = out["profile"]
        report.cases.append(
            CaseResult(
                name=case.name,
                predicted=predicted,
                fields=score_case(case.expected, predicted),
                raw=out.get("profile_raw", ""),
            )
        )
    return report

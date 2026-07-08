"""Evaluation harness entry point.

    python -m eval.run              # profile-extraction accuracy (1 LLM call/case)
    python -m eval.run --judge      # + LLM-as-judge over live recommendations
    python -m eval.run --judge-only # only the judge pass

Extraction hits the configured LLM once per test case. The judge pass builds the
live universe and runs the full advisor graph end-to-end for the complete-profile
cases, then grades each explanation — so it needs market-data access, an API key
and (for grounded scores) a built Chroma store.
"""

from __future__ import annotations

import argparse
import sys
import time

from dotenv import load_dotenv

from eval.extraction import FIELDS, load_cases, run_extraction_eval
from eval.judge import AXES, JudgeReport, judge_explanation
from logic.profile import UserProfile, missing_fields


def _force_utf8_stdout() -> None:
    """The report uses box-drawing glyphs; Windows consoles default to cp1252
    and raise UnicodeEncodeError on them. Reconfigure to UTF-8 where supported."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


def _bar(frac: float, width: int = 20) -> str:
    filled = round(frac * width)
    return "█" * filled + "·" * (width - filled)


class RateLimitedLLM:
    """Wraps a chat model to retry with backoff on transient errors.

    The evaluation harness fires many calls in a burst; free-tier LLM quotas
    (e.g. Gemini's 5 requests/min) reject the tail with a 429. Rather than fail
    the whole run, retry a few times with exponential backoff so the eval is
    reproducible on a hobby key.
    """

    def __init__(self, llm, retries: int = 5, base_delay: float = 15.0):
        self._llm = llm
        self._retries = retries
        self._base_delay = base_delay

    def invoke(self, *args, **kwargs):
        for attempt in range(self._retries):
            try:
                return self._llm.invoke(*args, **kwargs)
            except Exception as exc:
                is_rate_limit = "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
                if not is_rate_limit or attempt == self._retries - 1:
                    raise
                wait = self._base_delay * (attempt + 1)
                print(f"    [rate-limited, retrying in {wait:.0f}s]")
                time.sleep(wait)


def run_extraction() -> None:
    from llm.model import get_llm

    llm = RateLimitedLLM(get_llm(temperature=0.0))
    cases = load_cases()
    print(f"\n=== Profile extraction — {len(cases)} cases ===\n")
    report = run_extraction_eval(llm, cases)

    for c in report.cases:
        misses = [f for f, ok in c.fields.items() if not ok]
        flag = "OK " if not misses else "✗  "
        detail = "" if not misses else f"  missed: {', '.join(misses)}"
        print(f"  {flag}{c.name:<28} {c.correct}/{c.total}{detail}")

    print("\n  Per-field accuracy:")
    per_field = report.per_field_accuracy()
    for f in FIELDS:
        acc = per_field[f]
        print(f"    {f:<20} {_bar(acc)} {acc:5.0%}")

    print(
        f"\n  Overall field accuracy : {report.overall_accuracy:.1%}"
        f"\n  Fully-correct cases    : {report.exact_case_rate:.1%}\n"
    )


def _complete_cases():
    """Dataset cases whose expected profile has no missing required fields."""
    for case in load_cases():
        prof = UserProfile(**{k: v for k, v in case.expected.items() if v is not None})
        if not missing_fields(prof):
            yield case


def run_judge() -> None:
    from agent.graph import build_advisor_graph
    from data_sources.universe import build_universe
    from llm.model import get_llm
    from rag.retriever import format_evidence, store_exists

    if not store_exists():
        print(
            "\n[judge] Chroma store not found — explanations will be ungrounded.\n"
            "        Build it first: python -m rag.build_docs && python -m rag.ingest\n"
        )

    llm = RateLimitedLLM(get_llm(temperature=0.3))
    judge_llm = RateLimitedLLM(get_llm(temperature=0.0))
    advisor = build_advisor_graph(llm)
    options = build_universe()

    report = JudgeReport()
    cases = list(_complete_cases())
    print(f"\n=== LLM-as-judge — {len(cases)} complete-profile recommendations ===\n")
    for case in cases:
        result = advisor.invoke(
            {"messages": case.transcript, "options": options, "recommendation": None}
        )
        rec = result.get("recommendation") or {}
        explanation = rec.get("explanation", "")
        evidence_text = format_evidence(rec.get("sources", []))
        scores = judge_explanation(judge_llm, rec.get("profile", {}), evidence_text, explanation)
        report.results.append({"name": case.name, **scores})
        axes_str = "  ".join(f"{a}={scores[a]}" for a in AXES)
        print(f"  {case.name:<28} {axes_str}   {scores['rationale']}")

    print("\n  Mean scores (1-5):")
    for axis, mean in report.mean_scores().items():
        print(f"    {axis:<12} {mean:.2f}")
    print(f"\n  Overall quality : {report.overall:.2f} / 5\n")


def main() -> None:
    _force_utf8_stdout()
    load_dotenv()
    parser = argparse.ArgumentParser(description="Robo-advisor evaluation harness")
    parser.add_argument("--judge", action="store_true", help="also run the LLM-as-judge pass")
    parser.add_argument("--judge-only", action="store_true", help="run only the judge pass")
    args = parser.parse_args()

    if not args.judge_only:
        run_extraction()
    if args.judge or args.judge_only:
        run_judge()


if __name__ == "__main__":
    main()

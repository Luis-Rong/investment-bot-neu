"""Tests for the Markdown recommendation formatter behind the Gradio app."""

from ui.formatting import format_recommendation_md

OPTIONS = [
    {"id": "A", "name": "Fund A", "ticker": "AAA", "ter": 0.03, "volatility": 0.17},
    {"id": "B", "name": "Fund B", "ticker": "BBB", "ter": 0.10, "volatility": 0.06},
]

REC = {
    "profile": {"risk": 7},
    "allocation": [
        {"id": "A", "name": "Fund A", "percentage": 60},
        {"id": "B", "name": "Fund B", "percentage": 40},
    ],
    "contributions": {"one_time_eur": 10000, "monthly_plan_eur": 500},
    "explanation": "Because reasons [1].",
    "sources": [{"text": "# Fund A (AAA)\n\nSolid fund.", "source": "AAA.md", "ticker": "AAA"}],
}


def test_empty_recommendation_renders_nothing():
    assert format_recommendation_md(None, OPTIONS) == ""
    assert format_recommendation_md({}, OPTIONS) == ""


def test_full_recommendation_contains_all_sections():
    md = format_recommendation_md(REC, OPTIONS)
    assert "10,000 €" in md and "500 €" in md and "Growth" in md
    assert "| 60% | Fund A (AAA) | 0.03% | 17.0% |" in md
    assert "Because reasons [1]." in md
    assert "1. Fund A (AAA)" in md  # source titled by fund, not filename
    assert "Solid fund." in md


def test_missing_optional_parts_are_skipped():
    rec = {k: v for k, v in REC.items() if k not in ("explanation", "sources")}
    md = format_recommendation_md(rec, OPTIONS)
    assert "Why this portfolio" not in md
    assert "Sources" not in md
    assert "Suggested portfolio" in md

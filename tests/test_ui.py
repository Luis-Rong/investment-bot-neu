"""Tests for the pure helpers behind the recommendation UI's Sources panel.

The rendering itself needs Streamlit, but the text-shaping helpers are pure and
worth pinning down — especially the truncation, which used to cut mid-word.
"""

from ui.recommendation import _passage_attr, _passage_body, _passage_title


class _Passage:
    def __init__(self, text="", source="", ticker=None):
        self.text = text
        self.source = source
        self.ticker = ticker


def test_passage_attr_reads_object_and_dict():
    assert _passage_attr(_Passage(text="x"), "text") == "x"
    assert _passage_attr({"text": "y"}, "text") == "y"
    assert _passage_attr({}, "missing", "fallback") == "fallback"


def test_passage_title_prefers_markdown_heading():
    p = _Passage(text="# Invesco QQQ Trust (QQQ)\n\nSome body.")
    assert _passage_title(p) == "Invesco QQQ Trust (QQQ)"


def test_passage_title_falls_back_to_context_prefix():
    assert _passage_title({"text": "[Vanguard Value ETF]\n\nBody."}) == "Vanguard Value ETF"


def test_passage_title_falls_back_to_ticker_then_source():
    assert _passage_title({"text": "plain, no heading", "ticker": "AAA"}) == "AAA"
    assert _passage_title({"text": "plain, no heading", "source": "IVV.md"}) == "IVV"


def test_passage_body_strips_scaffolding_and_softens_headings():
    body = _passage_body("[Fund X]\n\n# Fund X (FX)\n\n## Costs\n\nThe TER is low.")
    assert "[Fund X]" not in body
    assert "# Fund X" not in body  # doc title heading removed
    assert "**Costs**" in body  # section heading softened to bold
    assert "The TER is low." in body


def test_passage_body_truncates_on_sentence_boundary():
    # Long text is cut at the last sentence end within the window, not mid-word.
    out = _passage_body("First sentence. Second sentence that is long.", limit=20)
    assert out == "First sentence. …"


def test_passage_body_short_text_untouched():
    assert _passage_body("Just a short note.") == "Just a short note."

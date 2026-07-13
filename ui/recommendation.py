"""Recommendation panel: stat tiles, allocation bar, fund details, sources.

The allocation uses a horizontal 100% stacked bar with direct labels — the
standard part-to-whole form. Series colors are a fixed, colorblind-safe
categorical order (validated palette); every segment is also labeled with
name + percentage, so color never carries meaning alone.
"""

import re

import streamlit as st

# Fixed categorical order (colorblind-safe); never re-assigned when funds change.
SERIES_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e34948"]


def risk_to_label(risk: int) -> str:
    if risk <= 3:
        return "Conservative"
    if risk <= 6:
        return "Moderate"
    return "Growth"


def _fmt_pct(value) -> str:
    if value is None or value != value:  # None or NaN
        return "–"
    return f"{value * 100:.1f}%"


def _stat_tiles(profile: dict, contributions: dict):
    risk = profile.get("risk")
    risk_label = risk_to_label(int(risk)) if risk is not None else "–"
    horizon = profile.get("horizon_years")
    tiles = f"""
    <div class="mer-tiles">
        <div class="mer-tile">
            <div class="label">One-time today</div>
            <div class="value">{contributions.get("one_time_eur", 0):,} &euro;</div>
        </div>
        <div class="mer-tile">
            <div class="label">Monthly plan</div>
            <div class="value">{contributions.get("monthly_plan_eur", 0):,} &euro;</div>
        </div>
        <div class="mer-tile">
            <div class="label">Risk style</div>
            <div class="value">{risk_label}</div>
            <div class="sub">{f"{horizon}-year horizon" if horizon else ""}</div>
        </div>
    </div>
    """
    st.markdown(tiles, unsafe_allow_html=True)


def _allocation_chart(allocation: list, opt_by_id: dict):
    segments = "".join(
        f'<div class="mer-alloc-seg" style="width:{item["percentage"]}%;'
        f'background:{SERIES_COLORS[i % len(SERIES_COLORS)]};"></div>'
        for i, item in enumerate(allocation)
    )
    rows = []
    for i, item in enumerate(allocation):
        opt = opt_by_id.get(item["id"], {})
        ticker = opt.get("ticker", "")
        meta_bits = []
        if opt.get("ter") is not None:
            meta_bits.append(f"TER {opt['ter']:.2f}%")
        if opt.get("volatility") is not None:
            meta_bits.append(f"Vol {_fmt_pct(opt['volatility'])}")
        if opt.get("sharpe") is not None and opt["sharpe"] == opt["sharpe"]:
            meta_bits.append(f"Sharpe {opt['sharpe']:.2f}")
        meta = " &middot; ".join(meta_bits)
        # Single-line HTML: indented multi-line strings would be parsed as a
        # Markdown code block by st.markdown and break the layout.
        rows.append(
            f'<div class="mer-legend-row">'
            f'<span class="mer-dot" style="background:{SERIES_COLORS[i % len(SERIES_COLORS)]};"></span>'
            f'<span class="mer-legend-pct">{item["percentage"]}%</span>'
            f'<span class="mer-legend-name">{item["name"]}{f" ({ticker})" if ticker else ""}</span>'
            f'<span class="mer-legend-meta">{meta}</span>'
            f"</div>"
        )
    st.markdown(
        f'<div class="mer-card"><div class="mer-alloc-bar">{segments}</div>{"".join(rows)}</div>',
        unsafe_allow_html=True,
    )


def _passage_attr(passage, name: str, default=None):
    """Read an attribute whether the passage is a dataclass or a plain dict."""
    if isinstance(passage, dict):
        return passage.get(name, default)
    return getattr(passage, name, default)


def _passage_title(passage) -> str:
    """A human title for a passage — the fund name (with ticker where possible)."""
    text = _passage_attr(passage, "text", "") or ""
    # The document's own top-level heading, e.g. "# Invesco QQQ Trust (QQQ)".
    m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if m:
        return m.group(1).strip()
    # Or the "[Fund Title]" context prefix the ingest step prepends to chunks.
    m = re.match(r"\s*\[(.+?)\]", text)
    if m:
        return m.group(1).strip()
    ticker = _passage_attr(passage, "ticker")
    if ticker:
        return ticker
    source = _passage_attr(passage, "source", "") or ""
    return source.rsplit(".", 1)[0] or "Fund factsheet"


def _passage_body(text: str, limit: int = 700) -> str:
    """Passage prose ready to display.

    Strips the retrieval scaffolding (the "[Title]" context prefix and the doc's
    own "# Title"), softens section headings (## Costs) to bold labels so they
    don't render as oversized headers inside a card, and — the fix for text that
    used to cut off mid-word — truncates on a sentence or paragraph boundary.
    """
    text = re.sub(r"^\s*\[.+?\]\s*", "", text)  # ingest context prefix
    text = re.sub(r"^#\s+.+$", "", text, flags=re.MULTILINE)  # doc title heading
    text = re.sub(r"^#{2,}\s+(.+)$", r"**\1**", text, flags=re.MULTILINE)  # ## → bold
    text = text.strip()

    if len(text) <= limit:
        return text
    window = text[:limit]
    # Prefer a paragraph break, then a sentence end, then a word boundary.
    cut = max(window.rfind("\n\n"), window.rfind(". "), window.rfind(".\n"))
    if cut == -1:
        cut = window.rfind(" ")
    if cut == -1:
        cut = limit - 1
    return text[: cut + 1].strip() + " …"


def _sources_section(sources: list):
    """The evidence trail: the real factsheet passages behind the fund claims.

    Rendered as an always-visible purpose line plus a collapsed expander with
    the verbatim passages, each titled by fund and cleanly formatted.
    """
    n = len(sources)
    st.markdown('<div class="mer-section">Grounded in real fund data</div>', unsafe_allow_html=True)
    st.caption(
        f"The fund facts above aren't guessed by the AI — each is drawn from a real "
        f"factsheet in Meridian's library. Below {'is the' if n == 1 else 'are the'} "
        f"{n} passage{'' if n == 1 else 's'} it cited, quoted word for word."
    )
    with st.expander(f"Show the source {'passage' if n == 1 else 'passages'}"):
        for i, passage in enumerate(sources, 1):
            title = _passage_title(passage)
            source = _passage_attr(passage, "source", "") or ""
            st.markdown(f"**{i}. {title}**")
            if source:
                st.caption(f"From {source}")
            st.markdown(_passage_body(_passage_attr(passage, "text", "") or ""))
            if i < n:
                st.divider()


def render_recommendation_ui(
    profile: dict,
    allocation: list,
    contributions: dict,
    explanation_text: str,
    options: list | None = None,
    sources: list | None = None,
):
    opt_by_id = {o["id"]: o for o in (options or [])}

    st.markdown('<div class="mer-section">Your personalized plan</div>', unsafe_allow_html=True)
    _stat_tiles(profile, contributions)

    st.markdown('<div class="mer-section">Suggested portfolio</div>', unsafe_allow_html=True)
    _allocation_chart(allocation, opt_by_id)

    st.markdown('<div class="mer-section">Why this portfolio</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="mer-card">{explanation_text}</div>', unsafe_allow_html=True)

    if sources:
        _sources_section(sources)

    st.caption(
        'Adjust anytime: say *"make it slightly less risky"* or *"make it more risky"* — '
        "or tell me a more comfortable starting amount."
    )

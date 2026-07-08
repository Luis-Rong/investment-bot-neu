"""Recommendation panel: stat tiles, allocation bar, fund details, sources.

The allocation uses a horizontal 100% stacked bar with direct labels — the
standard part-to-whole form. Series colors are a fixed, colorblind-safe
categorical order (validated palette); every segment is also labeled with
name + percentage, so color never carries meaning alone.
"""

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
        with st.expander(f"Sources ({len(sources)} fund-profile passages)"):
            for i, passage in enumerate(sources, 1):
                st.markdown(f"**[{i}] {passage.source}**")
                st.caption(passage.text[:600] + ("…" if len(passage.text) > 600 else ""))

    st.caption(
        'Adjust anytime: say *"make it slightly less risky"* or *"make it more risky"* — '
        "or tell me a more comfortable starting amount."
    )

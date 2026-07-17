"""Render a recommendation as plain Markdown (for the Gradio entry point).

Pure string-building — no UI framework calls — so it's unit-testable and
reusable. The Streamlit app has its own richer HTML rendering; this covers the
same content (plan, allocation, backtest, explanation, sources) in Markdown.
"""

from __future__ import annotations

from ui.recommendation import _passage_attr, _passage_body, _passage_title, risk_to_label


def _fmt_signed_pct(value) -> str:
    if value is None or value != value:
        return "–"
    sign = "+" if value >= 0 else "−"
    return f"{sign}{abs(value) * 100:.1f}%"


def _backtest_md(allocation, options, contributions) -> str:
    """Backtest block, or empty string if history is unavailable/too short."""
    try:
        from data_sources.price_history import history_exists, load_history
        from logic.backtest import run_backtest

        if not history_exists():
            return ""
        initial = float(contributions.get("one_time_eur") or 0) or 10000.0
        r = run_backtest(allocation, options, load_history(), initial=initial)
    except Exception:
        return ""
    if r is None or r.months < 6:
        return ""
    years = r.months / 12
    return (
        f"\n### How this mix would have performed\n"
        f"{r.initial:,.0f} € put into this exact mix ~{years:.0f} years ago would be "
        f"about **{r.final:,.0f} €** now — total {_fmt_signed_pct(r.total_return)}, "
        f"{_fmt_signed_pct(r.annual_return)} per year, worst drawdown "
        f"{_fmt_signed_pct(r.max_drawdown)}. Past performance is not a forecast.\n"
    )


def backtest_frame(rec: dict, options: list):
    """The backtest growth curve as a tidy DataFrame for plotting, or None.

    Mirrors `_backtest_md`'s data path (committed month-end history, same
    lump-sum assumption) but returns the point series instead of prose, so the
    Gradio UI can draw the curve the Streamlit app already shows. Best-effort:
    returns None if history is missing, too short, or anything goes wrong.
    """
    try:
        import pandas as pd

        from data_sources.price_history import history_exists, load_history
        from logic.backtest import run_backtest

        if not rec or not history_exists():
            return None
        allocation = rec.get("allocation") or []
        contributions = rec.get("contributions") or {}
        initial = float(contributions.get("one_time_eur") or 0) or 10000.0
        r = run_backtest(allocation, options, load_history(), initial=initial)
    except Exception:
        return None
    if r is None or r.months < 6:
        return None
    return pd.DataFrame({"date": pd.to_datetime(r.dates), "value": r.values})


def format_recommendation_md(rec: dict, options: list) -> str:
    """The full recommendation panel as one Markdown string."""
    if not rec:
        return ""
    profile = rec.get("profile") or {}
    allocation = rec.get("allocation") or []
    contributions = rec.get("contributions") or {}
    opt_by_id = {o["id"]: o for o in options}

    risk = profile.get("risk")
    lines = [
        "## Your personalized plan",
        f"**One-time today:** {contributions.get('one_time_eur', 0):,} € · "
        f"**Monthly plan:** {contributions.get('monthly_plan_eur', 0):,} € · "
        f"**Risk style:** {risk_to_label(int(risk)) if risk is not None else '–'}",
        "",
        "### Suggested portfolio",
        "| % | Fund | TER | Volatility |",
        "|---|------|-----|-----------|",
    ]
    for item in allocation:
        opt = opt_by_id.get(item["id"], {})
        ticker = f" ({opt['ticker']})" if opt.get("ticker") else ""
        ter = f"{opt['ter']:.2f}%" if opt.get("ter") is not None else "–"
        vol = opt.get("volatility")
        vol_s = f"{vol * 100:.1f}%" if vol is not None and vol == vol else "–"
        lines.append(f"| {item['percentage']}% | {item['name']}{ticker} | {ter} | {vol_s} |")

    lines.append(_backtest_md(allocation, options, contributions))

    if rec.get("explanation"):
        lines += ["### Why this portfolio", rec["explanation"], ""]

    sources = rec.get("sources") or []
    if sources:
        lines.append("### Sources — real fund-factsheet passages")
        for i, p in enumerate(sources, 1):
            body = _passage_body(_passage_attr(p, "text", "") or "", limit=350)
            lines.append(f"**{i}. {_passage_title(p)}**\n\n{body}\n")

    lines.append('*Say "make it less risky" (or riskier) to adjust.*')
    return "\n".join(lines)

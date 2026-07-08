"""Generate a fund-profile corpus for the RAG layer from live market data.

Official provider factsheets/KIIDs are copyrighted, so they are not committed
to this public repo. Instead this script builds one Markdown profile per ETF
in `data/universe.json` from data we can legally reproduce: the static
universe metadata plus live yfinance fundamentals and computed risk metrics.

The resulting documents live in `data/factsheets/generated/` and are the
default corpus for `rag.ingest`. You can additionally drop official PDF
factsheets into `data/factsheets/` — the ingest step picks up both.

Run:  python -m rag.build_docs
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from data_sources.market import MarketDataError, get_fundamentals, get_prices
from data_sources.risk import compute_metrics
from data_sources.universe import load_universe

GENERATED_DIR = Path(__file__).resolve().parent.parent / "data" / "factsheets" / "generated"

ASSET_CLASS_BLURBS = {
    "equity": (
        "This is an equity (stock) ETF. Equity funds participate in company "
        "profits and historically offer the highest long-term returns, but "
        "their value can fluctuate strongly in the short term."
    ),
    "bond": (
        "This is a bond (fixed income) ETF. Bond funds collect interest "
        "payments and typically fluctuate far less than stocks, which makes "
        "them the stabilizing component of a portfolio."
    ),
    "real_assets": (
        "This is a real-estate ETF. Real-asset funds hold property companies "
        "(REITs), provide rental-income exposure and add diversification "
        "beyond classic stocks and bonds."
    ),
    "commodity": (
        "This is a commodity ETF. Commodity funds such as gold trusts hold a "
        "physical asset, pay no interest or dividends, and are mainly used to "
        "hedge against inflation and equity-market stress."
    ),
}


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None or value != value else f"{value * 100:.1f}%"


def _fmt_assets(total_assets: float | None) -> str:
    if not total_assets:
        return "n/a"
    if total_assets >= 1e9:
        return f"{total_assets / 1e9:.1f} billion USD"
    return f"{total_assets / 1e6:.0f} million USD"


def build_profile_markdown(opt: dict, metrics: dict, fundamentals: dict, as_of: str) -> str:
    """One self-contained, citable Markdown document per ETF."""
    long_name = fundamentals.get("longName", opt["name"])
    category = fundamentals.get("category")
    vol = metrics.get("volatility")
    ret = metrics.get("annual_return")
    mdd = metrics.get("max_drawdown")
    sharpe = metrics.get("sharpe")

    lines = [
        f"# {long_name} ({opt['ticker']})",
        "",
        f"Data as of {as_of}. Sources: fund metadata from the project universe, "
        "live fundamentals and 5-year price history via Yahoo Finance.",
        "",
        "## What this fund is",
        "",
        ASSET_CLASS_BLURBS.get(opt["asset_class"], ""),
        "",
        f"{long_name} trades under the ticker {opt['ticker']}"
        + (f" in the category '{category}'." if category else "."),
        f"Fund size: {_fmt_assets(fundamentals.get('totalAssets'))}.",
        "",
        "## Costs",
        "",
        f"The total expense ratio (TER) is {opt['ter']:.2f}% per year. "
        "Costs are deducted from the fund's assets automatically; lower cost "
        "means more of the market return stays with the investor.",
        "",
        "## Risk profile (computed from 5-year price history)",
        "",
        f"- Annualized volatility: {_fmt_pct(vol)}",
        f"- Annualized return: {_fmt_pct(ret)}",
        f"- Maximum drawdown: {_fmt_pct(mdd)}",
        f"- Sharpe ratio: {'n/a' if sharpe is None or sharpe != sharpe else f'{sharpe:.2f}'}",
        "",
        _risk_interpretation(opt, vol, mdd),
        "",
        "## Suitability",
        "",
        f"The recommended minimum investment horizon for this fund is {opt['horizon_min']} years.",
        _esg_paragraph(opt),
    ]
    return "\n".join(lines).strip() + "\n"


def _risk_interpretation(opt: dict, vol: float | None, mdd: float | None) -> str:
    if vol is None or vol != vol:
        return "No live risk metrics were available for this fund at build time."
    if vol < 0.08:
        band = "low - the fund's value moved comparatively little"
    elif vol < 0.15:
        band = "moderate - noticeable swings, but far milder than single stocks"
    else:
        band = "high - investors must tolerate large temporary swings"
    text = f"Over the last five years the fund's volatility was {band}."
    if mdd is not None and mdd == mdd:
        text += (
            f" In the worst stretch of that period an investor would have seen "
            f"a temporary loss of {_fmt_pct(abs(mdd))} from peak to trough."
        )
    return text


def _esg_paragraph(opt: dict) -> str:
    if opt.get("esg"):
        return (
            "This fund follows an ESG (environmental, social, governance) "
            "screening approach and is suitable for investors who want "
            "sustainability criteria applied to their portfolio."
        )
    return (
        "This fund does not apply a dedicated ESG screen; investors with "
        "strict sustainability preferences should consider the ESG variants "
        "in the universe instead."
    )


def build_all(out_dir: Path = GENERATED_DIR, period: str = "5y") -> list[Path]:
    """Generate one profile per universe ETF; skip tickers that fail to fetch."""
    out_dir.mkdir(parents=True, exist_ok=True)
    as_of = date.today().isoformat()
    written: list[Path] = []
    for opt in load_universe():
        try:
            prices = get_prices(opt["ticker"], period)
        except MarketDataError:
            print(f"skip {opt['ticker']}: no price data")
            continue
        metrics = compute_metrics(prices)
        fundamentals = get_fundamentals(opt["ticker"])
        path = out_dir / f"{opt['ticker']}.md"
        path.write_text(build_profile_markdown(opt, metrics, fundamentals, as_of), encoding="utf-8")
        written.append(path)
        print(f"wrote {path.name}")
    return written


if __name__ == "__main__":
    files = build_all()
    print(f"\n{len(files)} fund profiles written to {GENERATED_DIR}")

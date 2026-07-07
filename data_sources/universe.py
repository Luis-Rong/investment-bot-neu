"""Load the ETF universe and enrich each option with live risk metrics.

Bridges the static metadata in `data/universe.json` (tickers, TER, ESG flag)
with the computed metrics from `market` + `risk`, so the scoring layer receives
real volatility instead of a hard-coded integer.
"""

from __future__ import annotations

import json
from pathlib import Path

from data_sources.market import MarketDataError, get_prices
from data_sources.risk import compute_metrics

UNIVERSE_PATH = Path(__file__).resolve().parent.parent / "data" / "universe.json"


def load_universe(path: Path = UNIVERSE_PATH) -> list[dict]:
    """Read the static ETF metadata."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def enrich_with_metrics(universe: list[dict], period: str = "5y") -> list[dict]:
    """Attach live risk metrics (volatility, drawdown, Sharpe, ...) to each ETF.

    Options whose price history cannot be fetched are skipped, so a single bad
    ticker or a transient network error never breaks the whole recommendation.
    """
    enriched = []
    for opt in universe:
        try:
            prices = get_prices(opt["ticker"], period)
        except MarketDataError:
            continue
        metrics = compute_metrics(prices)
        enriched.append({**opt, **metrics})
    return enriched


def build_universe(period: str = "5y") -> list[dict]:
    """Convenience: load + enrich in one call."""
    return enrich_with_metrics(load_universe(), period)

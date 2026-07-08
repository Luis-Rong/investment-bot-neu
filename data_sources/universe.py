"""Load the ETF universe and enrich each option with live risk metrics.

Bridges the static metadata in `data/universe.json` (tickers, TER, ESG flag)
with the computed metrics from `market` + `risk`, so the scoring layer receives
real volatility instead of a hard-coded integer.
"""

from __future__ import annotations

import json
import os
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


def _want_live() -> bool:
    """Whether to prefer live yfinance data over the committed snapshot.

    Off by default: the public/deployed instance runs on the frozen snapshot
    (Yahoo blocks datacenter IPs, and a cold start shouldn't fan out network
    calls). Set `USE_LIVE_DATA=1` locally for fresh metrics.
    """
    return os.getenv("USE_LIVE_DATA", "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_universe(prefer_live: bool | None = None, period: str = "5y") -> tuple[list[dict], dict]:
    """Return `(options, meta)`, choosing the best available data source.

    Order: live yfinance (only if requested/enabled) → committed snapshot →
    static metadata with no metrics. `meta` carries `source` ("live" |
    "snapshot" | "static") and, for the snapshot, its `as_of` date — so the UI
    can tell the visitor which they're looking at. Never raises: a network
    failure silently falls back to the snapshot.
    """
    # Local imports keep the snapshot module optional and avoid an import cycle.
    from data_sources.snapshot import load_snapshot, snapshot_exists, snapshot_meta

    if prefer_live is None:
        prefer_live = _want_live()

    if prefer_live:
        try:
            options = build_universe(period)
            if options:
                return options, {"source": "live", "as_of": None}
        except Exception:
            pass  # fall through to the snapshot

    if snapshot_exists():
        return load_snapshot(), {"source": "snapshot", **snapshot_meta()}

    # Last resort: static metadata, no risk metrics. Scoring skips options
    # without a volatility, so this degrades rather than crashes.
    return load_universe(), {"source": "static", "as_of": None}

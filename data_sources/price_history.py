"""Committed monthly price history for offline backtesting.

Backtesting a recommended allocation needs a real price series per fund, but
yfinance is blocked from cloud hosts and a daily series would be bulky. This
persists month-end adjusted closes per universe ticker to
`data/price_history.json` (committed), so the backtest runs with no network
everywhere — in the local demo and on the deployed Space.

Regenerate locally (with a working network):

    python -m data_sources.price_history
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from data_sources.market import MarketDataError, get_prices
from data_sources.universe import load_universe

HISTORY_PATH = Path(__file__).resolve().parent.parent / "data" / "price_history.json"


def build_history(period: str = "5y") -> dict[str, dict[str, float]]:
    """Month-end adjusted closes for every universe ticker that fetches."""
    prices: dict[str, dict[str, float]] = {}
    for opt in load_universe():
        ticker = opt["ticker"]
        try:
            series = get_prices(ticker, period)
        except MarketDataError:
            continue
        monthly = series.resample("ME").last().dropna()
        prices[ticker] = {d.strftime("%Y-%m-%d"): round(float(v), 4) for d, v in monthly.items()}
    return prices


def save_history(period: str = "5y", path: Path = HISTORY_PATH) -> int:
    """Fetch and write the history. Returns the number of tickers stored."""
    payload = {
        "as_of": date.today().isoformat(),
        "period": period,
        "prices": build_history(period),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    return len(payload["prices"])


def load_history(path: Path = HISTORY_PATH) -> dict[str, dict[str, float]]:
    """Return `{ticker: {date: close}}` from the committed file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)["prices"]


def history_meta(path: Path = HISTORY_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {"as_of": data.get("as_of"), "period": data.get("period")}


def history_exists(path: Path = HISTORY_PATH) -> bool:
    return path.exists()


if __name__ == "__main__":
    n = save_history()
    print(f"price history written: {n} tickers -> {HISTORY_PATH}")

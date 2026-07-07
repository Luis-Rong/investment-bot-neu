"""Thin wrapper around yfinance with in-process caching.

Keeps the network dependency isolated here so the rest of the codebase works
with plain pandas objects and dicts.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
import yfinance as yf


class MarketDataError(RuntimeError):
    """Raised when price data cannot be retrieved for a ticker."""


@lru_cache(maxsize=256)
def get_prices(ticker: str, period: str = "5y") -> pd.Series:
    """Adjusted daily close prices for `ticker` over `period`.

    Cached per (ticker, period) for the process lifetime. Raises
    MarketDataError if no data comes back (bad ticker, network issue).
    """
    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if df is None or df.empty:
        raise MarketDataError(f"No price data returned for '{ticker}'.")

    close = df["Close"]
    # yfinance returns a single-column DataFrame for one ticker; squeeze it.
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna()


def get_prices_frame(tickers: tuple[str, ...], period: str = "5y") -> pd.DataFrame:
    """Aligned adjusted-close prices for several tickers as one DataFrame.

    `tickers` is a tuple (hashable) so callers can cache upstream. Columns that
    fail to download are skipped rather than failing the whole request.
    """
    series = {}
    for ticker in tickers:
        try:
            series[ticker] = get_prices(ticker, period)
        except MarketDataError:
            continue
    if not series:
        raise MarketDataError(f"No price data for any of: {', '.join(tickers)}")
    return pd.DataFrame(series).dropna()


@lru_cache(maxsize=256)
def get_fundamentals(ticker: str) -> dict:
    """Best-effort fundamentals from yfinance `.info` (fields vary by ETF)."""
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return {}
    keys = ("longName", "category", "totalAssets", "annualReportExpenseRatio", "currency")
    return {k: info.get(k) for k in keys if info.get(k) is not None}

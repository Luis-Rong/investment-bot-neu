"""Risk metrics computed from a price history.

All functions are pure: they take a pandas price Series (or DataFrame) and
return numbers. No network, no I/O — so they are fully unit-testable offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Trading days per year, the standard annualization factor for daily data.
TRADING_DAYS = 252


def daily_returns(prices: pd.Series) -> pd.Series:
    """Simple daily returns from a price series, NaNs dropped."""
    return prices.pct_change().dropna()


def annualized_volatility(prices: pd.Series) -> float:
    """Annualized standard deviation of daily returns (e.g. 0.18 = 18%)."""
    rets = daily_returns(prices)
    if len(rets) < 2:
        return float("nan")
    return float(rets.std(ddof=1) * np.sqrt(TRADING_DAYS))


def annualized_return(prices: pd.Series) -> float:
    """Compound annual growth rate (CAGR) over the series."""
    if len(prices) < 2:
        return float("nan")
    total_return = prices.iloc[-1] / prices.iloc[0]
    years = len(prices) / TRADING_DAYS
    if years <= 0 or total_return <= 0:
        return float("nan")
    return float(total_return ** (1 / years) - 1)


def max_drawdown(prices: pd.Series) -> float:
    """Largest peak-to-trough decline as a negative fraction (e.g. -0.34)."""
    if len(prices) < 2:
        return float("nan")
    running_max = prices.cummax()
    drawdown = prices / running_max - 1.0
    return float(drawdown.min())


def sharpe_ratio(prices: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio. `risk_free_rate` is annual (e.g. 0.03 = 3%)."""
    rets = daily_returns(prices)
    if len(rets) < 2:
        return float("nan")
    excess_daily = rets.mean() - risk_free_rate / TRADING_DAYS
    std_daily = rets.std(ddof=1)
    if std_daily == 0:
        return float("nan")
    return float((excess_daily / std_daily) * np.sqrt(TRADING_DAYS))


def correlation_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Correlation matrix of daily returns for a DataFrame of price columns."""
    return prices.pct_change().dropna().corr()


def compute_metrics(prices: pd.Series, risk_free_rate: float = 0.0) -> dict:
    """Bundle the headline metrics for one instrument into a dict."""
    return {
        "volatility": annualized_volatility(prices),
        "annual_return": annualized_return(prices),
        "max_drawdown": max_drawdown(prices),
        "sharpe": sharpe_ratio(prices, risk_free_rate),
    }

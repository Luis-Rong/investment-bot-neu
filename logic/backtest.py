"""Backtest a recommended allocation on real historical prices.

Pure: takes the allocation, the fund universe and a `{ticker: {date: close}}`
price history, and returns the portfolio's growth curve plus headline stats.
No network, no I/O — so it's deterministic and unit-testable.

The curve blends each allocated fund's price series (normalized to the first
common date, weighted by allocation), so it answers "what would this exact
recommendation have done over the sample period". Past performance only — it is
descriptive, not a forecast.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

MONTHS_PER_YEAR = 12


@dataclass
class BacktestResult:
    dates: list[str]  # aligned month-end dates
    values: list[float]  # portfolio value at each date (starts at `initial`)
    initial: float
    final: float
    total_return: float  # fraction over the whole period, e.g. 0.62
    annual_return: float  # CAGR
    volatility: float  # annualized, from monthly returns
    max_drawdown: float  # worst peak-to-trough, negative fraction
    months: int


def _weights_by_ticker(allocation, options) -> dict[str, float]:
    """Allocation entries -> `{ticker: weight fraction}` (sums to ~1)."""
    opt_by_id = {o["id"]: o for o in options}
    weights: dict[str, float] = {}
    for item in allocation:
        ticker = opt_by_id.get(item["id"], {}).get("ticker")
        if ticker:
            weights[ticker] = weights.get(ticker, 0.0) + item["percentage"] / 100.0
    return weights


def run_backtest(allocation, options, price_history, initial: float = 10000.0):
    """Blend the allocated funds into one portfolio curve over the sample.

    Funds without price history are dropped and the remaining weights
    renormalized. Returns a `BacktestResult`, or `None` if no allocated fund has
    usable history (so callers can simply skip the section).
    """
    weights = _weights_by_ticker(allocation, options)
    series = {t: price_history[t] for t, _ in weights.items() if price_history.get(t)}
    if not series:
        return None

    common = set.intersection(*(set(s.keys()) for s in series.values()))
    dates = sorted(common)
    if len(dates) < 2:
        return None

    total_w = sum(weights[t] for t in series) or 1.0
    norm = {t: weights[t] / total_w for t in series}

    start = dates[0]
    growth = []  # portfolio value relative to start (1.0 at start)
    for d in dates:
        growth.append(sum(norm[t] * (series[t][d] / series[t][start]) for t in series))
    values = [initial * g for g in growth]

    total_return = growth[-1] - 1.0
    years = (len(dates) - 1) / MONTHS_PER_YEAR
    annual_return = growth[-1] ** (1 / years) - 1 if years > 0 and growth[-1] > 0 else float("nan")

    monthly_rets = [growth[i] / growth[i - 1] - 1 for i in range(1, len(growth))]
    if len(monthly_rets) >= 2:
        mean = sum(monthly_rets) / len(monthly_rets)
        var = sum((r - mean) ** 2 for r in monthly_rets) / (len(monthly_rets) - 1)
        volatility = math.sqrt(var) * math.sqrt(MONTHS_PER_YEAR)
    else:
        volatility = float("nan")

    peak = growth[0]
    max_drawdown = 0.0
    for g in growth:
        peak = max(peak, g)
        max_drawdown = min(max_drawdown, g / peak - 1)

    return BacktestResult(
        dates=dates,
        values=values,
        initial=initial,
        final=values[-1],
        total_return=total_return,
        annual_return=annual_return,
        volatility=volatility,
        max_drawdown=max_drawdown,
        months=len(dates) - 1,
    )

"""Tests for the FastMCP server (Phase 4).

All tests are CI-safe: no network and no model download. The two tools that
hit yfinance (`get_prices`, `compute_risk_metrics`) are exercised with the
market fetch monkeypatched, and factsheet retrieval is tested via the
store-missing placeholder path.
"""

from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from data_sources.market import MarketDataError
from mcp_server import server


def _registered_tool_names() -> set[str]:
    tools = asyncio.run(server.mcp.list_tools())
    return {t.name for t in tools}


def test_all_expected_tools_registered():
    names = _registered_tool_names()
    assert {
        "get_prices",
        "compute_risk_metrics",
        "list_universe",
        "target_volatility",
        "plan_contributions",
        "retrieve_factsheet",
    } <= names


def test_list_universe_returns_static_metadata():
    universe = server.list_universe()
    assert len(universe) == 25
    assert all("ticker" in opt for opt in universe)


def test_target_volatility_within_bounds():
    assert server.target_volatility(1) == pytest.approx(0.03)
    assert server.target_volatility(10) == pytest.approx(0.25)
    assert server.target_volatility(1) < server.target_volatility(10)


def test_plan_contributions_invests_full_lump_sum():
    plan = server.plan_contributions(10000, 200)
    assert plan == {"one_time_eur": 10000, "monthly_plan_eur": 200}


def test_get_prices_happy_path(monkeypatch):
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    series = pd.Series([100.0, 101.0, 102.5, 101.5, 103.0], index=idx)
    monkeypatch.setattr(server, "_get_prices", lambda ticker, period: series)

    out = server.get_prices("ivv", period="1y", limit=2)
    assert out["ticker"] == "IVV"
    assert out["observations"] == 5
    assert out["start_date"] == "2024-01-01"
    assert out["end_date"] == "2024-01-05"
    assert out["latest_close"] == 103.0
    assert out["closes"] == [
        {"date": "2024-01-04", "close": 101.5},
        {"date": "2024-01-05", "close": 103.0},
    ]


def test_get_prices_error_is_returned_not_raised(monkeypatch):
    def boom(ticker, period):
        raise MarketDataError("no data for 'ZZZZ'")

    monkeypatch.setattr(server, "_get_prices", boom)
    out = server.get_prices("ZZZZ")
    assert "error" in out and "ZZZZ" in out["error"]


def test_compute_risk_metrics_shape(monkeypatch):
    idx = pd.date_range("2024-01-01", periods=300, freq="D")
    series = pd.Series(range(100, 400), index=idx, dtype=float)
    monkeypatch.setattr(server, "_get_prices", lambda ticker, period: series)

    out = server.compute_risk_metrics("agg")
    assert out["ticker"] == "AGG"
    assert {"volatility", "annual_return", "max_drawdown", "sharpe"} <= set(out)


def test_retrieve_factsheet_placeholder_when_store_missing(monkeypatch):
    monkeypatch.setattr(server, "store_exists", lambda: False)
    out = server.retrieve_factsheet("gold hedge")
    assert "not built" in out

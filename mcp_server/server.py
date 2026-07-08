"""FastMCP server exposing the robo-advisor's data / risk / RAG tools.

This is the Phase-4 proof point for the IBM course's MCP module (course 9): the
exact same capabilities the LangGraph agent uses internally (`agent/tools.py`)
are published here as a standardized **Model Context Protocol** service, so any
MCP client — Claude Desktop, another agent, an IDE — can call live market data
and factsheet retrieval without importing this codebase.

The tool bodies stay thin: they delegate to the same underlying data / risk /
RAG functions the agent tools wrap, keeping business logic in one place.

Run it (stdio transport, the default MCP wiring):

    python -m mcp_server.server

Or over HTTP for quick manual testing:

    python -m mcp_server.server --transport streamable-http
"""

from __future__ import annotations

import argparse
from typing import Any

from mcp.server.fastmcp import FastMCP

from data_sources.market import MarketDataError
from data_sources.market import get_prices as _get_prices
from data_sources.risk import compute_metrics
from data_sources.universe import load_universe
from logic.contributions import compute_contributions
from logic.scoring import risk_to_target_volatility
from rag.retriever import format_evidence, retrieve, store_exists

mcp = FastMCP(
    "robo-advisor",
    instructions=(
        "Tools for building investment recommendations: live ETF price history "
        "and risk metrics (yfinance), the curated ETF universe, a risk-to-target-"
        "volatility mapping, contribution planning, and grounded fund-factsheet "
        "retrieval from the vector store. Prices/metrics require network access; "
        "factsheet retrieval requires the local Chroma store to have been built "
        "(python -m rag.build_docs && python -m rag.ingest)."
    ),
)


@mcp.tool()
def get_prices(ticker: str, period: str = "5y", limit: int = 30) -> dict[str, Any]:
    """Fetch adjusted daily close prices for an ETF ticker over `period`
    (e.g. "1y", "5y", "max"). Returns a summary plus the most recent `limit`
    closes; set limit=0 for the full series. Returns {"error": ...} on a bad
    ticker or network failure instead of raising.
    """
    try:
        prices = _get_prices(ticker, period)
    except MarketDataError as exc:
        return {"error": str(exc)}

    closes = [
        {"date": ts.strftime("%Y-%m-%d"), "close": round(float(v), 4)} for ts, v in prices.items()
    ]
    tail = closes if limit <= 0 else closes[-limit:]
    return {
        "ticker": ticker.upper(),
        "period": period,
        "observations": len(closes),
        "start_date": closes[0]["date"],
        "end_date": closes[-1]["date"],
        "latest_close": closes[-1]["close"],
        "closes": tail,
    }


@mcp.tool()
def compute_risk_metrics(ticker: str, period: str = "5y") -> dict[str, Any]:
    """Fetch live price history for an ETF ticker and return its risk metrics:
    annualized volatility and return, max drawdown, and Sharpe ratio. Returns
    {"error": ...} instead of raising if the ticker can't be fetched.
    """
    try:
        prices = _get_prices(ticker, period)
    except MarketDataError as exc:
        return {"error": str(exc)}
    return {"ticker": ticker.upper(), "period": period, **compute_metrics(prices)}


@mcp.tool()
def list_universe() -> list[dict[str, Any]]:
    """List the curated ETF universe (ticker, name, asset class, TER, ESG flag,
    minimum horizon) the advisor can recommend from. Static metadata only — no
    network call; use `compute_risk_metrics` for live figures on a ticker."""
    return load_universe()


@mcp.tool()
def target_volatility(risk: int) -> float:
    """Map a 1–10 risk appetite to a target annualized volatility (3%–25%)."""
    return risk_to_target_volatility(risk)


@mcp.tool()
def plan_contributions(saving_eur: int, monthly_saving_eur: int) -> dict[str, Any]:
    """Turn a one-time lump sum and a monthly saving amount into a realistic
    contribution plan (keeps a small cash buffer on larger lump sums)."""
    return compute_contributions(saving_eur, monthly_saving_eur)


@mcp.tool()
def retrieve_factsheet(query: str, ticker: str | None = None, k: int = 2) -> str:
    """Retrieve fund-factsheet passages relevant to `query` (optionally filtered
    to a single ETF by ticker) from the vector store, formatted as numbered,
    source-tagged evidence blocks. Returns a placeholder line if the store has
    not been built yet."""
    if not store_exists():
        return (
            "Factsheet store not built. Run `python -m rag.build_docs && "
            "python -m rag.ingest` to enable grounded retrieval."
        )
    return format_evidence(retrieve(query, k=k, ticker=ticker))


def main() -> None:
    parser = argparse.ArgumentParser(description="Robo-advisor MCP server.")
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "streamable-http", "sse"],
        help="MCP transport (default: stdio).",
    )
    args = parser.parse_args()
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()

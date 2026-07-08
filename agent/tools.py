"""Agent tools: `@tool` wrappers over the data / RAG / logic layers.

Each existing capability (market metrics, factsheet retrieval, scoring +
allocation, contribution planning) is exposed as a LangChain tool so the graph
nodes call them through one uniform interface — and so the exact same tools can
be re-exported by the MCP server in Phase 4 without touching business logic.

Nodes invoke these with `.invoke({...})`; the bodies stay thin, delegating to
the underlying pure/data functions.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from data_sources.market import MarketDataError, get_prices
from data_sources.risk import compute_metrics
from logic.allocation import build_allocation
from logic.contributions import compute_contributions
from logic.scoring import risk_to_target_volatility, score_options
from rag.retriever import format_evidence, retrieve


@tool
def fetch_risk_metrics(ticker: str, period: str = "5y") -> dict[str, Any]:
    """Fetch live price history for an ETF ticker and return its risk metrics
    (annualized volatility and return, max drawdown, Sharpe ratio).

    Returns {"error": ...} instead of raising if the ticker can't be fetched,
    so callers can degrade gracefully.
    """
    try:
        prices = get_prices(ticker, period)
    except MarketDataError as exc:
        return {"error": str(exc)}
    return compute_metrics(prices)


@tool
def target_volatility(risk: int) -> float:
    """Map a 1–10 risk appetite to a target annualized volatility (3%–25%)."""
    return risk_to_target_volatility(risk)


@tool
def score_and_allocate(
    profile: dict[str, Any], options: list[dict[str, Any]], top_n: int = 3
) -> list[dict[str, Any]]:
    """Score the ETF universe against the user profile (risk, horizon, ESG) and
    return the top-`top_n` allocation as percentages summing to 100."""
    scored = score_options(options, profile)
    return build_allocation(scored, top_n=top_n)


@tool
def plan_contributions(saving_eur: int, monthly_saving_eur: int) -> dict[str, Any]:
    """Turn a one-time lump sum and a monthly saving amount into a realistic
    contribution plan (keeps a small cash buffer on larger lump sums)."""
    return compute_contributions(saving_eur, monthly_saving_eur)


@tool
def retrieve_factsheet(query: str, ticker: str | None = None, k: int = 2) -> str:
    """Retrieve fund-factsheet passages relevant to `query` (optionally filtered
    to a single ETF by ticker) from the vector store, formatted as numbered,
    source-tagged evidence blocks. Returns a placeholder line if the store is
    empty."""
    passages = retrieve(query, k=k, ticker=ticker)
    return format_evidence(passages)


# The tools the MCP server (Phase 4) will re-export.
ALL_TOOLS = [
    fetch_risk_metrics,
    target_volatility,
    score_and_allocate,
    plan_contributions,
    retrieve_factsheet,
]

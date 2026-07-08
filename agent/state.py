"""Shared state for the advisor graph.

One `AdvisorState` flows through every node. `total=False` so nodes only set
the keys they produce; downstream nodes read what earlier ones wrote.
"""

from __future__ import annotations

from typing import Any, TypedDict


class AdvisorState(TypedDict, total=False):
    # --- inputs (set by the caller before invoking the graph) ---
    messages: list[dict[str, str]]  # chat transcript [{role, content}, ...]
    options: list[dict[str, Any]]  # ETF universe enriched with live metrics
    recommendation: dict[str, Any] | None  # existing recommendation, if any

    # --- extracted / routed ---
    profile: dict[str, Any]  # parsed UserProfile as a dict
    profile_raw: str  # raw LLM extraction output (debug)
    missing: list[str]  # required profile fields still unknown
    route: str  # recommend | adjust_risk | ask_question | answer_directly
    risk_delta: int  # requested risk change (adjust_risk path)
    risk_adjustment: dict[str, Any]  # {delta, old_risk, new_risk} for the UI

    # --- portfolio building ---
    target_volatility: float  # annualized vol implied by the risk score
    allocation: list[dict[str, Any]]  # [{id, name, percentage}, ...]
    contributions: dict[str, Any]  # {one_time_eur, monthly_plan_eur}
    evidence: list[Any]  # list[RetrievedPassage] cited in the explanation
    explanation: str  # grounded client-facing text

    # --- reflection ---
    critique: dict[str, Any]  # {ok: bool, feedback: str}
    iterations: int  # reflection revisions performed so far

    # --- outputs (read by the caller after the graph returns) ---
    assistant_message: str  # what to append to the chat

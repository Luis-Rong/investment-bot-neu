"""Tests for the agent layer — deterministic helpers, tools, and graph wiring.

No LLM calls or network here: the graph is built with a stub model and only its
structure is asserted; the pure routing/reflection helpers are tested directly.
"""

from __future__ import annotations

import math

import pytest

from agent.graph import (
    allocation_matches_target,
    build_advisor_graph,
    clamp,
    detect_risk_adjustment,
    last_user_text,
    portfolio_volatility,
    reflect_selector,
    select_route,
)
from agent.tools import plan_contributions, score_and_allocate, target_volatility

# --- fixtures --------------------------------------------------------------- #

OPTIONS = [
    {
        "id": "a",
        "name": "Safe Bond ETF",
        "ticker": "AGG",
        "ter": 0.05,
        "horizon_min": 1,
        "esg": False,
        "volatility": 0.05,
    },
    {
        "id": "b",
        "name": "World Equity ETF",
        "ticker": "URTH",
        "ter": 0.20,
        "horizon_min": 3,
        "esg": False,
        "volatility": 0.16,
    },
    {
        "id": "c",
        "name": "Emerging Markets ETF",
        "ticker": "EEM",
        "ter": 0.30,
        "horizon_min": 5,
        "esg": False,
        "volatility": 0.24,
    },
]


# --- deterministic helpers -------------------------------------------------- #


def test_clamp_bounds():
    assert clamp(15, 1, 10) == 10
    assert clamp(-3, 1, 10) == 1
    assert clamp(5, 1, 10) == 5


def test_last_user_text_returns_most_recent_user_turn():
    msgs = [
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second"},
    ]
    assert last_user_text(msgs) == "second"
    assert last_user_text([]) == ""


@pytest.mark.parametrize(
    "text,expected",
    [
        ("please make it safer", -1),
        ("I want it much more aggressive", 2),
        ("can you reduce risk a lot", -2),
        ("more risk please", 1),
        ("I like the plan", None),
        ("", None),
    ],
)
def test_detect_risk_adjustment(text, expected):
    assert detect_risk_adjustment(text) == expected


def test_portfolio_volatility_is_weight_average():
    alloc = [{"id": "a", "percentage": 50}, {"id": "b", "percentage": 50}]
    pv = portfolio_volatility(alloc, OPTIONS)
    assert pv == pytest.approx((0.5 * 0.05) + (0.5 * 0.16))


def test_portfolio_volatility_ignores_missing_metrics():
    opts = [{"id": "x", "name": "n", "percentage": 100}]  # no volatility key
    alloc = [{"id": "x", "percentage": 100}]
    assert math.isnan(portfolio_volatility(alloc, opts))


def test_allocation_matches_target_within_tolerance():
    alloc = [{"id": "a", "percentage": 100}]  # vol 0.05
    assert allocation_matches_target(alloc, OPTIONS, target_vol=0.05, tol=0.06)
    assert not allocation_matches_target(alloc, OPTIONS, target_vol=0.24, tol=0.06)


def test_allocation_matches_target_unjudgeable_passes():
    alloc = [{"id": "x", "percentage": 100}]
    opts = [{"id": "x", "percentage": 100}]  # no vol -> NaN -> don't block
    assert allocation_matches_target(alloc, opts, target_vol=0.10)


# --- edge selectors --------------------------------------------------------- #


def test_select_route_reads_state():
    assert select_route({"route": "recommend"}) == "recommend"
    assert select_route({}) == "ask_question"


def test_reflect_selector_revises_once_then_finalizes():
    # not ok, no revisions yet -> revise
    assert reflect_selector({"critique": {"ok": False}, "iterations": 0}) == "revise"
    # not ok but already revised -> finalize (bounded loop)
    assert reflect_selector({"critique": {"ok": False}, "iterations": 1}) == "finalize"
    # ok -> finalize
    assert reflect_selector({"critique": {"ok": True}, "iterations": 0}) == "finalize"


# --- tools ------------------------------------------------------------------ #


def test_target_volatility_tool_maps_risk():
    assert target_volatility.invoke({"risk": 1}) == pytest.approx(0.03)
    assert target_volatility.invoke({"risk": 10}) == pytest.approx(0.25)


def test_plan_contributions_tool():
    plan = plan_contributions.invoke({"saving_eur": 10000, "monthly_saving_eur": 200})
    assert plan["monthly_plan_eur"] == 200
    assert 0 < plan["one_time_eur"] <= 10000


def test_score_and_allocate_tool_sums_to_100():
    profile = {"risk": 8, "horizon_years": 10, "esg": False}
    alloc = score_and_allocate.invoke({"profile": profile, "options": OPTIONS, "top_n": 3})
    assert sum(a["percentage"] for a in alloc) == 100
    assert all("id" in a and "name" in a for a in alloc)


# --- graph construction ----------------------------------------------------- #


def test_build_advisor_graph_compiles_with_expected_nodes():
    class _StubLLM:
        def invoke(self, *_a, **_k):  # never called in this test
            raise AssertionError("LLM should not be invoked during graph construction")

    graph = build_advisor_graph(_StubLLM())
    nodes = set(graph.get_graph().nodes)
    for expected in [
        "extract_profile",
        "route",
        "ask_question",
        "answer_directly",
        "adjust_risk",
        "prepare",
        "compute_portfolio",
        "retrieve_factsheets",
        "generate_explanation",
        "reflect",
        "finalize",
    ]:
        assert expected in nodes

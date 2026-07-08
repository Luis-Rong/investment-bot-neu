"""The advisor agent as a LangGraph `StateGraph`.

    START
      │
      ▼
   extract_profile ──► route ──┬─ ask_question ─────────────► END
                               ├─ answer_directly ──────────► END
                               ├─ adjust_risk ─┐
                               └─ recommend ───┴─► prepare ─► compute_portfolio
                                                                    │
                                                                    ▼
                                                          retrieve_factsheets
                                                                    │
                                                                    ▼
                                                          generate_explanation ◄─┐
                                                                    │            │
                                                                    ▼        (revise)
                                                                 reflect ───────┘
                                                                    │
                                                                    ▼ (grounded)
                                                                finalize ─► END

Two IBM-course features are real here: **query routing** (the `route` node picks
one of four branches, using an LLM to disambiguate the incomplete-profile case)
and a **reflection loop** (`reflect` critiques the grounding of the explanation
and loops back to regenerate it once — the Reflexion pattern).

The deterministic helpers (`detect_risk_adjustment`, `portfolio_volatility`,
`allocation_matches_target`, the edge selectors) are pure and unit-tested; the
LLM-backed nodes are built as closures over an injected chat model.
"""

from __future__ import annotations

import json
import math
from typing import Any

from langgraph.graph import END, START, StateGraph

from agent.state import AdvisorState
from agent.tools import plan_contributions, score_and_allocate
from llm.model import response_text
from llm.prompts import (
    PROFILE_EXTRACTION_SYSTEM,
    get_critique_prompt,
    get_direct_answer_prompt,
    get_final_explanation_prompt,
    get_next_question_prompt,
    get_router_prompt,
)
from logic.profile import UserProfile, missing_fields
from logic.scoring import risk_to_target_volatility
from rag.retriever import format_evidence, retrieve, store_exists

MAX_REFLECTIONS = 1  # at most one explanation revision, to bound the loop
VOL_TOLERANCE = 0.06  # portfolio vol may drift this far from target before we flag it

_FIELD_QUESTIONS = {
    "goal": "What are you investing for (e.g., retirement, emergency fund, wealth building)?",
    "horizon_years": "How long do you want to invest for (in years)?",
    "risk": "On a scale from 1 to 10, how comfortable are you with risk?",
    "esg": "Do you prefer sustainable/ESG investments (yes or no)?",
    "saving_eur": "How much would you like to invest as a one-time amount today (in EUR)? If none, say 0.",
    "monthly_saving_eur": "How much would you like to invest each month via a savings plan (in EUR)? If none, say 0.",
}


# --------------------------------------------------------------------------- #
# Deterministic helpers (pure, unit-tested — no LLM, no network)
# --------------------------------------------------------------------------- #
def clamp(value: int, min_v: int, max_v: int) -> int:
    return max(min_v, min(max_v, value))


def last_user_text(messages: list[dict[str, str]]) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def detect_risk_adjustment(user_text: str):
    """Return a signed magnitude if the user asked to change risk, else None."""
    t = (user_text or "").strip().lower()
    if not t:
        return None

    less_keywords = [
        "less risky",
        "lower risk",
        "more conservative",
        "safer",
        "reduce risk",
        "less risk",
    ]
    more_keywords = [
        "more risky",
        "higher risk",
        "more aggressive",
        "riskier",
        "increase risk",
        "more risk",
    ]

    wants_less = any(k in t for k in less_keywords)
    wants_more = any(k in t for k in more_keywords)
    if not wants_less and not wants_more:
        return None

    strong = any(k in t for k in ["much", "a lot", "significantly", "way", "far"])
    magnitude = 2 if strong else 1
    return -magnitude if wants_less else magnitude


def portfolio_volatility(allocation: list[dict], options: list[dict]) -> float:
    """Weight-averaged annualized volatility of the allocated funds.

    A first-order proxy (ignores correlations) used only as a reflection sanity
    check. Returns NaN when no allocated fund has a usable metric.
    """
    by_id = {o["id"]: o for o in options}
    weighted = 0.0
    covered = 0.0
    for item in allocation:
        opt = by_id.get(item["id"], {})
        vol = opt.get("volatility")
        if vol is None or (isinstance(vol, float) and math.isnan(vol)):
            continue
        w = item["percentage"] / 100.0
        weighted += w * vol
        covered += w
    return weighted / covered if covered else float("nan")


def allocation_matches_target(
    allocation: list[dict], options: list[dict], target_vol: float, tol: float = VOL_TOLERANCE
) -> bool:
    """True if the portfolio's volatility is within `tol` of the risk target."""
    pv = portfolio_volatility(allocation, options)
    if math.isnan(pv):
        return True  # can't judge -> don't block
    return abs(pv - target_vol) <= tol


def select_route(state: AdvisorState) -> str:
    return state.get("route", "ask_question")


def reflect_selector(state: AdvisorState) -> str:
    critique = state.get("critique", {})
    iterations = state.get("iterations", 0)
    if not critique.get("ok", True) and iterations < MAX_REFLECTIONS:
        return "revise"
    return "finalize"


def _parse_json_object(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return {}
    return {}


def _history(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def extract_profile(llm, messages: list[dict[str, str]]) -> dict:
    """Run the profile-extraction LLM call over a chat transcript.

    Module-level so the evaluation harness (Phase 5) can exercise the *same*
    extraction path the graph uses in production, not a copy. Returns the parsed
    profile dict, the raw model output (for debugging), and the still-missing
    required fields.
    """
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    resp = llm.invoke(
        [
            {"role": "system", "content": PROFILE_EXTRACTION_SYSTEM},
            {"role": "user", "content": transcript},
        ]
    )
    raw = response_text(resp)
    data = _parse_json_object(raw)
    profile = UserProfile(**data) if data else UserProfile()
    return {
        "profile": profile.model_dump(),
        "profile_raw": raw,
        "missing": missing_fields(profile),
    }


# --------------------------------------------------------------------------- #
# Graph factory: nodes are closures over the injected LLM
# --------------------------------------------------------------------------- #
def build_advisor_graph(llm):
    """Compile and return the advisor `StateGraph` bound to `llm`."""

    # ---- extraction + routing ------------------------------------------- #
    def extract_profile_node(state: AdvisorState) -> dict:
        return extract_profile(llm, state["messages"])

    def route_node(state: AdvisorState) -> dict:
        missing = state.get("missing", [])
        rec = state.get("recommendation")
        last = last_user_text(state["messages"])

        # Deterministic override: an explicit risk tweak on an existing rec.
        delta = detect_risk_adjustment(last)
        if rec and delta is not None:
            return {"route": "adjust_risk", "risk_delta": delta}

        if not missing:
            return {"route": "recommend"}

        # Incomplete profile: let the LLM decide ask-next vs answer-first.
        prompt = get_router_prompt()
        resp = llm.invoke(
            prompt.format(missing_fields=", ".join(missing), history=_history(state["messages"]))
        )
        choice = response_text(resp).upper()
        return {"route": "answer_directly" if choice.startswith("ANSWER") else "ask_question"}

    def ask_question_node(state: AdvisorState) -> dict:
        missing = state.get("missing", [])
        prompt = get_next_question_prompt()
        resp = llm.invoke(
            prompt.format(
                history=_history(state["messages"]),
                missing_fields=", ".join(missing) if missing else "none",
            )
        )
        text = response_text(resp)
        # Safety net: model shouldn't say READY while fields are missing.
        if text == "READY_FOR_RECOMMENDATION" and missing:
            text = _FIELD_QUESTIONS.get(
                missing[0], "Could you share one more detail so I can personalize this?"
            )
        return {"assistant_message": text}

    def answer_directly_node(state: AdvisorState) -> dict:
        prompt = get_direct_answer_prompt()
        resp = llm.invoke(prompt.format(history=_history(state["messages"])))
        return {"assistant_message": response_text(resp)}

    def adjust_risk_node(state: AdvisorState) -> dict:
        rec = state.get("recommendation") or {}
        profile = dict(rec.get("profile") or state.get("profile") or {})
        delta = state.get("risk_delta", 0)
        old_risk = int(profile.get("risk", 5) or 5)
        new_risk = clamp(old_risk + delta, 1, 10)
        profile["risk"] = new_risk
        return {
            "profile": profile,
            "risk_adjustment": {"delta": delta, "old_risk": old_risk, "new_risk": new_risk},
        }

    # ---- portfolio building --------------------------------------------- #
    def prepare_node(state: AdvisorState) -> dict:
        profile = state["profile"]
        # Keep only options that actually carry live metrics (scoring needs them).
        usable = [
            o
            for o in state["options"]
            if o.get("volatility") is not None
            and not (isinstance(o.get("volatility"), float) and math.isnan(o["volatility"]))
        ]
        return {
            "options": usable,
            "target_volatility": risk_to_target_volatility(profile.get("risk", 5)),
        }

    def compute_portfolio_node(state: AdvisorState) -> dict:
        profile = state["profile"]
        allocation = score_and_allocate.invoke(
            {"profile": profile, "options": state["options"], "top_n": 3}
        )
        contrib = plan_contributions.invoke(
            {
                "saving_eur": profile.get("saving_eur", 0) or 0,
                "monthly_saving_eur": profile.get("monthly_saving_eur", 0) or 0,
            }
        )
        return {"allocation": allocation, "contributions": contrib}

    def retrieve_factsheets_node(state: AdvisorState) -> dict:
        if not store_exists():
            return {"evidence": []}
        profile = state["profile"]
        opt_by_id = {o["id"]: o for o in state["options"]}
        goal = profile.get("goal") or "long-term investing"
        passages: list[Any] = []
        for item in state["allocation"]:
            opt = opt_by_id.get(item["id"], {})
            ticker = opt.get("ticker")
            if not ticker:
                continue
            query = f"{opt.get('name', '')} risk profile costs suitability for {goal}"
            passages.extend(retrieve(query, k=1, ticker=ticker))
        return {"evidence": passages}

    def generate_explanation_node(state: AdvisorState) -> dict:
        profile = state["profile"]
        allocation = state["allocation"]
        contrib = state["contributions"]
        evidence = state.get("evidence", [])
        critique = state.get("critique")

        portfolio_lines = "\n".join(f"- {x['percentage']}% {x['name']}" for x in allocation)
        llm_input = f"""
User profile (context): {profile}

Portfolio proposal:
{portfolio_lines}

Contribution plan:
- One-time investment today: {contrib["one_time_eur"]} EUR
- Monthly savings plan: {contrib["monthly_plan_eur"]} EUR

EVIDENCE (fund-profile passages — cite these by number for any fund claim):
{format_evidence(evidence)}
"""
        # On a reflection revision, feed the critique back in (Reflexion).
        if critique and critique.get("feedback"):
            llm_input += (
                "\nA reviewer flagged the previous draft. Fix these issues:\n"
                f"{critique['feedback']}\n"
            )

        prompt = get_final_explanation_prompt()
        text = response_text(llm.invoke(prompt.format(input=llm_input)))

        iterations = state.get("iterations", 0) + (1 if critique else 0)
        return {"explanation": text, "iterations": iterations}

    def reflect_node(state: AdvisorState) -> dict:
        evidence = state.get("evidence", [])
        explanation = state["explanation"]

        # 1) Deterministic: does the portfolio's risk match the target?
        alloc_ok = allocation_matches_target(
            state["allocation"], state["options"], state.get("target_volatility", 0.0)
        )
        notes = []
        if not alloc_ok:
            pv = portfolio_volatility(state["allocation"], state["options"])
            notes.append(
                f"The portfolio's volatility ({pv:.0%}) drifts from the target "
                f"({state.get('target_volatility', 0.0):.0%}); acknowledge the risk level honestly."
            )

        # 2) LLM: is the explanation actually grounded in the evidence?
        llm_ok = True
        if evidence:  # only worth checking when there is evidence to cite
            crit_input = f"EVIDENCE:\n{format_evidence(evidence)}\n\nDRAFT:\n{explanation}"
            resp = llm.invoke(get_critique_prompt().format(input=crit_input))
            verdict = _parse_json_object(response_text(resp))
            llm_ok = bool(verdict.get("ok", True))
            if not llm_ok and verdict.get("feedback"):
                notes.append(str(verdict["feedback"]))

        ok = alloc_ok and llm_ok
        return {"critique": {"ok": ok, "feedback": " ".join(notes)}}

    def finalize_node(state: AdvisorState) -> dict:
        recommendation = {
            "profile": state["profile"],
            "allocation": state["allocation"],
            "contributions": state["contributions"],
            "explanation": state["explanation"],
            "sources": state.get("evidence", []),
            "missing_fields_at_finish": [],
        }
        adj = state.get("risk_adjustment")
        if adj:
            recommendation["risk_adjustment"] = adj
            message = state["explanation"]
        else:
            message = "I've prepared a personalized recommendation for you — see the summary below."
        return {"recommendation": recommendation, "assistant_message": message}

    # ---- wiring --------------------------------------------------------- #
    builder = StateGraph(AdvisorState)
    builder.add_node("extract_profile", extract_profile_node)
    builder.add_node("route", route_node)
    builder.add_node("ask_question", ask_question_node)
    builder.add_node("answer_directly", answer_directly_node)
    builder.add_node("adjust_risk", adjust_risk_node)
    builder.add_node("prepare", prepare_node)
    builder.add_node("compute_portfolio", compute_portfolio_node)
    builder.add_node("retrieve_factsheets", retrieve_factsheets_node)
    builder.add_node("generate_explanation", generate_explanation_node)
    builder.add_node("reflect", reflect_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "extract_profile")
    builder.add_edge("extract_profile", "route")
    builder.add_conditional_edges(
        "route",
        select_route,
        {
            "recommend": "prepare",
            "adjust_risk": "adjust_risk",
            "ask_question": "ask_question",
            "answer_directly": "answer_directly",
        },
    )
    builder.add_edge("ask_question", END)
    builder.add_edge("answer_directly", END)
    builder.add_edge("adjust_risk", "prepare")
    builder.add_edge("prepare", "compute_portfolio")
    builder.add_edge("compute_portfolio", "retrieve_factsheets")
    builder.add_edge("retrieve_factsheets", "generate_explanation")
    builder.add_edge("generate_explanation", "reflect")
    builder.add_conditional_edges(
        "reflect",
        reflect_selector,
        {"revise": "generate_explanation", "finalize": "finalize"},
    )
    builder.add_edge("finalize", END)

    return builder.compile()

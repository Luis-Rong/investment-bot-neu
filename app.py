import json

import streamlit as st
from dotenv import load_dotenv

from data_sources.universe import build_universe
from llm.model import get_llm
from llm.prompts import get_final_explanation_prompt, get_next_question_prompt
from logic.allocation import build_allocation
from logic.contributions import compute_contributions
from logic.profile import UserProfile, missing_fields
from logic.scoring import score_options
from ui.chat import add_message, get_user_input, init_chat_state, render_messages
from ui.recommendation import render_recommendation_ui


def clamp(value: int, min_v: int, max_v: int) -> int:
    return max(min_v, min(max_v, value))


def detect_risk_adjustment(user_text: str):
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

    slight = any(k in t for k in ["slightly", "a bit", "a little", "bit"])
    strong = any(k in t for k in ["much", "a lot", "significantly", "way", "far"])

    if strong:
        magnitude = 2
    elif slight:
        magnitude = 1
    else:
        magnitude = 1

    return -magnitude if wants_less else magnitude


def chat_history_as_langchain_messages(messages):
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def generate_recommendation_text(llm, user_dict, allocation, contributions):
    portfolio_lines = "\n".join([f"- {x['percentage']}% {x['name']}" for x in allocation])

    llm_input = f"""
User profile (context): {user_dict}

Portfolio proposal:
{portfolio_lines}

Contribution plan:
- One-time investment today: {contributions["one_time_eur"]} EUR
- Monthly savings plan: {contributions["monthly_plan_eur"]} EUR

Write a client-friendly recommendation in English:
- Mirror the user's goal and time horizon first.
- Explain the portfolio in plain language.
- Mention ESG if esg=true.
- Explain the one-time investment and the monthly plan clearly.
- If one_time_eur is 0, focus on the monthly plan.
- If monthly_plan_eur is 0, focus on the one-time investment.
Max 6 sentences.
"""
    final_prompt = get_final_explanation_prompt()
    return llm.invoke(final_prompt.format(input=llm_input)).content.strip()


def extract_profile(llm, messages):
    extract_system = """
Extract a user profile from the chat transcript as JSON.
Return ONLY JSON, no explanation, no markdown.

Fields:
goal (string or null)
horizon_years (int or null)
risk (int 1-10 or null)
esg (true/false or null)
saving_eur (int or null)
monthly_saving_eur (int or null)
experience (low/medium/high or null)
liquidity_need (low/medium/high or null)

If the user is unclear or you are not sure, use null.
If the user answers "yes/no" for ESG, map to true/false.
"""

    transcript = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

    resp = llm.invoke(
        [{"role": "system", "content": extract_system}, {"role": "user", "content": transcript}]
    )

    text = resp.content.strip()
    st.session_state.last_profile_extraction_raw = text

    try:
        data = json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(text[start : end + 1])
        else:
            data = {}

    return UserProfile(**data)


def ensure_initial_message():
    if len(st.session_state.messages) == 0:
        add_message(
            "assistant",
            "Hi! I am your robo-advisor prototype. What are you investing for (e.g., retirement, emergency fund, long-term wealth building)?",
        )


load_dotenv()

st.set_page_config(page_title="Robo-Advisor Chatbot", layout="centered")
st.title("Robo-Advisor Chatbot Prototype")


@st.cache_data(ttl=3600, show_spinner="Loading live market data...")
def load_investment_universe():
    """ETF universe enriched with live risk metrics. Cached for an hour."""
    return build_universe()


options = load_investment_universe()

init_chat_state()
ensure_initial_message()
render_messages()

llm = get_llm()

user_text = get_user_input()
if user_text:
    add_message("user", user_text)

    # If we already have a recommendation, allow risk adjustments in both directions
    if st.session_state.recommendation:
        delta = detect_risk_adjustment(user_text)
        if delta is not None:
            rec = st.session_state.recommendation
            user_dict = rec["profile"].copy()

            old_risk = int(user_dict.get("risk", 5))
            new_risk = clamp(old_risk + delta, 1, 10)
            user_dict["risk"] = new_risk

            scored = score_options(options, user_dict)
            allocation = build_allocation(scored, top_n=3)

            contrib = compute_contributions(
                user_dict.get("saving_eur", 0), user_dict.get("monthly_saving_eur", 0)
            )

            explanation = generate_recommendation_text(
                llm=llm, user_dict=user_dict, allocation=allocation, contributions=contrib
            )

            add_message("assistant", explanation)

            st.session_state.recommendation = {
                "profile": user_dict,
                "allocation": allocation,
                "contributions": contrib,
                "explanation": explanation,
                "missing_fields_at_finish": [],
                "risk_adjustment": {"delta": delta, "old_risk": old_risk, "new_risk": new_risk},
            }

            st.rerun()

    # Normal flow: extract profile and continue questions
    profile = extract_profile(llm, st.session_state.messages)
    st.session_state.profile = profile

    missing = missing_fields(profile)

    next_prompt = get_next_question_prompt()
    history = chat_history_as_langchain_messages(st.session_state.messages)

    bot_resp = llm.invoke(
        next_prompt.format(
            history=history, missing_fields=", ".join(missing) if missing else "none"
        )
    )
    bot_text = bot_resp.content.strip()

    # If model says READY too early, force a specific missing field question
    if bot_text == "READY_FOR_RECOMMENDATION" and len(missing) > 0:
        field = missing[0]
        mapping = {
            "goal": "What are you investing for (e.g., retirement, emergency fund, wealth building)?",
            "horizon_years": "How long do you want to invest for (in years)?",
            "risk": "On a scale from 1 to 10, how comfortable are you with risk?",
            "esg": "Do you prefer sustainable/ESG investments (yes or no)?",
            "saving_eur": "How much money do you want to invest as a one-time amount today (in EUR)? If none, say 0.",
            "monthly_saving_eur": "How much would you like to invest each month via a saving plan (in EUR)? If none, say 0.",
        }
        bot_text = mapping.get(
            field, "Could you share a quick detail so I can personalize the recommendation?"
        )

    if bot_text != "READY_FOR_RECOMMENDATION":
        add_message("assistant", bot_text)
        st.rerun()

    # READY and complete -> compute recommendation
    if bot_text == "READY_FOR_RECOMMENDATION" and len(missing) == 0:
        user_dict = profile.model_dump()

        scored = score_options(options, user_dict)
        allocation = build_allocation(scored, top_n=3)

        contrib = compute_contributions(
            user_dict.get("saving_eur", 0), user_dict.get("monthly_saving_eur", 0)
        )

        explanation = generate_recommendation_text(
            llm=llm, user_dict=user_dict, allocation=allocation, contributions=contrib
        )

        add_message(
            "assistant",
            "I have prepared a personalized recommendation for you. See the summary below.",
        )

        st.session_state.recommendation = {
            "profile": user_dict,
            "allocation": allocation,
            "contributions": contrib,
            "explanation": explanation,
            "missing_fields_at_finish": [],
        }

        st.rerun()


# Render recommendation UI + debug
if st.session_state.recommendation:
    rec = st.session_state.recommendation

    render_recommendation_ui(
        profile=rec["profile"],
        allocation=rec["allocation"],
        contributions=rec["contributions"],
        explanation_text=rec["explanation"],
    )

    with st.expander("Developer view (debug)"):
        st.write("Profile (structured):")
        st.json(rec["profile"])

        st.write("Portfolio allocation (structured):")
        st.json(rec["allocation"])

        st.write("Contribution plan:")
        st.json(rec["contributions"])

        st.write("Missing fields at finish (should be empty):")
        st.write(rec.get("missing_fields_at_finish"))

        risk_adj = rec.get("risk_adjustment")
        if risk_adj:
            st.write("Risk adjustment:")
            st.json(risk_adj)

        raw = st.session_state.get("last_profile_extraction_raw")
        if raw:
            st.write("Last profile extraction (raw model output):")
            st.code(raw)

import streamlit as st
from dotenv import load_dotenv

from agent.graph import build_advisor_graph
from data_sources.universe import build_universe
from llm.model import get_llm
from rag.retriever import store_exists
from ui.chat import add_message, get_user_input, init_chat_state, render_messages
from ui.recommendation import render_recommendation_ui
from ui.theme import inject_base_css, render_app_header, render_sidebar


def ensure_initial_message():
    if len(st.session_state.messages) == 0:
        add_message(
            "assistant",
            "Hi, I'm your investment advisor. Let's build a portfolio that fits you. "
            "What are you investing for — retirement, a big purchase, long-term wealth?",
        )


load_dotenv()

st.set_page_config(
    page_title="Meridian — AI Portfolio Advisor",
    page_icon="◆",
    layout="centered",
    initial_sidebar_state="expanded",
)

inject_base_css()


@st.cache_data(ttl=3600, show_spinner="Loading live market data...")
def load_investment_universe():
    """ETF universe enriched with live risk metrics. Cached for an hour."""
    return build_universe()


@st.cache_resource
def get_advisor_graph():
    """Compiled LangGraph advisor (routing + reflection). Built once per session."""
    return build_advisor_graph(get_llm())


options = load_investment_universe()
advisor = get_advisor_graph()

render_sidebar(n_funds=len(options), grounded=store_exists())
render_app_header()

init_chat_state()
ensure_initial_message()
render_messages()

user_text = get_user_input()
if user_text:
    add_message("user", user_text)

    with st.spinner("Thinking..."):
        result = advisor.invoke(
            {
                "messages": st.session_state.messages,
                "options": options,
                "recommendation": st.session_state.recommendation,
            }
        )

    # Debug: keep the raw profile extraction around for the developer view.
    if result.get("profile_raw"):
        st.session_state.last_profile_extraction_raw = result["profile_raw"]

    if result.get("recommendation"):
        st.session_state.recommendation = result["recommendation"]
    if result.get("profile"):
        st.session_state.profile = result["profile"]

    add_message("assistant", result.get("assistant_message", "..."))
    st.rerun()


# Render recommendation UI + debug
if st.session_state.recommendation:
    rec = st.session_state.recommendation

    render_recommendation_ui(
        profile=rec["profile"],
        allocation=rec["allocation"],
        contributions=rec["contributions"],
        explanation_text=rec["explanation"],
        options=options,
        sources=rec.get("sources", []),
    )

    with st.expander("Developer view (debug)"):
        st.write("Profile (structured):")
        st.json(rec["profile"])

        st.write("Portfolio allocation (structured):")
        st.json(rec["allocation"])

        st.write("Contribution plan:")
        st.json(rec["contributions"])

        risk_adj = rec.get("risk_adjustment")
        if risk_adj:
            st.write("Risk adjustment:")
            st.json(risk_adj)

        raw = st.session_state.get("last_profile_extraction_raw")
        if raw:
            st.write("Last profile extraction (raw model output):")
            st.code(raw)

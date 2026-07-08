import streamlit as st
from dotenv import load_dotenv

from agent.graph import build_advisor_graph
from data_sources.universe import resolve_universe
from llm.model import get_llm, has_provider_key
from ui.chat import add_message, get_user_input, init_chat_state, render_messages
from ui.demo import demo_exists, load_demo_recommendation
from ui.recommendation import render_recommendation_ui
from ui.theme import inject_base_css, render_app_header, render_byok, render_sidebar


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


@st.cache_data(ttl=3600, show_spinner="Loading market data...")
def load_investment_universe():
    """ETF universe (live metrics, or the committed snapshot). Cached for an hour."""
    return resolve_universe()


@st.cache_resource
def ensure_rag_store() -> bool:
    """Build the Chroma store from committed factsheets on first boot if missing."""
    from rag.ingest import build_store_if_missing

    return build_store_if_missing()


options, uni_meta = load_investment_universe()
grounded = ensure_rag_store()

# --- Resolve which LLM key to use: visitor's own (BYOK) > owner's env key ---
byok_model, byok_key = render_byok()
if byok_key:
    active_model, active_key, have_key = byok_model, byok_key, True
elif has_provider_key():
    active_model, active_key, have_key = None, None, True  # owner key from env/secrets
else:
    active_model, active_key, have_key = None, None, False

render_sidebar(
    n_funds=len(options),
    grounded=grounded,
    data_source=uni_meta.get("source", "live"),
    as_of=uni_meta.get("as_of"),
)
render_app_header()

init_chat_state()
ensure_initial_message()
render_messages()

user_text = get_user_input()
if user_text:
    add_message("user", user_text)

    if not have_key:
        add_message(
            "assistant",
            "To chat live I need an API key. Open **🔑 Use your own API key** in the "
            "sidebar and paste one — meanwhile the example recommendation below shows "
            "what a full result looks like.",
        )
        st.rerun()

    advisor = build_advisor_graph(get_llm(model=active_model, api_key=active_key))
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


# --- Recommendation panel: the live result, or the committed demo example ---
rec = st.session_state.recommendation
showing_demo = rec is None and demo_exists()
if showing_demo:
    st.info(
        "👀 **Example recommendation** — this is a saved sample so you can see the "
        "output without a key. Add your own key in the sidebar and start chatting to "
        "get one tailored to you."
    )
    rec = load_demo_recommendation()

if rec:
    render_recommendation_ui(
        profile=rec["profile"],
        allocation=rec["allocation"],
        contributions=rec["contributions"],
        explanation_text=rec["explanation"],
        options=options,
        sources=rec.get("sources", []),
    )

# Developer view only for real (live) recommendations.
if st.session_state.recommendation:
    rec = st.session_state.recommendation
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

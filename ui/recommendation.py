import streamlit as st


def risk_to_label(risk: int) -> str:
    if risk <= 3:
        return "Conservative"
    if risk <= 6:
        return "Moderate"
    return "Growth"


def render_recommendation_ui(
    profile: dict, allocation: list, contributions: dict, explanation_text: str
):
    st.subheader("Your personalized recommendation")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("One-time investment today", f"{contributions.get('one_time_eur', 0)} EUR")
    with c2:
        st.metric("Monthly savings plan", f"{contributions.get('monthly_plan_eur', 0)} EUR")
    with c3:
        r = profile.get("risk")
        st.metric("Risk style", risk_to_label(int(r)) if r is not None else "N/A")

    st.write(explanation_text)

    st.subheader("Suggested portfolio")
    for item in allocation:
        st.write(f"- {item['percentage']}% {item['name']}")

    st.subheader("Next steps")
    st.write(
        "1. If you want, you can change the risk level: 'make it slightly less risky' or 'make it more risky'."
    )
    st.write(
        "2. If the one-time amount feels too high, tell me a comfortable number and I will adjust it."
    )

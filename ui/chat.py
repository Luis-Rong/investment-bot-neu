import streamlit as st


def init_chat_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "profile" not in st.session_state:
        st.session_state.profile = None
    if "recommendation" not in st.session_state:
        st.session_state.recommendation = None


def render_messages():
    for msg in st.session_state.messages:
        avatar = "ui/assets/user.png" if msg["role"] == "user" else "ui/assets/bot.png"
        with st.chat_message(msg["role"], avatar=avatar):
            st.write(msg["content"])


def add_message(role: str, content: str):
    st.session_state.messages.append({"role": role, "content": content})


def get_user_input():
    return st.chat_input("Schreibe hier...")

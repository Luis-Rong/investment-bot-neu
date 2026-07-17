"""Gradio entry point — the same advisor, packaged for Hugging Face Spaces.

The Streamlit app (`app.py`) is the primary UI; this wraps the identical
LangGraph advisor in Gradio because HF's free tier hosts Gradio Spaces (the
Docker SDK moved to paid in mid-2026). It also satisfies the IBM course's
literal deliverable format.

Same deploy story as the Streamlit app: committed data snapshot (no live
market calls), vector store rebuilt from committed factsheets on first boot,
and bring-your-own-key so the public instance ships no shared LLM key.

Run locally:  python app_gradio.py
"""

from __future__ import annotations

import gradio as gr
import spaces
from dotenv import load_dotenv

from agent.graph import build_advisor_graph
from data_sources.universe import resolve_universe
from llm.model import BYOK_MODEL_CHOICES, get_llm, has_provider_key
from ui.formatting import backtest_frame, format_recommendation_md

load_dotenv()

# A calm, branded look so the Space doesn't ship the bare default Gradio theme.
THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#f7f7f5",
    block_background_fill="#ffffff",
    block_border_width="1px",
    block_shadow="0 1px 2px rgba(11,11,11,0.05)",
    block_radius="12px",
)

CSS = """
.gradio-container {max-width: 920px !important; margin: 0 auto;}
#mer-header h1 {font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.1rem;}
#mer-header p {color: #52514e; margin-top: 0;}
footer {display: none !important;}
"""


@spaces.GPU(duration=5)
def _zerogpu_startup_probe():
    """No-op — this app has no CUDA workload.

    HF's free Gradio-Space tier currently assigns ZeroGPU hardware by default
    and refuses to start ("No @spaces.GPU function detected during startup")
    unless at least one function wired into the app carries the decorator.
    Wiring this into `demo.load()` satisfies that check. Outside a ZeroGPU
    Space (local dev, other hosts) `spaces.GPU` is a no-op passthrough — see
    `spaces.zero.decorator._GPU`, which returns the function unmodified unless
    the `SPACES_ZERO_GPU` env var is set — so this never touches a GPU there.
    """
    return None


OPTIONS, UNI_META = resolve_universe()

NO_KEY_MESSAGE = (
    "To chat live I need an LLM API key. Open **Settings** below the chat and "
    "paste your own key (Google / Anthropic / OpenAI) — it stays in this "
    "session and is sent only to the provider you pick."
)

WELCOME = (
    "Hi, I'm your investment advisor. Let's build a portfolio that fits you. "
    "What are you investing for — retirement, a big purchase, long-term wealth?"
)

_default_graph = None


def _get_graph(model_label: str, api_key: str):
    """Advisor graph for this request: BYOK builds fresh, else a cached default."""
    global _default_graph
    api_key = (api_key or "").strip()
    if api_key:
        model = dict(BYOK_MODEL_CHOICES)[model_label]
        return build_advisor_graph(get_llm(model=model, api_key=api_key))
    if _default_graph is None:
        _default_graph = build_advisor_graph(get_llm())
    return _default_graph


def respond(message, chat_history, recommendation, model_label, api_key):
    """One conversational turn through the advisor graph."""
    message = (message or "").strip()
    if not message:
        return "", chat_history, recommendation, gr.update(), gr.update()

    chat_history = list(chat_history or [])
    chat_history.append({"role": "user", "content": message})

    if not (api_key or "").strip() and not has_provider_key():
        chat_history.append({"role": "assistant", "content": NO_KEY_MESSAGE})
        return "", chat_history, recommendation, gr.update(), gr.update()

    # The graph reads plain {role, content} dicts; skip the seeded welcome line.
    messages = [m for m in chat_history if m["content"] != WELCOME]
    try:
        result = _get_graph(model_label, api_key).invoke(
            {"messages": messages, "options": OPTIONS, "recommendation": recommendation}
        )
    except Exception as exc:  # surface provider errors (bad key, quota) in-chat
        chat_history.append(
            {"role": "assistant", "content": f"That didn't work: `{exc}` — check your key/model."}
        )
        return "", chat_history, recommendation, gr.update(), gr.update()

    chat_history.append({"role": "assistant", "content": result.get("assistant_message", "...")})
    if result.get("recommendation"):
        recommendation = result["recommendation"]

    if recommendation:
        panel = gr.update(value=format_recommendation_md(recommendation, OPTIONS), visible=True)
        frame = backtest_frame(recommendation, OPTIONS)
        plot = gr.update(value=frame, visible=True) if frame is not None else gr.update()
    else:
        panel = gr.update()
        plot = gr.update()
    return "", chat_history, recommendation, panel, plot


def build_demo() -> gr.Blocks:
    # NB: `theme`/`css` are applied in `launch()` below — Gradio 6.0 moved them
    # off the Blocks constructor (passing them here is silently ignored).
    with gr.Blocks(title="Meridian — AI Portfolio Advisor") as demo:
        gr.Markdown(
            "# Meridian — AI portfolio advisor\n"
            "Chat about your goal and risk comfort; get an ETF portfolio grounded "
            "in real fund data, with a historical backtest. *Prototype — not "
            "financial advice.*",
            elem_id="mer-header",
        )

        recommendation = gr.State(None)
        chatbot = gr.Chatbot(
            value=[{"role": "assistant", "content": WELCOME}],
            height=420,
            show_label=False,
        )
        msg = gr.Textbox(
            placeholder="Tell me about your investment goals...", show_label=False, submit_btn=True
        )

        with gr.Accordion("Settings — use your own API key", open=False):
            gr.Markdown(
                "The public demo ships no shared key. Paste your own to chat live — "
                "it stays in your session and goes only to the provider you select."
            )
            model_label = gr.Dropdown(
                choices=[label for label, _ in BYOK_MODEL_CHOICES],
                value=BYOK_MODEL_CHOICES[0][0],
                label="Model",
            )
            api_key = gr.Textbox(label="API key", type="password", placeholder="AIza… / sk-…")

        panel = gr.Markdown(visible=False)
        plot = gr.LinePlot(
            x="date",
            y="value",
            title="How this mix would have performed",
            x_title="",
            y_title="Portfolio value (€)",
            height=260,
            visible=False,
        )

        msg.submit(
            respond,
            [msg, chatbot, recommendation, model_label, api_key],
            [msg, chatbot, recommendation, panel, plot],
        )
        demo.load(_zerogpu_startup_probe, inputs=None, outputs=None)
    return demo


if __name__ == "__main__":
    build_demo().launch(theme=THEME, css=CSS)

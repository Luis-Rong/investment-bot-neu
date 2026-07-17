"""Gradio entry point — the same advisor, packaged for Hugging Face Spaces.

The Streamlit app (`app.py`) is the primary UI; this wraps the identical
LangGraph advisor in Gradio because HF's free tier hosts Gradio Spaces (the
Docker SDK moved to paid in mid-2026). It also satisfies the IBM course's
literal deliverable format.

The layout mirrors the Streamlit app: a left sidebar (branding, data/citation
badges, how-it-works, "start over", and the bring-your-own-key controls) and a
main column with the chat, a stat-tile + allocation-bar plan panel, a backtest
chart, and the rationale/sources. The shared `mer-*` CSS classes are the same
ones the Streamlit theme uses, so both apps read as one product.

Same deploy story as the Streamlit app: committed data snapshot (no live market
calls), vector store rebuilt from committed factsheets on first boot, and
bring-your-own-key so the public instance ships no shared LLM key.

Run locally:  python app_gradio.py
"""

from __future__ import annotations

import gradio as gr
import spaces
from dotenv import load_dotenv

from agent.graph import build_advisor_graph
from data_sources.universe import resolve_universe
from llm.model import BYOK_MODEL_CHOICES, get_llm, has_provider_key
from ui.demo import demo_exists, load_demo_recommendation
from ui.formatting import backtest_frame, format_explanation_md, format_plan_html

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

# The mer-* rules are copied from the Streamlit theme (ui/theme.py) so the plan
# panel — stat tiles and the colored allocation bar — looks identical here.
CSS = """
.gradio-container {max-width: 1080px !important; margin: 0 auto;}
#mer-header h1 {font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.1rem;}
#mer-header p {color: #52514e; margin-top: 0;}
footer {display: none !important;}

.mer-side-title {font-weight: 700; font-size: 1.15rem; color: #0b0b0b; margin-bottom: 0.1rem;}
.mer-side-sub {color: #52514e; font-size: 0.85rem; margin-bottom: 0.7rem;}
.mer-badges {display: flex; flex-direction: column; align-items: flex-start; gap: 0.4rem;
    margin-bottom: 0.4rem;}
.mer-badge {font-size: 0.72rem; font-weight: 600; color: #52514e; background: #f0efec;
    border: 1px solid rgba(11,11,11,0.08); border-radius: 999px; padding: 0.15rem 0.6rem;}
.mer-badge.on {color: #006300; background: #eaf5ea; border-color: rgba(0,99,0,0.15);}
.mer-disclaimer {font-size: 0.75rem; color: #52514e; background: #f0efec; border-radius: 8px;
    padding: 0.6rem 0.7rem; line-height: 1.45;}

.mer-card {background: #fcfcfb; border: 1px solid rgba(11,11,11,0.10); border-radius: 12px;
    padding: 1rem 1.1rem; margin-bottom: 0.75rem;}
.mer-tiles {display: flex; gap: 0.75rem; margin-bottom: 0.75rem;}
.mer-tile {flex: 1; background: #fcfcfb; border: 1px solid rgba(11,11,11,0.10);
    border-radius: 12px; padding: 0.8rem 1rem;}
.mer-tile .label {font-size: 0.72rem; font-weight: 600; color: #898781;
    text-transform: uppercase; letter-spacing: 0.05em;}
.mer-tile .value {font-size: 1.35rem; font-weight: 700; color: #0b0b0b; margin-top: 0.1rem;}
.mer-tile .sub {font-size: 0.78rem; color: #52514e;}
.mer-alloc-bar {display: flex; width: 100%; height: 34px; border-radius: 8px; overflow: hidden;
    gap: 2px; background: #fcfcfb; margin: 0.4rem 0 0.7rem 0;}
.mer-alloc-seg {height: 100%;}
.mer-legend-row {display: flex; align-items: center; gap: 0.55rem; padding: 0.45rem 0.2rem;
    border-bottom: 1px solid #e1e0d9; font-size: 0.9rem; color: #0b0b0b;}
.mer-legend-row:last-child {border-bottom: none;}
.mer-dot {width: 10px; height: 10px; border-radius: 3px; flex-shrink: 0;}
.mer-legend-pct {font-weight: 700; min-width: 3rem;}
.mer-legend-name {flex: 1;}
.mer-legend-meta {color: #898781; font-size: 0.78rem; font-variant-numeric: tabular-nums;}
.mer-section {font-size: 0.78rem; font-weight: 700; color: #898781; text-transform: uppercase;
    letter-spacing: 0.07em; margin: 1.3rem 0 0.5rem 0;}
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


def _ensure_grounded() -> bool:
    """Build the Chroma store from committed factsheets on first boot (like the
    Streamlit app), so cited sources appear. Best-effort — never blocks startup."""
    try:
        from rag.ingest import build_store_if_missing

        return build_store_if_missing()
    except Exception:
        return False


GROUNDED = _ensure_grounded()

# A committed example recommendation shown before the visitor brings a key, so
# the product is legible immediately — the same demo the Streamlit app renders.
DEMO_REC = load_demo_recommendation() if demo_exists() else None

NO_KEY_MESSAGE = (
    "To chat live I need an LLM API key. Open **🔑 Use your own API key** in the "
    "sidebar and paste one (Google / Anthropic / OpenAI) — it stays in this "
    "session and is sent only to the provider you pick. Meanwhile the example "
    "recommendation below shows what a full result looks like."
)

WELCOME = (
    "Hi, I'm your investment advisor. Let's build a portfolio that fits you. "
    "What are you investing for — retirement, a big purchase, long-term wealth?"
)


def _badges_html() -> str:
    """Sidebar data-source + citation badges, matching the Streamlit sidebar."""
    n = len(OPTIONS)
    source = UNI_META.get("source", "live")
    if source == "live":
        data = f'<span class="mer-badge on">Live data: {n} ETFs</span>'
    elif source == "snapshot":
        asof = f" ({UNI_META.get('as_of')})" if UNI_META.get("as_of") else ""
        data = f'<span class="mer-badge on">Snapshot data: {n} ETFs{asof}</span>'
    else:
        data = f'<span class="mer-badge">Metadata only: {n} ETFs</span>'
    grounded = (
        '<span class="mer-badge on">Citations: grounded</span>'
        if GROUNDED
        else '<span class="mer-badge">Citations: off</span>'
    )
    return f'<div class="mer-badges">{data}{grounded}</div>'


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


def _panel_updates(rec):
    """(plan_html, chart, explanation_md, example_banner) updates for a rec.

    `rec` None hides everything; a real rec shows the plan panel, the backtest
    chart (when history is available) and the rationale, and hides the banner.
    """
    if not rec:
        return (
            gr.update(value="", visible=False),
            gr.update(value=None, visible=False),
            gr.update(value="", visible=False),
        )
    frame = backtest_frame(rec, OPTIONS)
    return (
        gr.update(value=format_plan_html(rec, OPTIONS), visible=True),
        gr.update(value=frame, visible=frame is not None),
        gr.update(value=format_explanation_md(rec, OPTIONS), visible=True),
    )


def respond(message, chat_history, recommendation, model_label, api_key):
    """One conversational turn through the advisor graph."""
    message = (message or "").strip()
    noop = (gr.update(), gr.update(), gr.update(), gr.update())
    if not message:
        return ("", chat_history, recommendation, *noop)

    chat_history = list(chat_history or [])
    chat_history.append({"role": "user", "content": message})

    if not (api_key or "").strip() and not has_provider_key():
        chat_history.append({"role": "assistant", "content": NO_KEY_MESSAGE})
        return ("", chat_history, recommendation, *noop)

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
        return ("", chat_history, recommendation, *noop)

    chat_history.append({"role": "assistant", "content": result.get("assistant_message", "...")})

    if result.get("recommendation"):
        recommendation = result["recommendation"]
        top, plot, bottom = _panel_updates(recommendation)
        banner = gr.update(visible=False)  # real result replaces the example
    else:
        top, plot, bottom, banner = noop
    return ("", chat_history, recommendation, banner, top, plot, bottom)


def reset():
    """'Start over' — clear the chat and fall back to the example recommendation."""
    top, plot, bottom = _panel_updates(DEMO_REC)
    return (
        "",
        [{"role": "assistant", "content": WELCOME}],
        None,
        gr.update(visible=bool(DEMO_REC)),
        top,
        plot,
        bottom,
    )


def build_demo() -> gr.Blocks:
    # NB: `theme`/`css` are applied in `launch()` below — Gradio 6.0 moved them
    # off the Blocks constructor (passing them here is silently ignored).
    with gr.Blocks(title="Meridian — AI Portfolio Advisor") as demo:
        recommendation = gr.State(None)

        with gr.Sidebar(open=True, width=320):
            gr.HTML(
                '<div class="mer-side-title">&#9670; Meridian</div>'
                '<div class="mer-side-sub">Chat-based robo-advisor prototype</div>'
                f"{_badges_html()}"
            )
            gr.Markdown(
                "**How it works**\n"
                "1. Chat about your goal, horizon and risk comfort\n"
                "2. Real 5-year market data scores the fund universe\n"
                "3. You get a portfolio with cited fund facts\n"
                '4. Say *"make it less risky"* to adjust anytime'
            )
            clear_btn = gr.Button("Start over", variant="secondary")

            with gr.Accordion("🔑 Use your own API key", open=False):
                gr.Markdown(
                    "The public demo ships no shared key. Paste your own to chat live — "
                    "it stays in your browser session and is sent only to the provider "
                    "you select."
                )
                model_label = gr.Dropdown(
                    choices=[label for label, _ in BYOK_MODEL_CHOICES],
                    value=BYOK_MODEL_CHOICES[0][0],
                    label="Model",
                )
                api_key = gr.Textbox(
                    label="API key", type="password", placeholder="AIza… / sk-… / sk-ant-…"
                )

            gr.HTML(
                '<div class="mer-disclaimer">Prototype for demonstration purposes. Not '
                "financial advice. Past performance does not predict future returns.</div>"
            )

        gr.Markdown(
            "# ◆ Meridian — AI portfolio advisor\n"
            "Chat about your goal and risk comfort; get an ETF portfolio grounded in "
            "real fund data, with a historical backtest.",
            elem_id="mer-header",
        )

        chatbot = gr.Chatbot(
            value=[{"role": "assistant", "content": WELCOME}],
            height=420,
            show_label=False,
        )
        msg = gr.Textbox(
            placeholder="Tell me about your investment goals...", show_label=False, submit_btn=True
        )

        example_banner = gr.Markdown(
            "👀 **Example recommendation** — a saved sample so you can see the output "
            "without a key. Add your own key in the sidebar and start chatting to get "
            "one tailored to you.",
            visible=bool(DEMO_REC),
        )
        plan_panel = gr.HTML(
            value=format_plan_html(DEMO_REC, OPTIONS) if DEMO_REC else "",
            visible=bool(DEMO_REC),
        )
        _init_frame = backtest_frame(DEMO_REC, OPTIONS) if DEMO_REC else None
        plot = gr.LinePlot(
            value=_init_frame,
            x="date",
            y="value",
            title="How this mix would have performed",
            x_title="",
            y_title="Portfolio value (€)",
            height=260,
            visible=_init_frame is not None,
        )
        explanation_panel = gr.Markdown(
            value=format_explanation_md(DEMO_REC, OPTIONS) if DEMO_REC else "",
            visible=bool(DEMO_REC),
        )

        outputs = [
            msg,
            chatbot,
            recommendation,
            example_banner,
            plan_panel,
            plot,
            explanation_panel,
        ]
        msg.submit(
            respond,
            [msg, chatbot, recommendation, model_label, api_key],
            outputs,
        )
        clear_btn.click(
            reset,
            None,
            [msg, chatbot, recommendation, example_banner, plan_panel, plot, explanation_panel],
        )
        demo.load(_zerogpu_startup_probe, inputs=None, outputs=None)
    return demo


if __name__ == "__main__":
    build_demo().launch(theme=THEME, css=CSS)

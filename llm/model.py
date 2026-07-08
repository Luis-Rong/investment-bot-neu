"""Multi-provider LLM factory.

The active model is chosen by the `LLM_MODEL` env var in the
`"<provider>:<model>"` format LangChain's `init_chat_model` understands, e.g.

    google_genai:gemini-2.5-flash
    anthropic:claude-sonnet-5
    openai:gpt-4o-mini
    watsonx:ibm/granite-3-8b-instruct

Only the provider package matching the configured model needs to be installed
(see the `providers` extra in `pyproject.toml`). Switching providers is a
config change, not a code change.
"""

from __future__ import annotations

import os

from langchain.chat_models import init_chat_model

DEFAULT_MODEL = "google_genai:gemini-3.1-flash-lite"


def get_llm(temperature: float = 0.3):
    """Return a chat model for the provider/model in `LLM_MODEL` (or the default)."""
    model = os.getenv("LLM_MODEL", DEFAULT_MODEL)
    return init_chat_model(model, temperature=temperature)


def response_text(resp) -> str:
    """Plain text from a chat-model response, robust across content shapes.

    Content-block-aware models (e.g. Gemini 3.x) return `.content` as a list of
    block dicts (`[{"type": "text", "text": "..."}]`) instead of a flat string;
    LangChain's `AIMessage.text` property normalizes that. Plain test doubles
    that only set a string `.content` (no `.text` attribute) predate that
    convention, so fall back to `.content` for those — keeps stubs unchanged.
    """
    text = getattr(resp, "text", None)
    if text is not None:
        return str(text).strip()
    content = resp.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "".join(parts).strip()
    return str(content).strip()

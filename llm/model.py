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

Bring-your-own-key (BYOK): `get_llm` accepts an explicit `api_key`, passed
straight to the provider constructor as a keyword argument. It is deliberately
**not** written to `os.environ` — the public deployment serves many visitors
from one process, and a shared env var would leak one visitor's key into
another's requests.
"""

from __future__ import annotations

import os

from langchain.chat_models import init_chat_model

DEFAULT_MODEL = "google_genai:gemini-3.1-flash-lite"

# Environment variable each provider reads its key from (owner-supplied keys).
PROVIDER_KEY_ENV = {
    "google_genai": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "watsonx": "WATSONX_APIKEY",
}

# Constructor keyword each provider accepts a key through, for BYOK (no env var).
PROVIDER_KEY_KWARG = {
    "google_genai": "google_api_key",
    "anthropic": "api_key",
    "openai": "api_key",
}

# Models offered in the BYOK picker: (label, "provider:model").
BYOK_MODEL_CHOICES = [
    ("Google — Gemini 3.1 Flash Lite", "google_genai:gemini-3.1-flash-lite"),
    ("Google — Gemini 2.5 Flash", "google_genai:gemini-2.5-flash"),
    ("Anthropic — Claude Sonnet 5", "anthropic:claude-sonnet-5"),
    ("OpenAI — GPT-4o mini", "openai:gpt-4o-mini"),
]


def resolve_model(model: str | None = None) -> str:
    """The active `provider:model` string (explicit arg, env, or default)."""
    return model or os.getenv("LLM_MODEL", DEFAULT_MODEL)


def provider_of(model: str | None = None) -> str:
    """The provider prefix of a `provider:model` string."""
    return resolve_model(model).split(":", 1)[0]


def has_provider_key(model: str | None = None) -> bool:
    """Whether an owner-supplied key for this model's provider is in the env."""
    env_var = PROVIDER_KEY_ENV.get(provider_of(model))
    return bool(env_var and os.getenv(env_var))


def get_llm(temperature: float = 0.3, model: str | None = None, api_key: str | None = None):
    """Return a chat model for the given (or configured) provider/model.

    When `api_key` is supplied (BYOK) it is passed as a constructor keyword for
    the provider, never via `os.environ`.
    """
    model = resolve_model(model)
    kwargs = {"temperature": temperature}
    if api_key:
        kwarg = PROVIDER_KEY_KWARG.get(model.split(":", 1)[0], "api_key")
        kwargs[kwarg] = api_key
    return init_chat_model(model, **kwargs)


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

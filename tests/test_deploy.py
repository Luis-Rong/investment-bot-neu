"""Tests for the Phase 6 deploy layer: snapshot, universe fallback, BYOK.

All offline — network calls (yfinance) and LLM construction are monkeypatched,
so these run in CI without keys or connectivity.
"""

from __future__ import annotations

import json

import pytest

import data_sources.snapshot as snap
import data_sources.universe as universe
import llm.model as model
from ui.demo import DEMO_PATH, load_demo_recommendation

FAKE_OPTIONS = [
    {"id": "a", "ticker": "AAA", "name": "Fund A", "ter": 0.1, "volatility": 0.15},
    {"id": "b", "ticker": "BBB", "name": "Fund B", "ter": 0.2, "volatility": 0.20},
]


# --- snapshot round-trip ---------------------------------------------------- #


def test_snapshot_save_and_load_roundtrip(tmp_path, monkeypatch):
    path = tmp_path / "snap.json"
    monkeypatch.setattr(snap, "build_universe", lambda period="5y": FAKE_OPTIONS)

    n = snap.save_snapshot(path=path, as_of="2026-01-01")
    assert n == 2
    assert snap.snapshot_exists(path)

    loaded = snap.load_snapshot(path)
    assert loaded == FAKE_OPTIONS

    meta = snap.snapshot_meta(path)
    assert meta == {"as_of": "2026-01-01", "period": "5y", "n": 2}


def test_saved_snapshot_is_valid_json(tmp_path, monkeypatch):
    path = tmp_path / "snap.json"
    monkeypatch.setattr(snap, "build_universe", lambda period="5y": FAKE_OPTIONS)
    snap.save_snapshot(path=path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data) == {"as_of", "period", "options"}


# --- resolve_universe source selection & fallback --------------------------- #


def test_resolve_prefers_snapshot_by_default(monkeypatch):
    monkeypatch.delenv("USE_LIVE_DATA", raising=False)

    def _boom(period="5y"):
        raise AssertionError("live fetch must not run when snapshot is preferred")

    monkeypatch.setattr(universe, "build_universe", _boom)
    options, meta = universe.resolve_universe()
    assert meta["source"] == "snapshot"
    assert len(options) > 0


def test_resolve_live_falls_back_to_snapshot_on_error(monkeypatch):
    monkeypatch.setattr(
        universe, "build_universe", lambda period="5y": (_ for _ in ()).throw(RuntimeError("net"))
    )
    options, meta = universe.resolve_universe(prefer_live=True)
    assert meta["source"] == "snapshot"
    assert len(options) > 0


def test_resolve_live_used_when_available(monkeypatch):
    monkeypatch.setattr(universe, "build_universe", lambda period="5y": FAKE_OPTIONS)
    options, meta = universe.resolve_universe(prefer_live=True)
    assert meta["source"] == "live"
    assert options == FAKE_OPTIONS


def test_resolve_static_when_no_snapshot(monkeypatch):
    monkeypatch.setattr(snap, "snapshot_exists", lambda *a, **k: False)
    options, meta = universe.resolve_universe(prefer_live=False)
    assert meta["source"] == "static"
    assert len(options) > 0  # falls back to raw metadata


def test_want_live_env_flag(monkeypatch):
    monkeypatch.setenv("USE_LIVE_DATA", "1")
    assert universe._want_live() is True
    monkeypatch.setenv("USE_LIVE_DATA", "no")
    assert universe._want_live() is False
    monkeypatch.delenv("USE_LIVE_DATA", raising=False)
    assert universe._want_live() is False


# --- BYOK / provider key resolution ----------------------------------------- #


def test_resolve_model_precedence(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    assert model.resolve_model() == model.DEFAULT_MODEL
    assert model.resolve_model("anthropic:claude-sonnet-5") == "anthropic:claude-sonnet-5"
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o-mini")
    assert model.resolve_model() == "openai:gpt-4o-mini"


def test_provider_of():
    assert model.provider_of("google_genai:gemini-3.1-flash-lite") == "google_genai"
    assert model.provider_of("anthropic:claude-sonnet-5") == "anthropic"


def test_has_provider_key(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "google_genai:gemini-3.1-flash-lite")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert model.has_provider_key() is False
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza-x")
    assert model.has_provider_key() is True


@pytest.mark.parametrize(
    "model_str,expected_kwarg",
    [
        ("google_genai:gemini-3.1-flash-lite", "google_api_key"),
        ("anthropic:claude-sonnet-5", "api_key"),
        ("openai:gpt-4o-mini", "api_key"),
    ],
)
def test_get_llm_passes_byok_key_as_kwarg_not_env(monkeypatch, model_str, expected_kwarg):
    captured = {}

    def fake_init(model_id, **kwargs):
        captured["model"] = model_id
        captured["kwargs"] = kwargs
        return "LLM"

    monkeypatch.setattr(model, "init_chat_model", fake_init)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    model.get_llm(model=model_str, api_key="secret-key")
    assert captured["model"] == model_str
    assert captured["kwargs"][expected_kwarg] == "secret-key"
    # The key must not leak into the shared process environment.
    assert "secret-key" not in {v for v in (__import__("os").environ.values())}


def test_get_llm_without_key_omits_key_kwarg(monkeypatch):
    captured = {}
    monkeypatch.setattr(model, "init_chat_model", lambda m, **k: captured.update(kwargs=k))
    model.get_llm(model="google_genai:gemini-3.1-flash-lite")
    assert "google_api_key" not in captured["kwargs"]
    assert captured["kwargs"] == {"temperature": 0.3}


# --- committed demo recommendation ------------------------------------------ #


def test_demo_recommendation_wellformed():
    assert DEMO_PATH.exists()
    rec = load_demo_recommendation()
    assert set(rec) >= {"profile", "allocation", "contributions", "explanation", "sources"}
    assert sum(a["percentage"] for a in rec["allocation"]) == 100
    # Sources rebuilt as passages the renderer can read via attribute access.
    for passage in rec["sources"]:
        assert passage.text and passage.source

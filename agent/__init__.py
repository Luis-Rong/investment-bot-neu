"""Agentic layer: LangGraph state machine with query routing + reflection.

- `state.py`  — the shared `AdvisorState` TypedDict.
- `tools.py`  — `@tool` wrappers around the data / RAG / logic functions.
- `graph.py`  — the compiled `StateGraph` (route → prepare → compute → retrieve
  → explain → reflect loop) plus its deterministic helpers.
"""

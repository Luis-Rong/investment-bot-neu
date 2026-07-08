"""Phase 5 — evaluation harness.

Two evaluators over the LLM pipeline:

- `extraction` — a labelled test set (chat transcript → expected `UserProfile`)
  measuring how accurately the profile-extraction node parses free-form chat.
- `judge` — an LLM-as-judge pass scoring recommendation explanations on
  grounding, relevance, clarity and safety.

The *scoring* logic (field comparison, aggregation, judge-output parsing) is
pure and unit-tested offline in `tests/test_eval.py`; the *runners* call a real
LLM and are driven from `python -m eval.run`.
"""

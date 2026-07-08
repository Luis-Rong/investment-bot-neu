"""Committed example recommendation for the no-key demo state.

A public visitor without an API key still sees a full, real recommendation
(generated once from the live advisor graph, then frozen) so the product is
legible before they bring their own key. Regenerate by re-running the advisor
graph for a sample profile and dumping `recommendation` to the JSON below.
"""

from __future__ import annotations

import json
from pathlib import Path

DEMO_PATH = Path(__file__).resolve().parent.parent / "data" / "demo_recommendation.json"


def demo_exists(path: Path = DEMO_PATH) -> bool:
    return path.exists()


def load_demo_recommendation(path: Path = DEMO_PATH) -> dict:
    """Return the example recommendation, sources rebuilt as `RetrievedPassage`.

    The renderer accesses `source.text` / `source.source`, so the JSON's source
    dicts are re-wrapped in the dataclass the live path produces.
    """
    from rag.retriever import RetrievedPassage

    with open(path, encoding="utf-8") as f:
        rec = dict(json.load(f)["recommendation"])
    rec["sources"] = [RetrievedPassage(**s) for s in rec.get("sources", [])]
    return rec

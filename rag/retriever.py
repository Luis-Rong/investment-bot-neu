"""Query the factsheet vector store and return passages with source citations.

This is the grounding layer: instead of letting the LLM free-generate fund
claims, callers retrieve real passages here and cite them. Returns an empty
list (instead of raising) when the store hasn't been built yet, so the app
degrades gracefully to ungrounded mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rag.ingest import CHROMA_DIR, COLLECTION_NAME


@dataclass
class RetrievedPassage:
    text: str
    source: str  # filename of the originating document — the citation
    ticker: str | None
    score: float  # cosine distance; lower = more relevant


def store_exists(persist_dir: Path = CHROMA_DIR) -> bool:
    return persist_dir.exists() and any(persist_dir.iterdir())


def retrieve(
    query: str,
    k: int = 4,
    ticker: str | None = None,
    persist_dir: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    embedding_function=None,
) -> list[RetrievedPassage]:
    """Top-`k` passages for `query`, optionally restricted to one fund's docs."""
    if not store_exists(persist_dir):
        return []

    import chromadb

    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        kwargs = {"embedding_function": embedding_function} if embedding_function else {}
        collection = client.get_collection(collection_name, **kwargs)
    except Exception:
        return []

    where = {"ticker": ticker} if ticker else None
    result = collection.query(query_texts=[query], n_results=k, where=where)

    passages = []
    for text, meta, dist in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
    ):
        passages.append(
            RetrievedPassage(
                text=text,
                source=meta.get("source", "unknown"),
                ticker=meta.get("ticker"),
                score=dist,
            )
        )
    return passages


def format_evidence(passages: list[RetrievedPassage]) -> str:
    """Render passages as a numbered, source-tagged block for an LLM prompt."""
    if not passages:
        return "No factsheet passages available."
    blocks = []
    for i, p in enumerate(passages, 1):
        blocks.append(f"[{i}] (source: {p.source})\n{p.text}")
    return "\n\n".join(blocks)

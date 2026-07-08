"""Tests for the RAG layer: chunking, document iteration, store round-trip.

The store round-trip injects a deterministic fake embedding function, so no
model download and no network are needed — CI-safe.
"""

import hashlib

import pytest
from chromadb import EmbeddingFunction

from rag.ingest import build_store, chunk_text, iter_documents
from rag.retriever import format_evidence, retrieve


def test_chunk_empty_text():
    assert chunk_text("") == []
    assert chunk_text("\n\n   \n\n") == []


def test_chunk_short_text_single_chunk():
    text = "One paragraph.\n\nAnother paragraph."
    assert chunk_text(text, chunk_size=900) == [text]


def test_chunk_splits_and_overlaps():
    paragraphs = [f"Paragraph {i} " + "x" * 300 for i in range(6)]
    text = "\n\n".join(paragraphs)
    chunks = chunk_text(text, chunk_size=700, overlap=1)

    assert len(chunks) > 1
    # Overlap: the last paragraph of chunk N reappears at the start of chunk N+1.
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.startswith(prev.split("\n\n")[-1])
    # Nothing lost: every paragraph appears somewhere.
    joined = "\n\n".join(chunks)
    for para in paragraphs:
        assert para in joined


def test_contextualize_chunks_prefixes_title():
    from rag.ingest import contextualize_chunks, doc_title

    text = "# iShares Core S&P 500 ETF (IVV)\n\nintro"
    title = doc_title(text, "IVV.md")
    assert title == "iShares Core S&P 500 ETF (IVV)"

    chunks = ["# iShares Core S&P 500 ETF (IVV)\n\nintro", "## Risk profile\n\n- vol: 17%"]
    out = contextualize_chunks(chunks, title)
    assert out[0] == chunks[0]  # already contains the title
    assert out[1].startswith(f"[{title}]")


def test_iter_documents_reads_md_and_infers_ticker(tmp_path):
    (tmp_path / "IVV.md").write_text("# iShares Core S&P 500\n\nEquity fund.", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("General investing notes.", encoding="utf-8")
    (tmp_path / "ignored.csv").write_text("a,b", encoding="utf-8")

    docs = list(iter_documents(tmp_path))

    assert len(docs) == 2
    by_source = {meta["source"]: meta for _, meta in docs}
    assert by_source["IVV.md"]["ticker"] == "IVV"
    assert "ticker" not in by_source["notes.txt"]


class _FakeEmbedder(EmbeddingFunction):
    """Deterministic embedding: texts sharing words get closer vectors."""

    def __call__(self, input):  # chroma's EmbeddingFunction protocol
        return [self._embed(text) for text in input]

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def _embed(text: str) -> list[float]:
        vec = [0.0] * 64
        for word in text.lower().split():
            idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % 64
            vec[idx] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    @staticmethod
    def name() -> str:
        return "fake-embedder"


@pytest.fixture
def small_store(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "IVV.md").write_text(
        "# iShares Core S&P 500 ETF\n\nLow cost equity fund tracking large US stocks.",
        encoding="utf-8",
    )
    (docs / "AGG.md").write_text(
        "# iShares Core US Aggregate Bond ETF\n\nBond fund holding investment grade bonds.",
        encoding="utf-8",
    )
    persist = tmp_path / "chroma"
    embedder = _FakeEmbedder()
    n = build_store(docs_dir=docs, persist_dir=persist, embedding_function=embedder)
    assert n == 2
    return persist, embedder


def test_store_roundtrip_returns_cited_passages(small_store):
    persist, embedder = small_store
    hits = retrieve(
        "bond fund with investment grade bonds",
        k=1,
        persist_dir=persist,
        embedding_function=embedder,
    )
    assert len(hits) == 1
    assert hits[0].source == "AGG.md"
    assert hits[0].ticker == "AGG"
    assert "bond" in hits[0].text.lower()


def test_retrieve_with_ticker_filter(small_store):
    persist, embedder = small_store
    hits = retrieve(
        "equity fund", k=2, ticker="IVV", persist_dir=persist, embedding_function=embedder
    )
    assert hits and all(h.ticker == "IVV" for h in hits)


def test_retrieve_missing_store_returns_empty(tmp_path):
    assert retrieve("anything", persist_dir=tmp_path / "nope") == []


def test_format_evidence():
    assert "No factsheet passages" in format_evidence([])
    from rag.retriever import RetrievedPassage

    block = format_evidence(
        [RetrievedPassage(text="Some fact.", source="IVV.md", ticker="IVV", score=0.1)]
    )
    assert "[1]" in block and "IVV.md" in block and "Some fact." in block

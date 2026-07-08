"""Build the Chroma vector store from the factsheet corpus.

Reads every document under `data/factsheets/` — the generated Markdown fund
profiles (see `rag.build_docs`) plus any official PDF factsheets/KIIDs the
user drops in — chunks them paragraph-aware, and persists the embeddings to
`chroma_db/` (gitignored).

Embeddings use Chroma's built-in ONNX MiniLM model (all-MiniLM-L6-v2), which
runs locally and free — no API key, no torch install needed.

Run:  python -m rag.ingest
"""

from __future__ import annotations

import json
from pathlib import Path

from rag import _compat  # noqa: F401  — swaps in pysqlite3 before chromadb loads

FACTSHEETS_DIR = Path(__file__).resolve().parent.parent / "data" / "factsheets"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "factsheets"
UNIVERSE_PATH = Path(__file__).resolve().parent.parent / "data" / "universe.json"


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 1) -> list[str]:
    """Paragraph-aware chunking: pack whole paragraphs up to `chunk_size` chars.

    `overlap` is the number of trailing paragraphs repeated at the start of
    the next chunk, so context (like the section heading) carries over.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[list[str]] = []
    current: list[str] = []
    length = 0
    for para in paragraphs:
        if current and length + len(para) > chunk_size:
            chunks.append(current)
            current = current[-overlap:] if overlap else []
            length = sum(len(p) for p in current)
        current.append(para)
        length += len(para)
    if current:
        chunks.append(current)
    return ["\n\n".join(c) for c in chunks]


def doc_title(text: str, fallback: str) -> str:
    """First Markdown heading of the document, or `fallback`."""
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def contextualize_chunks(chunks: list[str], title: str) -> list[str]:
    """Prefix continuation chunks with the document title.

    A chunk holding only e.g. the risk-metrics section would otherwise not
    say which fund it describes, which hurts embedding relevance badly.
    """
    return [c if title in c else f"[{title}]\n\n{c}" for c in chunks]


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader  # imported lazily: only needed for PDF input

    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _known_tickers() -> set[str]:
    with open(UNIVERSE_PATH, encoding="utf-8") as f:
        return {opt["ticker"] for opt in json.load(f)}


def iter_documents(docs_dir: Path = FACTSHEETS_DIR):
    """Yield (text, metadata) for every .md/.txt/.pdf under `docs_dir`.

    Metadata carries the citation source (filename) and, when the filename
    stem matches a universe ticker, the ticker for filtered retrieval.
    """
    tickers = _known_tickers()
    for path in sorted(docs_dir.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt", ".pdf"}:
            continue
        if path.suffix.lower() == ".pdf":
            text = _read_pdf(path)
        else:
            text = path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        metadata = {"source": path.name}
        if path.stem.upper() in tickers:
            metadata["ticker"] = path.stem.upper()
        yield text, metadata


def build_store(
    docs_dir: Path = FACTSHEETS_DIR,
    persist_dir: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    embedding_function=None,
) -> int:
    """(Re)build the vector store; returns the number of chunks indexed.

    `embedding_function` lets tests inject a cheap deterministic embedder;
    None means Chroma's default local ONNX MiniLM model.
    """
    import chromadb

    client = chromadb.PersistentClient(path=str(persist_dir))
    # Rebuild from scratch so deleted/renamed source files don't leave stale chunks.
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    kwargs = {"embedding_function": embedding_function} if embedding_function else {}
    collection = client.create_collection(collection_name, **kwargs)

    total = 0
    for text, metadata in iter_documents(docs_dir):
        chunks = contextualize_chunks(chunk_text(text), doc_title(text, metadata["source"]))
        collection.add(
            ids=[f"{metadata['source']}:{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[metadata] * len(chunks),
        )
        total += len(chunks)
        print(f"indexed {metadata['source']}: {len(chunks)} chunks")
    return total


def build_store_if_missing(
    docs_dir: Path = FACTSHEETS_DIR,
    persist_dir: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> bool:
    """Build the store from the committed factsheets iff it doesn't exist yet.

    The generated fund profiles are committed but `chroma_db/` is gitignored, so
    a fresh deploy has the corpus but no vector store. This rebuilds it offline
    (no yfinance) on first boot. Returns True if a store is present afterwards;
    swallows failures (e.g. Chroma unavailable) so the app still starts, just
    ungrounded.
    """
    if persist_dir.exists() and any(persist_dir.iterdir()):
        return True
    generated = docs_dir / "generated"
    if not generated.exists() or not any(generated.glob("*.md")):
        return False
    try:
        build_store(docs_dir, persist_dir, collection_name)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    generated = FACTSHEETS_DIR / "generated"
    if not generated.exists() or not any(generated.glob("*.md")):
        raise SystemExit(
            "No fund profiles found. Run `python -m rag.build_docs` first "
            f"(or drop PDF factsheets into {FACTSHEETS_DIR})."
        )
    n = build_store()
    print(f"\n{n} chunks persisted to {CHROMA_DIR}")

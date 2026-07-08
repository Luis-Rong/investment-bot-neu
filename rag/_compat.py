"""SQLite compatibility shim for hosts with an old system `sqlite3`.

Chroma requires SQLite >= 3.35, but some managed hosts (notably Streamlit
Community Cloud's Debian image) ship an older build, which makes
`import chromadb` raise at startup. When the `pysqlite3-binary` wheel is
installed (see `requirements.txt`, Linux-only marker), swapping it in for the
stdlib `sqlite3` before Chroma imports fixes this. On dev machines with a
recent SQLite the wheel isn't present and this is a harmless no-op.

Import this module **before** importing `chromadb`.
"""

from __future__ import annotations

import sys


def ensure_modern_sqlite() -> None:
    try:
        import pysqlite3  # type: ignore
    except ModuleNotFoundError:
        return  # system sqlite3 is fine (dev machines) — nothing to do
    sys.modules["sqlite3"] = pysqlite3
    sys.modules["sqlite3.dbapi2"] = pysqlite3.dbapi2


ensure_modern_sqlite()

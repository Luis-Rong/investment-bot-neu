"""Frozen snapshot of the enriched ETF universe for offline / demo use.

Live `yfinance` calls are unreliable from cloud hosts — Yahoo rate-limits and
often outright blocks datacenter IPs — and they add latency to every cold
start. This module persists a point-in-time snapshot of `build_universe()`
output to `data/universe_snapshot.json` (committed) so the deployed app can
load a full, metric-enriched universe with **no network at all**.

Regenerate locally (with a working network) via:

    python -m data_sources.snapshot
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from data_sources.universe import build_universe

SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "data" / "universe_snapshot.json"


def save_snapshot(
    period: str = "5y",
    path: Path = SNAPSHOT_PATH,
    as_of: str | None = None,
) -> int:
    """Fetch live metrics and write the snapshot. Returns the ETF count."""
    options = build_universe(period)
    payload = {
        "as_of": as_of or date.today().isoformat(),
        "period": period,
        "options": options,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return len(options)


def load_snapshot(path: Path = SNAPSHOT_PATH) -> list[dict]:
    """Read the enriched ETF options from the committed snapshot."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)["options"]


def snapshot_meta(path: Path = SNAPSHOT_PATH) -> dict:
    """As-of date, period and ETF count for the snapshot (for UI badges)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        "as_of": data.get("as_of"),
        "period": data.get("period"),
        "n": len(data.get("options", [])),
    }


def snapshot_exists(path: Path = SNAPSHOT_PATH) -> bool:
    return path.exists()


if __name__ == "__main__":
    n = save_snapshot()
    print(f"snapshot written: {n} ETFs -> {SNAPSHOT_PATH}")

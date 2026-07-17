"""Turn scored options into a percentage allocation, with per-asset-class caps.

The raw allocation weights each selected fund by its score. On top of that we
enforce **asset-class caps** so no single class can dominate the portfolio just
because it happens to have run hot recently — most importantly gold/commodities,
whose trailing risk-adjusted return can top the whole universe after a strong
few years without that being a signal of *future* strength. Weight trimmed off a
capped class is redistributed proportionally to the uncapped holdings.
"""

# Maximum share (in %) any single asset class may hold. Classes not listed are
# uncapped. Commodities (gold) are capped because a strong trailing window
# otherwise lets them crowd out a diversified core.
CLASS_CAPS = {"commodity": 15.0}


def _apply_class_caps(items, caps):
    """Trim any capped class down to its ceiling, redistributing the freed
    weight to holdings in uncapped classes (proportional to current weight).

    Operates in-place on `items` (each a dict with `asset_class` and float
    `weight`). Iterates a bounded number of times so a redistribution that would
    push another capped class over its ceiling still settles.
    """
    if not caps:
        return
    for _ in range(len(items) + 2):
        totals = {}
        for it in items:
            totals[it["asset_class"]] = totals.get(it["asset_class"], 0.0) + it["weight"]

        freed = 0.0
        for cls, cap in caps.items():
            total = totals.get(cls, 0.0)
            if total > cap + 1e-9:
                scale = cap / total
                for it in items:
                    if it["asset_class"] == cls:
                        freed += it["weight"] * (1 - scale)
                        it["weight"] *= scale

        if freed <= 1e-9:
            break

        # Redistribute only to classes that aren't capped, so a capped class
        # can't simply reabsorb the weight we just trimmed.
        receivers = [it for it in items if it["asset_class"] not in caps]
        base = sum(it["weight"] for it in receivers)
        if base <= 1e-9:
            break  # nothing to receive (degenerate) — leave the shortfall to rounding
        for it in receivers:
            it["weight"] += freed * (it["weight"] / base)


def build_allocation(scored_options, top_n=3, class_caps=None):
    caps = CLASS_CAPS if class_caps is None else class_caps
    selected = scored_options[:top_n]
    if not selected:
        return []

    total_score = sum(score for _, score in selected) or 1
    items = [
        {
            "id": opt["id"],
            "name": opt["name"],
            "asset_class": opt.get("asset_class"),
            "weight": (score / total_score) * 100.0,
        }
        for opt, score in selected
    ]

    _apply_class_caps(items, caps)

    for it in items:
        it["percentage"] = round(it["weight"])

    # Rundungsfehler ausgleichen, damit Summe 100 ist. Absorb the drift into the
    # largest *uncapped* holding so the correction can't push a capped class back
    # over its ceiling; fall back to the first holding if everything is capped.
    diff = 100 - sum(it["percentage"] for it in items)
    uncapped = [it for it in items if it["asset_class"] not in caps]
    target = max(uncapped, key=lambda it: it["weight"]) if uncapped else items[0]
    target["percentage"] += diff

    return [{"id": it["id"], "name": it["name"], "percentage": it["percentage"]} for it in items]

from logic.allocation import build_allocation


def _scored(*scores):
    # Build fake (option, score) pairs the way score_options returns them.
    return [({"id": f"OPT_{i}", "name": f"Option {i}"}, s) for i, s in enumerate(scores)]


def test_percentages_sum_to_100():
    allocation = build_allocation(_scored(9.0, 6.0, 3.0), top_n=3)
    assert sum(item["percentage"] for item in allocation) == 100


def test_respects_top_n():
    allocation = build_allocation(_scored(9.0, 8.0, 7.0, 6.0, 5.0), top_n=3)
    assert len(allocation) == 3


def test_rounding_diff_absorbed_by_first():
    # Three equal scores round to 33 each (=99); the missing 1% goes to item[0].
    allocation = build_allocation(_scored(1.0, 1.0, 1.0), top_n=3)
    assert sum(item["percentage"] for item in allocation) == 100
    assert allocation[0]["percentage"] == 34


def test_empty_input_returns_empty():
    assert build_allocation([], top_n=3) == []


def test_higher_score_gets_higher_share():
    allocation = build_allocation(_scored(9.0, 1.0), top_n=2)
    assert allocation[0]["percentage"] > allocation[1]["percentage"]


def _scored_classes(*triples):
    # (id, asset_class, score) triples, the way score_options yields options.
    return [
        ({"id": f"OPT_{i}", "name": f"Option {i}", "asset_class": cls}, score)
        for i, (cls, score) in enumerate(triples)
    ]


def test_commodity_capped_at_15_percent():
    # A dominant commodity (gold) would take ~60% by raw score; the cap trims it.
    alloc = build_allocation(
        _scored_classes(("commodity", 9.0), ("bond", 3.0), ("bond", 3.0)), top_n=3
    )
    gold = next(a for a in alloc if a["id"] == "OPT_0")
    assert gold["percentage"] == 15
    assert sum(a["percentage"] for a in alloc) == 100


def test_cap_redistributes_to_uncapped_holdings():
    alloc = build_allocation(
        _scored_classes(("commodity", 9.0), ("bond", 3.0), ("bond", 3.0)), top_n=3
    )
    bonds = [a["percentage"] for a in alloc if a["id"] != "OPT_0"]
    # The 45%+ freed from gold lands on the two equal-score bonds, evenly.
    assert bonds == [43, 42] or bonds == [42, 43] or bonds == [43, 43]


def test_uncapped_classes_untouched():
    # No commodity present -> caps are a no-op, weighting is purely by score.
    alloc = build_allocation(
        _scored_classes(("equity", 6.0), ("bond", 3.0), ("bond", 1.0)), top_n=3
    )
    assert alloc[0]["percentage"] == 60

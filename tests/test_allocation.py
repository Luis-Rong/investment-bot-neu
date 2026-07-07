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

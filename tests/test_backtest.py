import math

from logic.backtest import run_backtest

OPTIONS = [
    {"id": "A", "ticker": "AAA"},
    {"id": "B", "ticker": "BBB"},
    {"id": "C", "ticker": "CCC"},
]


def _months(n):
    return [f"m{i:02d}" for i in range(n)]


def _series(prices):
    """`{month: price}` for consecutive months."""
    return dict(zip(_months(len(prices)), prices, strict=True))


def test_single_fund_doubling_over_a_year():
    hist = {"AAA": _series([100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 195, 198, 200])}
    alloc = [{"id": "A", "name": "A", "percentage": 100}]
    r = run_backtest(alloc, OPTIONS, hist, initial=1000)
    assert r is not None
    assert math.isclose(r.total_return, 1.0)
    assert math.isclose(r.final, 2000.0)
    assert math.isclose(r.annual_return, 1.0)  # 12 monthly steps = 1 year, value doubled
    assert r.max_drawdown == 0.0  # monotonic climb
    assert r.months == 12
    assert r.values[0] == 1000.0


def test_blend_averages_the_funds_by_weight():
    hist = {"AAA": _series([100, 200]), "BBB": _series([100, 100])}
    alloc = [
        {"id": "A", "name": "A", "percentage": 50},
        {"id": "B", "name": "B", "percentage": 50},
    ]
    r = run_backtest(alloc, OPTIONS, hist, initial=1000)
    # 50% doubled + 50% flat -> +50% overall
    assert math.isclose(r.total_return, 0.5)
    assert math.isclose(r.final, 1500.0)


def test_weights_renormalize_when_a_fund_has_no_history():
    hist = {"AAA": _series([100, 200])}  # CCC absent
    alloc = [
        {"id": "A", "name": "A", "percentage": 50},
        {"id": "C", "name": "C", "percentage": 50},
    ]
    r = run_backtest(alloc, OPTIONS, hist, initial=1000)
    # Only AAA survives -> renormalized to 100%, so it doubles.
    assert math.isclose(r.total_return, 1.0)


def test_max_drawdown_captures_the_worst_dip():
    hist = {"AAA": _series([100, 150, 75, 120])}
    alloc = [{"id": "A", "name": "A", "percentage": 100}]
    r = run_backtest(alloc, OPTIONS, hist, initial=1000)
    assert math.isclose(r.max_drawdown, -0.5)  # peak 150 -> trough 75


def test_returns_none_without_usable_history():
    alloc = [{"id": "A", "name": "A", "percentage": 100}]
    assert run_backtest(alloc, OPTIONS, {}, initial=1000) is None
    assert run_backtest(alloc, OPTIONS, {"ZZZ": _series([1, 2])}, initial=1000) is None

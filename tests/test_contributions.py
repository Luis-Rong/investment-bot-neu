from logic.contributions import compute_contributions


def test_no_savings_returns_zero():
    result = compute_contributions(0, 0)
    assert result == {"one_time_eur": 0, "monthly_plan_eur": 0}


def test_small_lump_sum_invested_fully():
    # Below the 500 EUR threshold: invest everything, no buffer.
    result = compute_contributions(300, 0)
    assert result["one_time_eur"] == 300


def test_large_lump_sum_keeps_buffer():
    # 1000 EUR: 10% buffer = 100 EUR -> invest 900.
    result = compute_contributions(1000, 0)
    assert result["one_time_eur"] == 900


def test_buffer_capped_at_1000():
    # 20000 EUR: 10% would be 2000, but the buffer is capped at 1000.
    result = compute_contributions(20000, 0)
    assert result["one_time_eur"] == 19000


def test_monthly_plan_passed_through():
    result = compute_contributions(0, 250)
    assert result["monthly_plan_eur"] == 250


def test_none_inputs_treated_as_zero():
    result = compute_contributions(None, None)
    assert result == {"one_time_eur": 0, "monthly_plan_eur": 0}


def test_negative_monthly_clamped_to_zero():
    result = compute_contributions(0, -50)
    assert result["monthly_plan_eur"] == 0

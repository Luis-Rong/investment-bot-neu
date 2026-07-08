from logic.contributions import compute_contributions


def test_no_savings_returns_zero():
    result = compute_contributions(0, 0)
    assert result == {"one_time_eur": 0, "monthly_plan_eur": 0}


def test_small_lump_sum_invested_fully():
    result = compute_contributions(300, 0)
    assert result["one_time_eur"] == 300


def test_large_lump_sum_invested_fully():
    # The lump sum is already earmarked for investing -> invest all of it.
    result = compute_contributions(1000, 0)
    assert result["one_time_eur"] == 1000


def test_very_large_lump_sum_invested_fully():
    result = compute_contributions(20000, 0)
    assert result["one_time_eur"] == 20000


def test_monthly_plan_passed_through():
    result = compute_contributions(0, 250)
    assert result["monthly_plan_eur"] == 250


def test_none_inputs_treated_as_zero():
    result = compute_contributions(None, None)
    assert result == {"one_time_eur": 0, "monthly_plan_eur": 0}


def test_negative_monthly_clamped_to_zero():
    result = compute_contributions(0, -50)
    assert result["monthly_plan_eur"] == 0

def compute_contributions(saving_eur: int, monthly_saving_eur: int):
    """
    Returns a contribution plan:
    - one_time_eur: the lump sum to invest today
    - monthly_plan_eur: the amount to invest each month

    `saving_eur` is the amount the user has already earmarked for investing, so
    it's invested in full — we don't silently hold part of it back.
    """
    one_time = max(0, int(saving_eur or 0))
    monthly_plan = max(0, int(monthly_saving_eur or 0))
    return {"one_time_eur": one_time, "monthly_plan_eur": monthly_plan}

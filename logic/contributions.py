def compute_contributions(saving_eur: int, monthly_saving_eur: int):
    """
    Returns a realistic contribution plan:
    - one_time_eur: amount to invest today (cannot exceed saving_eur)
    - monthly_plan_eur: amount to invest each month
    """
    saving_eur = int(saving_eur or 0)
    monthly_saving_eur = int(monthly_saving_eur or 0)

    # Realistic defaults:
    # If user has a lump sum, suggest investing most of it, but not all (keep a small buffer)
    # If small amounts, suggest investing it all (no need for buffer)
    if saving_eur <= 0:
        one_time = 0
    elif saving_eur < 500:
        one_time = saving_eur
    else:
        # keep a small buffer of 10% up to 1000 EUR
        buffer_amt = min(int(round(saving_eur * 0.10, -1)), 1000)
        one_time = max(0, saving_eur - buffer_amt)

    monthly_plan = max(0, monthly_saving_eur)

    return {"one_time_eur": int(one_time), "monthly_plan_eur": int(monthly_plan)}

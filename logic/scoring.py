def score_options(options, user):
    scored = []
    for opt in options:
        if user["horizon_years"] < opt["horizon_min"]:
            continue

        if user["esg"] is True and opt.get("esg", False) is False:
            # Wenn Nutzer ESG will, dann nicht-ESG Optionen abwerten oder rausfiltern.
            # Hier: rausfiltern fuer Klarheit.
            continue

        risk_score = 10 - abs(opt["risk"] - user["risk"])
        fee_score = 1 / opt["fee"]

        total_score = risk_score * 0.7 + fee_score * 0.3
        scored.append((opt, total_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

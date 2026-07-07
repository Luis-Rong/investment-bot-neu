from logic.scoring import score_options

# NOTE: these lock in the current (prototype) scoring behaviour based on the
# static "risk" integer. Phase 1 replaces that input with real volatility;
# these tests will be updated alongside that change.

OPTIONS = [
    {"id": "A", "name": "A", "risk": 8, "horizon_min": 7, "esg": False, "fee": 0.20},
    {"id": "B", "name": "B", "risk": 3, "horizon_min": 2, "esg": False, "fee": 0.30},
    {"id": "C", "name": "C", "risk": 7, "horizon_min": 5, "esg": True, "fee": 0.22},
]


def test_filters_options_below_horizon():
    user = {"horizon_years": 3, "risk": 5, "esg": False}
    scored = score_options(OPTIONS, user)
    ids = [opt["id"] for opt, _ in scored]
    assert "A" not in ids  # horizon_min 7 > 3
    assert "B" in ids


def test_esg_preference_filters_non_esg():
    user = {"horizon_years": 10, "risk": 7, "esg": True}
    scored = score_options(OPTIONS, user)
    ids = [opt["id"] for opt, _ in scored]
    assert ids == ["C"]  # only the ESG option survives


def test_results_sorted_by_score_desc():
    user = {"horizon_years": 10, "risk": 8, "esg": False}
    scored = score_options(OPTIONS, user)
    scores = [s for _, s in scored]
    assert scores == sorted(scores, reverse=True)


def test_closest_risk_match_scores_highest():
    user = {"horizon_years": 10, "risk": 8, "esg": False}
    scored = score_options(OPTIONS, user)
    assert scored[0][0]["id"] == "A"  # risk 8 matches user risk 8 exactly

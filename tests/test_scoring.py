import math

from logic.scoring import risk_to_target_volatility, score_options

# Options carry live-computed `volatility` + `sharpe` plus static
# `ter`/`horizon_min`/`esg` metadata.
OPTIONS = [
    # high vol, GOOD risk-adjusted return, cheap, non-ESG
    {"id": "A", "name": "A", "volatility": 0.18, "sharpe": 0.9, "ter": 0.05, "horizon_min": 7, "esg": False},
    # low vol, flat return
    {"id": "B", "name": "B", "volatility": 0.05, "sharpe": 0.0, "ter": 0.30, "horizon_min": 2, "esg": False},
    # mid vol, ESG
    {"id": "C", "name": "C", "volatility": 0.16, "sharpe": 0.7, "ter": 0.22, "horizon_min": 5, "esg": True},
    # high vol, TERRIBLE risk-adjusted return (the REIT-style trap)
    {"id": "D", "name": "D", "volatility": 0.19, "sharpe": 0.2, "ter": 0.13, "horizon_min": 5, "esg": False},
]


def test_risk_maps_to_target_volatility():
    assert risk_to_target_volatility(1) < risk_to_target_volatility(10)
    assert math.isclose(risk_to_target_volatility(1), 0.03)
    assert math.isclose(risk_to_target_volatility(10), 0.25)


def test_filters_options_below_horizon():
    user = {"horizon_years": 3, "risk": 5, "esg": False}
    ids = [opt["id"] for opt, _ in score_options(OPTIONS, user)]
    assert "A" not in ids  # horizon_min 7 > 3
    assert "B" in ids


def test_esg_preference_filters_non_esg():
    user = {"horizon_years": 10, "risk": 7, "esg": True}
    ids = [opt["id"] for opt, _ in score_options(OPTIONS, user)]
    assert ids == ["C"]


def test_results_sorted_by_score_desc():
    user = {"horizon_years": 10, "risk": 8, "esg": False}
    scores = [s for _, s in score_options(OPTIONS, user)]
    assert scores == sorted(scores, reverse=True)


def test_aggressive_user_prefers_return_quality_over_raw_volatility():
    # Risk 10: A and D are both ~high vol, but A has a far better Sharpe.
    # Return quality must rank A above the volatile-but-low-return D.
    user = {"horizon_years": 10, "risk": 10, "esg": False}
    scored = score_options(OPTIONS, user)
    ids = [opt["id"] for opt, _ in scored]
    assert ids[0] == "A"
    assert ids.index("A") < ids.index("D")


def test_conservative_user_prefers_low_volatility():
    # Risk 1: exceeding the tiny target vol is penalised hard, so the calm
    # option B wins even though it has the worst Sharpe/fee.
    user = {"horizon_years": 10, "risk": 1, "esg": False}
    scored = score_options(OPTIONS, user)
    assert scored[0][0]["id"] == "B"


def test_esg_fund_deprioritized_when_user_indifferent():
    user = {"horizon_years": 10, "risk": 7, "esg": False}
    scored = dict((opt["id"], s) for opt, s in score_options(OPTIONS, user))
    # C is ESG; without the tilt penalty its score would be 1.5 higher.
    # Compare against A (similar-quality non-ESG) — C must not outrank A.
    assert scored["A"] > scored["C"]


def test_option_without_metrics_is_skipped():
    opts = OPTIONS + [
        {
            "id": "X",
            "name": "X",
            "volatility": float("nan"),
            "sharpe": 1.0,
            "ter": 0.1,
            "horizon_min": 1,
            "esg": False,
        }
    ]
    user = {"horizon_years": 10, "risk": 5, "esg": False}
    ids = [opt["id"] for opt, _ in score_options(opts, user)]
    assert "X" not in ids

import math

from logic.scoring import risk_to_target_volatility, score_options

# Options carry live-computed `volatility` plus static `ter`/`horizon_min`/`esg`.
OPTIONS = [
    {"id": "A", "name": "A", "volatility": 0.20, "ter": 0.20, "horizon_min": 7, "esg": False},
    {"id": "B", "name": "B", "volatility": 0.05, "ter": 0.30, "horizon_min": 2, "esg": False},
    {"id": "C", "name": "C", "volatility": 0.16, "ter": 0.22, "horizon_min": 5, "esg": True},
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


def test_high_risk_user_prefers_high_volatility_option():
    user = {"horizon_years": 10, "risk": 10, "esg": False}
    scored = score_options(OPTIONS, user)
    assert scored[0][0]["id"] == "A"  # vol 0.20 is closest to the 0.25 target


def test_option_without_metrics_is_skipped():
    opts = OPTIONS + [
        {
            "id": "X",
            "name": "X",
            "volatility": float("nan"),
            "ter": 0.1,
            "horizon_min": 1,
            "esg": False,
        }
    ]
    user = {"horizon_years": 10, "risk": 5, "esg": False}
    ids = [opt["id"] for opt, _ in score_options(opts, user)]
    assert "X" not in ids

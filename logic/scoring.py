"""Score ETF options against a user profile.

Pure function: it reads metrics already attached to each option (notably
`volatility`, computed upstream from real prices in data_sources) plus the
static `ter`/`horizon_min`/`esg` metadata. No network here — this stays
deterministic and unit-testable.
"""

from __future__ import annotations

import math

# Map the user's 1..10 risk appetite to a target annualized volatility.
# risk 1 -> ~3% (very conservative), risk 10 -> ~25% (aggressive equity).
MIN_TARGET_VOL = 0.03
MAX_TARGET_VOL = 0.25


def risk_to_target_volatility(risk: int) -> float:
    """Linear map from a 1..10 risk score to a target annualized volatility."""
    risk = max(1, min(10, int(risk)))
    return MIN_TARGET_VOL + (risk - 1) / 9 * (MAX_TARGET_VOL - MIN_TARGET_VOL)


def _risk_score(option_vol: float, target_vol: float) -> float:
    """0..10, highest when the option's volatility matches the target."""
    # A full target-range mismatch (~22 vol points) costs ~10 points.
    return max(0.0, 10.0 - abs(option_vol - target_vol) * 40.0)


def _fee_score(ter: float) -> float:
    """0..10, highest for the cheapest funds (TER in percent, e.g. 0.20)."""
    return max(0.0, 10.0 - ter * 10.0)


def score_options(options, user):
    target_vol = risk_to_target_volatility(user["risk"])
    scored = []
    for opt in options:
        if user["horizon_years"] < opt["horizon_min"]:
            continue

        # If the user wants ESG, keep only ESG-labeled options for clarity.
        if user.get("esg") is True and opt.get("esg", False) is False:
            continue

        vol = opt.get("volatility")
        if vol is None or math.isnan(vol):
            continue  # no live metrics -> can't score responsibly

        risk_score = _risk_score(vol, target_vol)
        fee_score = _fee_score(opt["ter"])

        total_score = risk_score * 0.7 + fee_score * 0.3
        scored.append((opt, total_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

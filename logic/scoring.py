"""Score ETF options against a user profile.

Pure function: it reads metrics already attached to each option (`volatility`,
`sharpe`, computed upstream from real prices in data_sources) plus the static
`ter`/`horizon_min`/`esg` metadata. No network here — this stays deterministic
and unit-testable.

Scoring philosophy (three parts, weighted):

1. **Risk fit** — the user's 1..10 risk appetite maps to a target volatility.
   A fund scores highest near that target, penalised for missing it
   *asymmetrically*: overshooting the tolerance is punished hard, undershooting
   only mildly. (Scoring by raw closeness handed aggressive users the single most
   volatile asset, e.g. a low-return REIT; penalising overshoot only then let a
   near-cash fund with a flattering Sharpe crowd into a 30-year portfolio. The
   asymmetric penalty avoids both.)
2. **Return quality** — risk-adjusted return (Sharpe). This is what now favours
   higher-returning assets *within* the risk tolerance, so "I want high returns"
   surfaces strong equities, not just anything volatile.
3. **Fee** — cheaper is better (TER).

Weights are module-level constants so they're easy to tune.
"""

from __future__ import annotations

import math

# Map the user's 1..10 risk appetite to a target annualized volatility.
# risk 1 -> ~3% (very conservative), risk 10 -> ~25% (aggressive equity).
MIN_TARGET_VOL = 0.03
MAX_TARGET_VOL = 0.25

# Component weights (sum to 1.0).
W_RISK_FIT = 0.5
W_QUALITY = 0.25
W_FEE = 0.25

# Penalty (score points per unit of volatility) for missing the risk target.
# Asymmetric: exceeding the user's tolerance is the real harm (steep), but sitting
# far *below* it is also wrong — it's how a near-cash fund with a flattering
# Sharpe would otherwise sneak into an aggressive, 30-year portfolio. The mild
# under-penalty pulls each profile toward assets that actually fit its horizon.
OVER_VOL_PENALTY = 80.0
UNDER_VOL_PENALTY = 20.0
# Sharpe ratio at which the quality score saturates to 10/10.
SHARPE_FULL_MARKS = 1.25
# Points removed from ESG-badged funds when the user hasn't asked for ESG, so an
# indifferent user isn't steered into them (and doesn't get an ESG/plain duplicate).
ESG_TILT_PENALTY = 1.5


def risk_to_target_volatility(risk: int) -> float:
    """Linear map from a 1..10 risk score to a target annualized volatility."""
    risk = max(1, min(10, int(risk)))
    return MIN_TARGET_VOL + (risk - 1) / 9 * (MAX_TARGET_VOL - MIN_TARGET_VOL)


def _risk_fit_score(option_vol: float, target_vol: float) -> float:
    """0..10; highest near the target, penalised more for over- than undershoot."""
    if option_vol > target_vol:
        penalty = (option_vol - target_vol) * OVER_VOL_PENALTY
    else:
        penalty = (target_vol - option_vol) * UNDER_VOL_PENALTY
    return max(0.0, 10.0 - penalty)


def _quality_score(sharpe: float | None) -> float:
    """0..10 from the risk-adjusted return (Sharpe). Missing/negative -> 0."""
    if sharpe is None or (isinstance(sharpe, float) and math.isnan(sharpe)):
        return 0.0
    return max(0.0, min(10.0, sharpe / SHARPE_FULL_MARKS * 10.0))


def _fee_score(ter: float) -> float:
    """0..10, highest for the cheapest funds (TER in percent, e.g. 0.20)."""
    return max(0.0, 10.0 - ter * 10.0)


def score_options(options, user):
    target_vol = risk_to_target_volatility(user["risk"])
    wants_esg = user.get("esg") is True

    scored = []
    for opt in options:
        if user["horizon_years"] < opt["horizon_min"]:
            continue

        # If the user wants ESG, keep only ESG-labeled options for clarity.
        if wants_esg and opt.get("esg", False) is False:
            continue

        vol = opt.get("volatility")
        if vol is None or math.isnan(vol):
            continue  # no live metrics -> can't score responsibly

        total = (
            _risk_fit_score(vol, target_vol) * W_RISK_FIT
            + _quality_score(opt.get("sharpe")) * W_QUALITY
            + _fee_score(opt["ter"]) * W_FEE
        )

        # Don't steer an indifferent user toward ESG-badged funds (and avoid an
        # ESG/plain near-duplicate crowding the portfolio).
        if not wants_esg and opt.get("esg", False):
            total -= ESG_TILT_PENALTY

        scored.append((opt, total))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

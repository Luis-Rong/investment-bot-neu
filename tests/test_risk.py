import numpy as np
import pandas as pd

from data_sources.risk import (
    annualized_return,
    annualized_volatility,
    max_drawdown,
    sharpe_ratio,
)


def test_flat_prices_have_zero_volatility():
    prices = pd.Series([100.0] * 300)
    assert annualized_volatility(prices) == 0.0


def test_steady_growth_has_positive_return_and_low_drawdown():
    # 1% per step, monotonically increasing -> no drawdown.
    prices = pd.Series(100 * (1.01 ** np.arange(300)))
    assert annualized_return(prices) > 0
    assert max_drawdown(prices) == 0.0


def test_max_drawdown_captures_peak_to_trough():
    # Up to 200, down to 100 -> -50% drawdown.
    prices = pd.Series([100, 150, 200, 150, 100, 120])
    assert np.isclose(max_drawdown(prices), -0.5)


def test_volatility_scales_with_dispersion():
    calm = pd.Series(100 + np.tile([0, 1], 150))
    wild = pd.Series(100 + np.tile([0, 20], 150))
    assert annualized_volatility(wild) > annualized_volatility(calm)


def test_sharpe_is_finite_for_growing_series():
    rng = np.random.default_rng(0)
    prices = pd.Series(100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 300)))
    assert np.isfinite(sharpe_ratio(prices))


def test_too_short_series_returns_nan():
    prices = pd.Series([100.0])
    assert np.isnan(annualized_volatility(prices))

import numpy as np
import pandas as pd

from src.candidate_models import (
    DynamicJumpParams,
    GarchParams,
    ImpliedSurfaceParams,
    RegimeLabel,
    SVCJMomentParams,
    detect_market_regime,
    dynamic_jump_price,
    fit_implied_surface_multi_regime,
    garch_price,
    implied_surface_price,
    implied_surface_price_multi_regime,
    price_candidate_models,
    summarize_candidate_errors,
    summarize_model_params,
    summarize_regime_surfaces,
    svcj_moment_price,
)
from src.merton_jump import MertonParams, merton_price


def _option_frame():
    return pd.DataFrame(
        {
            "currency": ["BTC", "BTC"],
            "instrument_name": ["BTC-C", "BTC-P"],
            "option_type": ["call", "put"],
            "underlying_price": [100.0, 100.0],
            "strike": [100.0, 95.0],
            "time_to_maturity": [30 / 365.25, 30 / 365.25],
            "rate": [0.01, 0.01],
            "market_price_usd": [5.0, 4.0],
            "log_moneyness": [0.0, np.log(0.95)],
            "open_interest": [10.0, 10.0],
            "volume": [1.0, 1.0],
            "mark_iv_decimal": [0.50, 0.55],
            "valuation_datetime": pd.to_datetime(["2026-01-01", "2026-01-01"], utc=True),
        }
    )


def test_merton_accepts_vectorized_sigma():
    prices = merton_price(
        [100.0, 100.0],
        [100.0, 95.0],
        [30 / 365.25, 30 / 365.25],
        0.01,
        ["call", "put"],
        sigma=[0.3, 0.4],
        jump_intensity=[5.0, 7.0],
        jump_mean=-0.02,
        jump_vol=0.05,
    )
    assert np.asarray(prices).shape == (2,)
    assert np.all(np.asarray(prices) > 0.0)


def test_candidate_prices_are_positive_and_summarizable():
    options = _option_frame()
    surface = ImpliedSurfaceParams((0.5,) + (0.0,) * 11, tuple(str(i) for i in range(12)))
    svcj = SVCJMomentParams(
        v0=0.25,
        theta=0.20,
        kappa_v=2.0,
        variance_jump_mean=0.02,
        leverage_rho=-0.3,
        variance_multiplier=1.0,
        jump_intensity=5.0,
        jump_mean=-0.02,
        jump_vol=0.05,
    )
    dynamic = DynamicJumpParams(
        sigma=0.35,
        base_intensity=4.0,
        current_intensity=8.0,
        mean_reversion=3.0,
        jump_mean=-0.02,
        jump_vol=0.05,
    )
    garch = GarchParams(
        omega=1e-5,
        alpha=0.08,
        beta=0.90,
        last_variance=2e-4,
        mean_return=0.0,
        periods_per_year=365.25,
    )

    for values in [
        implied_surface_price(options, surface),
        svcj_moment_price(options, svcj),
        dynamic_jump_price(options, dynamic),
        garch_price(options, garch),
    ]:
        assert np.all(np.isfinite(values))
        assert np.all(values > 0.0)

    priced = price_candidate_models(
        options,
        historical_sigma=0.3,
        merton_params=MertonParams(0.3, 5.0, -0.02, 0.05),
        surface_params=surface,
        svcj_params=svcj,
        dynamic_params=dynamic,
        garch_params=garch,
    )
    summary = summarize_candidate_errors(priced)
    assert set(summary["model"]) == {
        "Black-Scholes",
        "Merton",
        "SVCJ proxy",
        "MR-ISVM surface",
        "Dynamic jump",
        "GARCH variance",
    }

    params = summarize_model_params(surface, svcj, dynamic, garch)
    assert {"MR-ISVM surface", "SVCJ proxy", "Dynamic jump", "GARCH variance"}.issubset(set(params["model"]))
    assert "xi" in set(params["parameter"])


def test_multi_regime_surface_prices_with_fallback():
    base = _option_frame()
    options = pd.concat([base] * 15, ignore_index=True)
    options["strike"] = np.linspace(80.0, 120.0, len(options))
    options["log_moneyness"] = np.log(options["strike"] / options["underlying_price"])
    options["mark_iv_decimal"] = 0.45 + 0.15 * np.abs(options["log_moneyness"])
    options["market_price_usd"] = np.linspace(2.0, 8.0, len(options))
    options["valuation_datetime"] = pd.date_range("2026-01-01", periods=len(options), freq="D", tz="UTC")

    returns = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-12-01", periods=45, freq="D", tz="UTC"),
            "log_return": [0.001] * 30 + [0.10, -0.12] * 7 + [0.08],
        }
    )
    returns.attrs["periods_per_year"] = 365.25

    assert detect_market_regime(returns, lookback=15, high_vol_thresh=0.10) == RegimeLabel.STRESS

    params = fit_implied_surface_multi_regime(options, returns, max_options_per_regime=12)
    prices = implied_surface_price_multi_regime(options, params)
    regimes = summarize_regime_surfaces(params)

    assert params.n_regimes >= 1
    assert len(params.active_params.coefficients) == 12
    assert np.all(np.isfinite(prices))
    assert np.all(prices > 0.0)
    assert not regimes.empty

import numpy as np
import pandas as pd

from src.candidate_models import (
    DynamicJumpParams,
    GarchParams,
    ImpliedSurfaceParams,
    SVCJMomentParams,
    dynamic_jump_price,
    garch_price,
    implied_surface_price,
    price_candidate_models,
    summarize_candidate_errors,
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
    surface = ImpliedSurfaceParams((0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), tuple(str(i) for i in range(9)))
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

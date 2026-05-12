"""Delta-hedging and stress-test utilities."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from .black_scholes import bs_greeks, bs_price
from .merton_jump import merton_greeks, merton_price, simulate_merton_paths


def make_gap_stress_path(
    historical_prices: Iterable[float],
    *,
    shock_log_return: float = -0.15,
    shock_index: int | None = None,
) -> np.ndarray:
    """Inject a one-time log-price gap into a historical path."""

    path = np.asarray(list(historical_prices), dtype=float)
    if path.ndim != 1 or len(path) < 3:
        raise ValueError("historical_prices must contain at least three observations")
    idx = shock_index if shock_index is not None else len(path) // 2
    idx = int(np.clip(idx, 1, len(path) - 1))
    stressed = path.copy()
    stressed[idx:] = stressed[idx:] * math.exp(shock_log_return)
    return stressed


def _model_price_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    model: str,
    merton_params: dict[str, float] | None,
) -> tuple[float, float]:
    if model == "black_scholes":
        price = float(bs_price(S, K, T, r, sigma, option_type))
        delta = float(bs_greeks(S, K, T, r, sigma, option_type)["delta"])
        return price, delta
    if model == "merton":
        params = merton_params or {}
        price = float(merton_price(S, K, T, r, option_type, sigma=sigma, **params))
        delta = float(merton_greeks(S, K, T, r, option_type, sigma=sigma, **params)["delta"])
        return price, delta
    raise ValueError("model must be 'black_scholes' or 'merton'")


def delta_hedge_short_option(
    path: Iterable[float],
    *,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "put",
    model: str = "black_scholes",
    merton_params: dict[str, float] | None = None,
) -> dict[str, float]:
    """Simulate discrete delta hedging for a short option.

    The hedge uses only the current spot at each rebalance, then realizes the
    next price move. The final P&L is hedge portfolio value minus option payoff.
    """

    spots = np.asarray(list(path), dtype=float)
    if spots.ndim != 1 or len(spots) < 2:
        raise ValueError("path must contain at least two spot observations")
    dt = T / (len(spots) - 1)
    premium, delta = _model_price_delta(spots[0], K, T, r, sigma, option_type, model, merton_params)
    cash = premium - delta * spots[0]
    total_turnover = abs(delta * spots[0])

    for i in range(1, len(spots) - 1):
        cash *= math.exp(r * dt)
        tau = max(T - i * dt, 1e-8)
        _, new_delta = _model_price_delta(spots[i], K, tau, r, sigma, option_type, model, merton_params)
        trade = (new_delta - delta) * spots[i]
        cash -= trade
        total_turnover += abs(trade)
        delta = new_delta

    cash *= math.exp(r * dt)
    final_spot = spots[-1]
    payoff = max(final_spot - K, 0.0) if option_type.lower().startswith("c") else max(K - final_spot, 0.0)
    hedge_value = delta * final_spot + cash
    pnl = hedge_value - payoff
    return {
        "initial_spot": float(spots[0]),
        "final_spot": float(final_spot),
        "premium": float(premium),
        "payoff": float(payoff),
        "hedge_value": float(hedge_value),
        "pnl": float(pnl),
        "absolute_pnl": float(abs(pnl)),
        "turnover": float(total_turnover),
        "model": model,
    }


def run_hedge_experiment(
    paths: np.ndarray,
    *,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "put",
    merton_params: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compare Black-Scholes and Merton hedge errors across simulated paths."""

    rows = []
    for path_id, path in enumerate(paths):
        for model in ("black_scholes", "merton"):
            rows.append(
                {
                    "path_id": path_id,
                    **delta_hedge_short_option(
                        path,
                        K=K,
                        T=T,
                        r=r,
                        sigma=sigma,
                        option_type=option_type,
                        model=model,
                        merton_params=merton_params,
                    ),
                }
            )
    return pd.DataFrame(rows)


def simulate_jump_hedge_experiment(
    S0: float,
    *,
    K: float,
    T: float,
    r: float,
    sigma: float,
    jump_intensity: float,
    jump_mean: float,
    jump_vol: float,
    steps: int = 60,
    paths: int = 250,
    seed: int = 4331,
    option_type: str = "put",
) -> pd.DataFrame:
    """Simulate jump paths and compare hedge error distributions."""

    simulated = simulate_merton_paths(
        S0,
        T,
        r=r,
        sigma=sigma,
        jump_intensity=jump_intensity,
        jump_mean=jump_mean,
        jump_vol=jump_vol,
        steps=steps,
        paths=paths,
        seed=seed,
    )
    return run_hedge_experiment(
        simulated,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type=option_type,
        merton_params={
            "jump_intensity": jump_intensity,
            "jump_mean": jump_mean,
            "jump_vol": jump_vol,
        },
    )


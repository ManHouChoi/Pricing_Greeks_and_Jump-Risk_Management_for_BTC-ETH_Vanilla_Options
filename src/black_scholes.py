"""Black-Scholes prices, Greeks, and implied volatility utilities."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from scipy.special import ndtr as _ndtr
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal envs
    _ndtr = None


SQRT_2PI = math.sqrt(2.0 * math.pi)


def _normal_cdf(x: Any) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    if _ndtr is not None:
        return _ndtr(values)
    erf = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf(values / math.sqrt(2.0)))


def _normal_pdf(x: Any) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    return np.exp(-0.5 * values * values) / SQRT_2PI


def _is_call(option_type: Any) -> np.ndarray:
    values = np.asarray(option_type)
    lowered = np.char.lower(values.astype(str))
    return np.char.startswith(lowered, "c")


def _maybe_scalar(values: np.ndarray) -> float | np.ndarray:
    return float(values) if values.ndim == 0 else values


def _broadcast_inputs(S: Any, K: Any, T: Any, r: Any, sigma: Any, q: Any):
    return np.broadcast_arrays(
        np.asarray(S, dtype=float),
        np.asarray(K, dtype=float),
        np.asarray(T, dtype=float),
        np.asarray(r, dtype=float),
        np.asarray(sigma, dtype=float),
        np.asarray(q, dtype=float),
    )


def d1_d2(S: Any, K: Any, T: Any, r: Any, sigma: Any, q: Any = 0.0):
    """Return Black-Scholes ``d1`` and ``d2`` arrays."""

    S, K, T, r, sigma, q = _broadcast_inputs(S, K, T, r, sigma, q)
    sqrt_T = np.sqrt(np.maximum(T, 1e-16))
    vol_sqrt_T = np.maximum(sigma * sqrt_T, 1e-16)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrt_T
    d2 = d1 - vol_sqrt_T
    return d1, d2


def bs_price(
    S: Any,
    K: Any,
    T: Any,
    r: Any,
    sigma: Any,
    option_type: Any = "call",
    q: Any = 0.0,
) -> float | np.ndarray:
    """Black-Scholes price with optional continuous carry/dividend yield ``q``."""

    S, K, T, r, sigma, q = _broadcast_inputs(S, K, T, r, sigma, q)
    calls = np.broadcast_to(_is_call(option_type), S.shape)
    call_intrinsic = np.maximum(S - K, 0.0)
    put_intrinsic = np.maximum(K - S, 0.0)
    intrinsic = np.where(calls, call_intrinsic, put_intrinsic)

    valid = (S > 0.0) & (K > 0.0) & (T > 0.0) & (sigma > 0.0)
    out = intrinsic.astype(float)
    if np.any(valid):
        d1, d2 = d1_d2(S[valid], K[valid], T[valid], r[valid], sigma[valid], q[valid])
        discounted_spot = S[valid] * np.exp(-q[valid] * T[valid])
        discounted_strike = K[valid] * np.exp(-r[valid] * T[valid])
        call = discounted_spot * _normal_cdf(d1) - discounted_strike * _normal_cdf(d2)
        put = discounted_strike * _normal_cdf(-d2) - discounted_spot * _normal_cdf(-d1)
        out[valid] = np.where(calls[valid], call, put)
    return _maybe_scalar(out)


def bs_greeks(
    S: Any,
    K: Any,
    T: Any,
    r: Any,
    sigma: Any,
    option_type: Any = "call",
    q: Any = 0.0,
) -> dict[str, float | np.ndarray]:
    """Return Delta, Gamma, Vega, Theta, and Rho.

    Vega and Rho are reported per unit change in volatility/rate, not per
    one-percentage-point change.
    """

    S, K, T, r, sigma, q = _broadcast_inputs(S, K, T, r, sigma, q)
    calls = np.broadcast_to(_is_call(option_type), S.shape)
    valid = (S > 0.0) & (K > 0.0) & (T > 0.0) & (sigma > 0.0)

    delta = np.where(calls, (S > K).astype(float), -(S < K).astype(float))
    gamma = np.full(S.shape, np.nan)
    vega = np.full(S.shape, np.nan)
    theta = np.full(S.shape, np.nan)
    rho = np.full(S.shape, np.nan)

    if np.any(valid):
        d1, d2 = d1_d2(S[valid], K[valid], T[valid], r[valid], sigma[valid], q[valid])
        sqrt_T = np.sqrt(T[valid])
        exp_q = np.exp(-q[valid] * T[valid])
        exp_r = np.exp(-r[valid] * T[valid])
        pdf_d1 = _normal_pdf(d1)
        nd1 = _normal_cdf(d1)
        nd2 = _normal_cdf(d2)
        nmd1 = _normal_cdf(-d1)
        nmd2 = _normal_cdf(-d2)

        delta_call = exp_q * nd1
        delta_put = exp_q * (nd1 - 1.0)
        gamma_v = exp_q * pdf_d1 / (S[valid] * sigma[valid] * sqrt_T)
        vega_v = S[valid] * exp_q * pdf_d1 * sqrt_T

        theta_common = -S[valid] * exp_q * pdf_d1 * sigma[valid] / (2.0 * sqrt_T)
        theta_call = (
            theta_common
            - r[valid] * K[valid] * exp_r * nd2
            + q[valid] * S[valid] * exp_q * nd1
        )
        theta_put = (
            theta_common
            + r[valid] * K[valid] * exp_r * nmd2
            - q[valid] * S[valid] * exp_q * nmd1
        )
        rho_call = K[valid] * T[valid] * exp_r * nd2
        rho_put = -K[valid] * T[valid] * exp_r * nmd2

        delta[valid] = np.where(calls[valid], delta_call, delta_put)
        gamma[valid] = gamma_v
        vega[valid] = vega_v
        theta[valid] = np.where(calls[valid], theta_call, theta_put)
        rho[valid] = np.where(calls[valid], rho_call, rho_put)

    return {
        "delta": _maybe_scalar(delta),
        "gamma": _maybe_scalar(gamma),
        "vega": _maybe_scalar(vega),
        "theta": _maybe_scalar(theta),
        "rho": _maybe_scalar(rho),
    }


def implied_volatility(
    market_price: Any,
    S: Any,
    K: Any,
    T: Any,
    r: Any,
    option_type: Any = "call",
    q: Any = 0.0,
    *,
    tol: float = 1e-8,
    max_iter: int = 120,
    max_vol: float = 10.0,
) -> float | np.ndarray:
    """Recover Black-Scholes implied volatility using robust bisection."""

    price, S, K, T, r, _, q = np.broadcast_arrays(
        np.asarray(market_price, dtype=float),
        np.asarray(S, dtype=float),
        np.asarray(K, dtype=float),
        np.asarray(T, dtype=float),
        np.asarray(r, dtype=float),
        np.asarray(0.0, dtype=float),
        np.asarray(q, dtype=float),
    )
    calls = np.broadcast_to(_is_call(option_type), price.shape)
    intrinsic = np.where(calls, np.maximum(S - K, 0.0), np.maximum(K - S, 0.0))
    valid = (price >= intrinsic - 1e-10) & (S > 0.0) & (K > 0.0) & (T > 0.0)
    out = np.full(price.shape, np.nan)
    if not np.any(valid):
        return _maybe_scalar(out)

    low = np.full(price.shape, 1e-8)
    high = np.full(price.shape, 1.0)
    for _ in range(12):
        model_high = np.asarray(bs_price(S, K, T, r, high, option_type, q))
        needs_higher = (model_high < price) & valid & (high < max_vol)
        if not np.any(needs_higher):
            break
        high = np.where(needs_higher, np.minimum(high * 2.0, max_vol), high)

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        model = np.asarray(bs_price(S, K, T, r, mid, option_type, q))
        low = np.where((model < price) & valid, mid, low)
        high = np.where((model >= price) & valid, mid, high)
        if np.nanmax(high[valid] - low[valid]) < tol:
            break

    out[valid] = 0.5 * (low[valid] + high[valid])
    return _maybe_scalar(out)


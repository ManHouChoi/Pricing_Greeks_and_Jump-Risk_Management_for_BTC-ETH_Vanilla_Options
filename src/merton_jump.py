"""Merton jump-diffusion pricing and jump-aware Greeks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .black_scholes import bs_greeks, bs_price


@dataclass(frozen=True)
class MertonParams:
    """Risk-neutral Merton jump-diffusion parameters."""

    sigma: float
    jump_intensity: float
    jump_mean: float
    jump_vol: float

    @property
    def kappa(self) -> float:
        return math.exp(self.jump_mean + 0.5 * self.jump_vol * self.jump_vol) - 1.0


def _maybe_scalar(values: np.ndarray) -> float | np.ndarray:
    return float(values) if values.ndim == 0 else values


def _poisson_weights(lam_t: np.ndarray, max_jumps: int | None, tol: float) -> list[np.ndarray]:
    if max_jumps is None:
        lam_max = float(np.nanmax(lam_t)) if lam_t.size else 0.0
        max_jumps = max(20, int(math.ceil(lam_max + 8.0 * math.sqrt(lam_max + 1.0))))
    weights: list[np.ndarray] = []
    weight = np.exp(-lam_t)
    cumulative = weight.copy()
    weights.append(weight)
    for n in range(1, max_jumps + 1):
        weight = weight * lam_t / n
        cumulative = cumulative + weight
        weights.append(weight)
        if np.nanmax(1.0 - cumulative) < tol:
            break
    return weights


def merton_price(
    S: Any,
    K: Any,
    T: Any,
    r: Any,
    option_type: Any = "call",
    *,
    sigma: float,
    jump_intensity: float,
    jump_mean: float,
    jump_vol: float,
    max_jumps: int | None = None,
    poisson_tol: float = 1e-10,
) -> float | np.ndarray:
    """Price a European vanilla option under Merton jump diffusion.

    The implementation conditions on the number of jumps. Conditional on ``n``
    jumps, log-price remains Gaussian with adjusted drift and variance, which
    can be priced as a generalized Black-Scholes contract with continuous carry.
    """

    S, K, T, r = np.broadcast_arrays(
        np.asarray(S, dtype=float),
        np.asarray(K, dtype=float),
        np.asarray(T, dtype=float),
        np.asarray(r, dtype=float),
    )
    if jump_intensity <= 1e-14:
        return bs_price(S, K, T, r, sigma, option_type)
    if sigma <= 0.0 or jump_intensity < 0.0 or jump_vol < 0.0:
        raise ValueError("sigma and jump_vol must be positive; jump_intensity must be non-negative")

    lam_t = jump_intensity * np.maximum(T, 0.0)
    kappa = math.exp(jump_mean + 0.5 * jump_vol * jump_vol) - 1.0
    weights = _poisson_weights(lam_t, max_jumps, poisson_tol)
    price = np.zeros_like(S, dtype=float)

    for n, weight in enumerate(weights):
        total_var = sigma * sigma + (n * jump_vol * jump_vol) / np.maximum(T, 1e-16)
        sigma_n = np.sqrt(np.maximum(total_var, 1e-16))
        drift_n = r - jump_intensity * kappa + (n * jump_mean + 0.5 * n * jump_vol * jump_vol) / np.maximum(T, 1e-16)
        q_n = r - drift_n
        price = price + weight * np.asarray(bs_price(S, K, T, r, sigma_n, option_type, q=q_n))
    return _maybe_scalar(price)


def merton_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    *,
    sigma: float,
    jump_intensity: float,
    jump_mean: float,
    jump_vol: float,
) -> dict[str, float]:
    """Finite-difference Greeks for the Merton model."""

    if jump_intensity <= 1e-14:
        return {key: float(value) for key, value in bs_greeks(S, K, T, r, sigma, option_type).items()}

    def price(s=S, k=K, t=T, rate=r, vol=sigma):
        return float(
            merton_price(
                s,
                k,
                max(t, 1e-8),
                rate,
                option_type,
                sigma=max(vol, 1e-8),
                jump_intensity=jump_intensity,
                jump_mean=jump_mean,
                jump_vol=jump_vol,
            )
        )

    h_s = max(1e-3, 1e-4 * S)
    h_v = max(1e-5, 1e-4 * sigma)
    h_r = 1e-4
    h_t = min(max(1e-5, 1e-4 * T), max(0.5 * T, 1e-5))

    base = price()
    up_s = price(s=S + h_s)
    dn_s = price(s=max(S - h_s, 1e-8))
    delta = (up_s - dn_s) / (2.0 * h_s)
    gamma = (up_s - 2.0 * base + dn_s) / (h_s * h_s)
    vega = (price(vol=sigma + h_v) - price(vol=max(sigma - h_v, 1e-8))) / (2.0 * h_v)
    rho = (price(rate=r + h_r) - price(rate=r - h_r)) / (2.0 * h_r)
    theta = (price(t=max(T - h_t, 1e-8)) - price(t=T + h_t)) / (2.0 * h_t)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def simulate_merton_paths(
    S0: float,
    T: float,
    *,
    r: float,
    sigma: float,
    jump_intensity: float,
    jump_mean: float,
    jump_vol: float,
    steps: int = 365,
    paths: int = 1000,
    seed: int | None = 4331,
) -> np.ndarray:
    """Simulate Merton jump-diffusion paths under the risk-neutral measure."""

    rng = np.random.default_rng(seed)
    dt = T / steps
    kappa = math.exp(jump_mean + 0.5 * jump_vol * jump_vol) - 1.0
    out = np.empty((paths, steps + 1), dtype=float)
    out[:, 0] = S0
    drift = (r - jump_intensity * kappa - 0.5 * sigma * sigma) * dt
    diffusion_scale = sigma * math.sqrt(dt)
    for i in range(1, steps + 1):
        z = rng.standard_normal(paths)
        n_jumps = rng.poisson(jump_intensity * dt, paths)
        jump_log = rng.normal(jump_mean * n_jumps, jump_vol * np.sqrt(n_jumps))
        out[:, i] = out[:, i - 1] * np.exp(drift + diffusion_scale * z + jump_log)
    return out

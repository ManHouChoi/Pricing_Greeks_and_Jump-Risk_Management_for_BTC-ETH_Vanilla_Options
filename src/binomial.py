"""Cox-Ross-Rubinstein binomial tree pricing."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd

from .black_scholes import bs_price


OptionStyle = Literal["european", "american"]


def crr_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    *,
    steps: int = 250,
    style: OptionStyle = "european",
    q: float = 0.0,
) -> float:
    """Price a vanilla option with a CRR binomial tree."""

    if T <= 0.0:
        return max(S - K, 0.0) if option_type.lower().startswith("c") else max(K - S, 0.0)
    if steps < 1:
        raise ValueError("steps must be positive")
    if S <= 0.0 or K <= 0.0 or sigma <= 0.0:
        raise ValueError("S, K, and sigma must be positive")

    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    growth = math.exp((r - q) * dt)
    p = (growth - d) / (u - d)
    if p < 0.0 or p > 1.0:
        raise ValueError("risk-neutral probability is outside [0, 1]")

    j = np.arange(steps + 1)
    terminal_spots = S * (u ** j) * (d ** (steps - j))
    is_call = option_type.lower().startswith("c")
    values = np.maximum(terminal_spots - K, 0.0) if is_call else np.maximum(K - terminal_spots, 0.0)
    discount = math.exp(-r * dt)

    for level in range(steps - 1, -1, -1):
        values = discount * (p * values[1 : level + 2] + (1.0 - p) * values[: level + 1])
        if style == "american":
            spots = S * (u ** np.arange(level + 1)) * (d ** (level - np.arange(level + 1)))
            exercise = np.maximum(spots - K, 0.0) if is_call else np.maximum(K - spots, 0.0)
            values = np.maximum(values, exercise)
    return float(values[0])


def convergence_table(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    *,
    steps_grid: tuple[int, ...] = (25, 50, 100, 250, 500, 1000),
    q: float = 0.0,
) -> pd.DataFrame:
    """Return CRR convergence diagnostics against Black-Scholes."""

    benchmark = float(bs_price(S, K, T, r, sigma, option_type, q))
    rows = []
    for steps in steps_grid:
        price = crr_price(S, K, T, r, sigma, option_type, steps=steps, q=q)
        rows.append(
            {
                "steps": steps,
                "crr_price": price,
                "black_scholes_price": benchmark,
                "absolute_error": abs(price - benchmark),
                "relative_error": abs(price - benchmark) / max(benchmark, 1e-12),
            }
        )
    return pd.DataFrame(rows)


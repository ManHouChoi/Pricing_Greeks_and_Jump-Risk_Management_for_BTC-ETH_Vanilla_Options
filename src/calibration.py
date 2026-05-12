"""Historical parameter estimation and market calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .black_scholes import bs_price
from .merton_jump import MertonParams, merton_price
from .preprocessing import compute_log_returns, detect_jumps

try:
    from scipy.optimize import minimize
except ModuleNotFoundError:  # pragma: no cover
    minimize = None


SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60


@dataclass(frozen=True)
class CalibrationResult:
    """Container for Merton calibration output."""

    params: MertonParams
    rmse: float
    mae: float
    n_options: int
    success: bool
    message: str


def _periods_per_year(timestamps_ms: pd.Series) -> float:
    diffs = timestamps_ms.sort_values().diff().dropna() / 1000.0
    median_seconds = float(diffs.median()) if not diffs.empty else 24 * 60 * 60
    return SECONDS_PER_YEAR / max(median_seconds, 1.0)


def estimate_historical_merton_params(
    index_history: pd.DataFrame,
    *,
    z_threshold: float = 3.0,
) -> tuple[MertonParams, pd.DataFrame]:
    """Estimate initial Merton parameters from historical index returns."""

    returns = detect_jumps(compute_log_returns(index_history), z_threshold=z_threshold)
    ppy = _periods_per_year(returns["timestamp"])
    years = max(len(returns) / ppy, 1e-12)
    non_jump_returns = returns.loc[~returns["is_jump"], "log_return"]
    jump_returns = returns.loc[returns["is_jump"], "log_return"]

    if len(non_jump_returns) < 3:
        non_jump_returns = returns["log_return"]
    sigma = float(non_jump_returns.std(ddof=1) * np.sqrt(ppy))
    sigma = float(np.clip(sigma, 0.03, 3.0))

    if len(jump_returns) == 0:
        jump_intensity = 0.25
        jump_mean = 0.0
        jump_vol = max(float(returns["log_return"].std(ddof=1) * 2.0), 0.01)
    else:
        jump_intensity = float(len(jump_returns) / years)
        jump_mean = float(jump_returns.mean())
        jump_vol = float(jump_returns.std(ddof=1)) if len(jump_returns) > 1 else abs(jump_mean)
        jump_vol = max(jump_vol, 0.01)

    params = MertonParams(
        sigma=sigma,
        jump_intensity=float(np.clip(jump_intensity, 1e-4, 250.0)),
        jump_mean=float(np.clip(jump_mean, -1.0, 1.0)),
        jump_vol=float(np.clip(jump_vol, 1e-4, 2.0)),
    )
    returns.attrs["periods_per_year"] = ppy
    return params, returns


def calibration_sample(
    options: pd.DataFrame,
    *,
    max_options: int = 250,
) -> pd.DataFrame:
    """Select a liquid, numerically stable calibration subset."""

    required = ["market_price_usd", "underlying_price", "strike", "time_to_maturity", "option_type"]
    sample = options.dropna(subset=required).copy()
    sample = sample[
        sample["market_price_usd"].gt(0.0)
        & sample["underlying_price"].gt(0.0)
        & sample["strike"].gt(0.0)
        & sample["time_to_maturity"].gt(0.0)
    ]
    if sample.empty:
        return sample
    sample["calibration_rank"] = sample[["open_interest", "volume"]].fillna(0.0).sum(axis=1)
    sample = sample.sort_values(["calibration_rank", "market_price_usd"], ascending=False)
    return sample.head(max_options).reset_index(drop=True)


def calibrate_merton_to_options(
    options: pd.DataFrame,
    initial: MertonParams,
    *,
    max_options: int = 250,
    rate_override: float | None = None,
) -> CalibrationResult:
    """Calibrate Merton parameters by minimizing weighted option price errors."""

    if minimize is None:  # pragma: no cover
        raise RuntimeError("scipy is required for calibration. Install requirements.txt first.")

    sample = calibration_sample(options, max_options=max_options)
    if sample.empty:
        return CalibrationResult(initial, np.nan, np.nan, 0, False, "No usable options")

    S = sample["underlying_price"].to_numpy(float)
    K = sample["strike"].to_numpy(float)
    T = sample["time_to_maturity"].to_numpy(float)
    market = sample["market_price_usd"].to_numpy(float)
    r = np.full(len(sample), rate_override, dtype=float) if rate_override is not None else sample["rate"].fillna(0.0).to_numpy(float)
    option_type = sample["option_type"].to_numpy(str)
    spread = sample.get("spread_usd", pd.Series(np.nan, index=sample.index)).fillna(0.0).to_numpy(float)
    weights = 1.0 / np.maximum(spread + 0.02 * market, 1.0)
    weights = weights / np.nanmean(weights)

    bounds = [(0.03, 3.0), (1e-4, 250.0), (-1.0, 1.0), (1e-4, 2.0)]
    x0 = np.array(
        [
            np.clip(initial.sigma, *bounds[0]),
            np.clip(initial.jump_intensity, *bounds[1]),
            np.clip(initial.jump_mean, *bounds[2]),
            np.clip(initial.jump_vol, *bounds[3]),
        ],
        dtype=float,
    )

    def objective(x: np.ndarray) -> float:
        sigma, jump_intensity, jump_mean, jump_vol = x
        model = np.asarray(
            merton_price(
                S,
                K,
                T,
                r,
                option_type,
                sigma=float(sigma),
                jump_intensity=float(jump_intensity),
                jump_mean=float(jump_mean),
                jump_vol=float(jump_vol),
            )
        )
        errors = (model - market) / np.maximum(market, 1.0)
        return float(np.nanmean(weights * errors * errors))

    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 300})
    params = MertonParams(
        sigma=float(result.x[0]),
        jump_intensity=float(result.x[1]),
        jump_mean=float(result.x[2]),
        jump_vol=float(result.x[3]),
    )
    model = np.asarray(
        merton_price(
            S,
            K,
            T,
            r,
            option_type,
            sigma=params.sigma,
            jump_intensity=params.jump_intensity,
            jump_mean=params.jump_mean,
            jump_vol=params.jump_vol,
        )
    )
    errors = model - market
    return CalibrationResult(
        params=params,
        rmse=float(np.sqrt(np.nanmean(errors * errors))),
        mae=float(np.nanmean(np.abs(errors))),
        n_options=len(sample),
        success=bool(result.success),
        message=str(result.message),
    )


def compare_model_errors(
    options: pd.DataFrame,
    merton_params: MertonParams,
    *,
    bs_sigma: float,
    rate_override: float | None = None,
) -> pd.DataFrame:
    """Add Black-Scholes and Merton model price/error columns to an option frame."""

    df = options.copy()
    r = rate_override if rate_override is not None else df["rate"].fillna(0.0)
    df["bs_model_usd"] = bs_price(
        df["underlying_price"],
        df["strike"],
        df["time_to_maturity"],
        r,
        bs_sigma,
        df["option_type"],
    )
    df["merton_model_usd"] = merton_price(
        df["underlying_price"],
        df["strike"],
        df["time_to_maturity"],
        r,
        df["option_type"],
        sigma=merton_params.sigma,
        jump_intensity=merton_params.jump_intensity,
        jump_mean=merton_params.jump_mean,
        jump_vol=merton_params.jump_vol,
    )
    df["bs_error_usd"] = df["bs_model_usd"] - df["market_price_usd"]
    df["merton_error_usd"] = df["merton_model_usd"] - df["market_price_usd"]
    df["abs_bs_error_usd"] = df["bs_error_usd"].abs()
    df["abs_merton_error_usd"] = df["merton_error_usd"].abs()
    return df


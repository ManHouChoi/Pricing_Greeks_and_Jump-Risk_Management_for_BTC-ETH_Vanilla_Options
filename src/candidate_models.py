"""Comparable crypto option model candidates beyond Black-Scholes and Merton."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .black_scholes import bs_price
from .calibration import calibration_sample
from .merton_jump import MertonParams, merton_price

try:
    from scipy.optimize import minimize
except ModuleNotFoundError:  # pragma: no cover
    minimize = None


@dataclass(frozen=True)
class ImpliedSurfaceParams:
    """Reduced MR-ISVM surface fitted to the current market regime."""

    coefficients: tuple[float, ...]
    feature_names: tuple[str, ...]
    sigma_floor: float = 0.05
    sigma_cap: float = 3.0
    fit_rmse: float = float("nan")


@dataclass(frozen=True)
class GarchParams:
    """GARCH(1, 1) state estimated from historical log returns."""

    omega: float
    alpha: float
    beta: float
    last_variance: float
    mean_return: float
    periods_per_year: float

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta


@dataclass(frozen=True)
class SVCJMomentParams:
    """Moment-matched stochastic-volatility-with-correlated-jumps proxy."""

    v0: float
    theta: float
    kappa_v: float
    variance_jump_mean: float
    leverage_rho: float
    variance_multiplier: float
    jump_intensity: float
    jump_mean: float
    jump_vol: float


@dataclass(frozen=True)
class DynamicJumpParams:
    """Merton-style jump model with mean-reverting jump intensity."""

    sigma: float
    base_intensity: float
    current_intensity: float
    mean_reversion: float
    jump_mean: float
    jump_vol: float


def _option_rates(options: pd.DataFrame, rate_override: float | None = None) -> pd.Series | np.ndarray:
    if rate_override is not None:
        return np.full(len(options), rate_override, dtype=float)
    return options["rate"].fillna(0.0)


def _surface_design(options: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    x = options.get("log_moneyness")
    if x is None:
        x = np.log(options["strike"].to_numpy(float) / options["underlying_price"].to_numpy(float))
    else:
        x = x.to_numpy(float)
    t = np.maximum(options["time_to_maturity"].to_numpy(float), 1e-8)
    put = options["option_type"].astype(str).str.lower().str.startswith("p").to_numpy(float)
    short = (t <= 30.0 / 365.25).astype(float)
    features = (
        "intercept",
        "log_moneyness",
        "abs_log_moneyness",
        "log_moneyness_sq",
        "sqrt_T",
        "T",
        "put",
        "put_x_log_moneyness",
        "short_maturity",
    )
    design = np.column_stack(
        [
            np.ones(len(options)),
            x,
            np.abs(x),
            x * x,
            np.sqrt(t),
            t,
            put,
            put * x,
            short,
        ]
    )
    return design, features


def fit_implied_surface(
    options: pd.DataFrame,
    *,
    max_options: int = 80,
    ridge: float = 1e-4,
) -> ImpliedSurfaceParams:
    """Fit a reduced MR-ISVM implied-volatility surface on a calibration sample."""

    sample = calibration_sample(options, max_options=max_options)
    sample = sample.dropna(subset=["mark_iv_decimal"]).copy()
    sample = sample[sample["mark_iv_decimal"].between(0.01, 4.0)]
    if sample.empty:
        raise ValueError("No usable options for implied-surface calibration")

    X, feature_names = _surface_design(sample)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = sample["mark_iv_decimal"].to_numpy(float)
    y = np.nan_to_num(y, nan=float(np.nanmedian(y)), posinf=float(np.nanmedian(y)), neginf=float(np.nanmedian(y)))
    liquidity = sample[["open_interest", "volume"]].fillna(0.0).sum(axis=1).to_numpy(float)
    liquidity = np.nan_to_num(liquidity, nan=0.0, posinf=0.0, neginf=0.0)
    weights = np.sqrt(1.0 + liquidity / max(np.nanmedian(liquidity + 1.0), 1.0))
    weights = np.clip(weights, 1.0, 10.0)
    Xw = X * weights[:, None]
    yw = y * weights
    penalty = ridge * np.eye(X.shape[1])
    penalty[0, 0] = 0.0
    xtx = np.einsum("ni,nj->ij", Xw, Xw)
    xty = np.einsum("ni,n->i", Xw, yw)
    try:
        coefs = np.linalg.solve(xtx + penalty, xty)
    except np.linalg.LinAlgError:
        coefs = np.linalg.lstsq(xtx + penalty, xty, rcond=None)[0]
    fitted = np.einsum("ij,j->i", X, coefs)
    rmse = float(np.sqrt(np.mean((fitted - y) ** 2)))
    return ImpliedSurfaceParams(tuple(float(v) for v in coefs), feature_names, fit_rmse=rmse)


def predict_implied_surface_iv(options: pd.DataFrame, params: ImpliedSurfaceParams) -> np.ndarray:
    """Predict current-regime implied volatility for each option row."""

    X, _ = _surface_design(options)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = np.einsum("ij,j->i", X, np.asarray(params.coefficients, dtype=float))
    return np.clip(sigma, params.sigma_floor, params.sigma_cap)


def implied_surface_price(
    options: pd.DataFrame,
    params: ImpliedSurfaceParams,
    *,
    rate_override: float | None = None,
) -> np.ndarray:
    """Price options by plugging the fitted regime surface into Black-Scholes."""

    sigma = predict_implied_surface_iv(options, params)
    return np.asarray(
        bs_price(
            options["underlying_price"],
            options["strike"],
            options["time_to_maturity"],
            _option_rates(options, rate_override),
            sigma,
            options["option_type"],
        )
    )


def fit_garch_params(return_frame: pd.DataFrame) -> GarchParams:
    """Estimate a Gaussian GARCH(1, 1) model from log returns."""

    returns = return_frame["log_return"].dropna().to_numpy(float)
    if len(returns) < 30:
        raise ValueError("At least 30 returns are required for GARCH estimation")
    ppy = float(return_frame.attrs.get("periods_per_year", 365.25))
    mean_return = float(np.mean(returns))
    centered = returns - mean_return
    sample_var = float(np.var(centered, ddof=1))

    def recursion(omega: float, alpha: float, beta: float) -> np.ndarray:
        h = np.empty_like(centered)
        prev = max(sample_var, 1e-12)
        for i, value in enumerate(centered):
            lagged_sq = centered[i - 1] ** 2 if i > 0 else sample_var
            prev = omega + alpha * lagged_sq + beta * prev
            h[i] = max(prev, 1e-12)
        return h

    if minimize is None:  # pragma: no cover
        alpha, beta = 0.06, 0.90
        omega = max(sample_var * (1.0 - alpha - beta), 1e-12)
    else:
        def objective(x: np.ndarray) -> float:
            omega, alpha, beta = x
            if alpha + beta >= 0.999:
                return 1e9 + 1e6 * (alpha + beta - 0.999) ** 2
            h = recursion(float(omega), float(alpha), float(beta))
            return float(0.5 * np.sum(np.log(h) + centered * centered / h))

        x0 = np.array([max(sample_var * 0.04, 1e-10), 0.06, 0.90], dtype=float)
        bounds = [(1e-12, max(sample_var, 1e-4)), (1e-5, 0.45), (1e-5, 0.995)]
        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
        omega, alpha, beta = (float(v) for v in result.x)
        if alpha + beta >= 0.999:
            beta = 0.998 - alpha

    h = recursion(omega, alpha, beta)
    return GarchParams(
        omega=float(omega),
        alpha=float(alpha),
        beta=float(beta),
        last_variance=float(h[-1]),
        mean_return=mean_return,
        periods_per_year=ppy,
    )


def garch_average_variance(params: GarchParams, T: np.ndarray | float) -> np.ndarray:
    """Forecast average annualized variance over each option horizon."""

    horizon = np.asarray(T, dtype=float)
    out = np.empty_like(horizon, dtype=float)
    phi = min(max(params.persistence, 1e-8), 0.999)
    long_var = params.omega / max(1.0 - phi, 1e-8)
    flat = horizon.reshape(-1)
    values = []
    for t in flat:
        n_steps = max(1, int(np.ceil(max(t, 1e-8) * params.periods_per_year)))
        powers = phi ** np.arange(1, n_steps + 1)
        forecasts = long_var + powers * (params.last_variance - long_var)
        values.append(float(np.mean(np.maximum(forecasts, 1e-12)) * params.periods_per_year))
    out.reshape(-1)[:] = values
    return out


def garch_price(
    options: pd.DataFrame,
    params: GarchParams,
    *,
    rate_override: float | None = None,
) -> np.ndarray:
    """GARCH option proxy using forecast conditional variance in Black-Scholes."""

    sigma = np.sqrt(garch_average_variance(params, options["time_to_maturity"].to_numpy(float)))
    return np.asarray(
        bs_price(
            options["underlying_price"],
            options["strike"],
            options["time_to_maturity"],
            _option_rates(options, rate_override),
            sigma,
            options["option_type"],
        )
    )


def _variance_state_from_returns(return_frame: pd.DataFrame, merton_params: MertonParams) -> SVCJMomentParams:
    returns = return_frame.copy()
    r = returns["log_return"].dropna()
    ppy = float(return_frame.attrs.get("periods_per_year", 365.25))
    rolling_var = r.rolling(20, min_periods=5).var() * ppy
    rolling_var = rolling_var.dropna()
    theta = float(np.clip(merton_params.sigma ** 2, 1e-6, 9.0))
    v0 = float(np.clip(rolling_var.iloc[-1] if not rolling_var.empty else theta, 0.05 * theta, 9.0))
    acf = float(rolling_var.autocorr(lag=1)) if len(rolling_var) > 10 else 0.75
    kappa_v = float(np.clip(-np.log(np.clip(acf, 0.05, 0.98)) * ppy, 0.25, 12.0))

    jump_returns = returns.loc[returns.get("is_jump", False), "log_return"]
    if len(jump_returns) > 0:
        variance_jump_mean = float(np.maximum(jump_returns.pow(2).mean() * ppy - theta, 0.05 * theta))
    else:
        variance_jump_mean = 0.05 * theta
    variance_jump_mean = float(np.clip(variance_jump_mean, 1e-8, 4.0 * theta))

    variance_change = rolling_var.diff().dropna()
    aligned_returns = r.reindex(variance_change.index)
    if len(variance_change) > 5 and aligned_returns.std(ddof=1) > 0 and variance_change.std(ddof=1) > 0:
        leverage_rho = float(np.corrcoef(aligned_returns, variance_change)[0, 1])
    else:
        leverage_rho = 0.0

    return SVCJMomentParams(
        v0=v0,
        theta=theta,
        kappa_v=kappa_v,
        variance_jump_mean=variance_jump_mean,
        leverage_rho=float(np.clip(leverage_rho, -0.95, 0.95)),
        variance_multiplier=1.0,
        jump_intensity=merton_params.jump_intensity,
        jump_mean=merton_params.jump_mean,
        jump_vol=merton_params.jump_vol,
    )


def svcj_average_variance(params: SVCJMomentParams, T: np.ndarray | float) -> np.ndarray:
    """Moment-match the average variance of an SVCJ process over the option life."""

    horizon = np.maximum(np.asarray(T, dtype=float), 1e-8)
    kappa = max(params.kappa_v, 1e-8)
    decay_avg = (1.0 - np.exp(-kappa * horizon)) / (kappa * horizon)
    base_avg = params.theta + (params.v0 - params.theta) * decay_avg
    jump_avg = (
        params.jump_intensity
        * params.variance_jump_mean
        * (horizon / kappa - (1.0 - np.exp(-kappa * horizon)) / (kappa * kappa))
        / horizon
    )
    return np.maximum((base_avg + jump_avg) * params.variance_multiplier, 1e-10)


def svcj_moment_price(
    options: pd.DataFrame,
    params: SVCJMomentParams,
    *,
    rate_override: float | None = None,
) -> np.ndarray:
    """Price options with a moment-matched SVCJ approximation."""

    avg_var = svcj_average_variance(params, options["time_to_maturity"].to_numpy(float))
    sigma_eff = np.sqrt(avg_var)
    return np.asarray(
        merton_price(
            options["underlying_price"],
            options["strike"],
            options["time_to_maturity"],
            _option_rates(options, rate_override),
            options["option_type"],
            sigma=sigma_eff,
            jump_intensity=params.jump_intensity,
            jump_mean=params.jump_mean,
            jump_vol=params.jump_vol,
        )
    )


def calibrate_svcj_moment_proxy(
    options: pd.DataFrame,
    return_frame: pd.DataFrame,
    merton_params: MertonParams,
    *,
    max_options: int = 80,
) -> SVCJMomentParams:
    """Calibrate a one-factor SVCJ variance multiplier to option prices."""

    base = _variance_state_from_returns(return_frame, merton_params)
    sample = calibration_sample(options, max_options=max_options)
    if sample.empty or minimize is None:
        return base

    market = sample["market_price_usd"].to_numpy(float)

    def objective(x: np.ndarray) -> float:
        params = replace(base, variance_multiplier=float(x[0]))
        model = svcj_moment_price(sample, params)
        errors = (model - market) / np.maximum(market, 1.0)
        return float(np.mean(errors * errors))

    result = minimize(objective, np.array([1.0]), method="L-BFGS-B", bounds=[(0.25, 4.0)])
    return replace(base, variance_multiplier=float(result.x[0]))


def _initial_dynamic_jump_params(return_frame: pd.DataFrame, merton_params: MertonParams) -> DynamicJumpParams:
    jumps = return_frame.get("is_jump", pd.Series(False, index=return_frame.index)).astype(bool)
    ppy = float(return_frame.attrs.get("periods_per_year", 365.25))
    unconditional = max(float(jumps.mean()), 1.0 / max(len(jumps), 1))
    recent_window = max(20, int(round(ppy / 12.0)))
    recent = max(float(jumps.tail(recent_window).mean()), unconditional)
    current_scale = np.clip(recent / unconditional, 0.75, 4.0)
    return DynamicJumpParams(
        sigma=merton_params.sigma,
        base_intensity=float(max(1e-4, 0.65 * merton_params.jump_intensity)),
        current_intensity=float(max(1e-4, merton_params.jump_intensity * current_scale)),
        mean_reversion=6.0,
        jump_mean=merton_params.jump_mean,
        jump_vol=merton_params.jump_vol,
    )


def dynamic_effective_intensity(params: DynamicJumpParams, T: np.ndarray | float) -> np.ndarray:
    """Average the mean-reverting intensity over the option horizon."""

    horizon = np.maximum(np.asarray(T, dtype=float), 1e-8)
    beta = max(params.mean_reversion, 1e-8)
    decay_avg = (1.0 - np.exp(-beta * horizon)) / (beta * horizon)
    return np.maximum(params.base_intensity + (params.current_intensity - params.base_intensity) * decay_avg, 1e-8)


def dynamic_jump_price(
    options: pd.DataFrame,
    params: DynamicJumpParams,
    *,
    rate_override: float | None = None,
) -> np.ndarray:
    """Price options under a Merton mixture with maturity-dependent intensity."""

    lambda_eff = dynamic_effective_intensity(params, options["time_to_maturity"].to_numpy(float))
    return np.asarray(
        merton_price(
            options["underlying_price"],
            options["strike"],
            options["time_to_maturity"],
            _option_rates(options, rate_override),
            options["option_type"],
            sigma=params.sigma,
            jump_intensity=lambda_eff,
            jump_mean=params.jump_mean,
            jump_vol=params.jump_vol,
        )
    )


def calibrate_dynamic_jump(
    options: pd.DataFrame,
    return_frame: pd.DataFrame,
    merton_params: MertonParams,
    *,
    max_options: int = 80,
) -> DynamicJumpParams:
    """Calibrate dynamic jump intensity parameters on the same option sample."""

    base = _initial_dynamic_jump_params(return_frame, merton_params)
    sample = calibration_sample(options, max_options=max_options)
    if sample.empty or minimize is None:
        return base
    market = sample["market_price_usd"].to_numpy(float)

    def objective(x: np.ndarray) -> float:
        sigma, base_intensity, current_intensity, mean_reversion = x
        if current_intensity < base_intensity:
            return 1e6 + (base_intensity - current_intensity) ** 2
        params = DynamicJumpParams(
            sigma=float(sigma),
            base_intensity=float(base_intensity),
            current_intensity=float(current_intensity),
            mean_reversion=float(mean_reversion),
            jump_mean=base.jump_mean,
            jump_vol=base.jump_vol,
        )
        model = dynamic_jump_price(sample, params)
        errors = (model - market) / np.maximum(market, 1.0)
        return float(np.mean(errors * errors))

    x0 = np.array([base.sigma, base.base_intensity, base.current_intensity, base.mean_reversion], dtype=float)
    bounds = [(0.03, 3.0), (1e-4, 250.0), (1e-4, 400.0), (0.25, 30.0)]
    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 250})
    return DynamicJumpParams(
        sigma=float(result.x[0]),
        base_intensity=float(result.x[1]),
        current_intensity=float(result.x[2]),
        mean_reversion=float(result.x[3]),
        jump_mean=base.jump_mean,
        jump_vol=base.jump_vol,
    )


def price_candidate_models(
    options: pd.DataFrame,
    *,
    historical_sigma: float,
    merton_params: MertonParams,
    surface_params: ImpliedSurfaceParams,
    svcj_params: SVCJMomentParams,
    dynamic_params: DynamicJumpParams,
    garch_params: GarchParams,
) -> pd.DataFrame:
    """Return long-form model prices and errors for the full option universe."""

    model_prices = {
        "Black-Scholes": np.asarray(
            bs_price(
                options["underlying_price"],
                options["strike"],
                options["time_to_maturity"],
                options["rate"].fillna(0.0),
                historical_sigma,
                options["option_type"],
            )
        ),
        "Merton": np.asarray(
            merton_price(
                options["underlying_price"],
                options["strike"],
                options["time_to_maturity"],
                options["rate"].fillna(0.0),
                options["option_type"],
                sigma=merton_params.sigma,
                jump_intensity=merton_params.jump_intensity,
                jump_mean=merton_params.jump_mean,
                jump_vol=merton_params.jump_vol,
            )
        ),
        "SVCJ proxy": svcj_moment_price(options, svcj_params),
        "MR-ISVM surface": implied_surface_price(options, surface_params),
        "Dynamic jump": dynamic_jump_price(options, dynamic_params),
        "GARCH variance": garch_price(options, garch_params),
    }

    rows = []
    base_cols = ["currency", "instrument_name", "option_type", "strike", "time_to_maturity", "log_moneyness"]
    market = options["market_price_usd"].to_numpy(float)
    for model, prices in model_prices.items():
        frame = options[base_cols].copy()
        frame["model"] = model
        frame["market_price_usd"] = market
        frame["model_price_usd"] = prices
        frame["error_usd"] = frame["model_price_usd"] - frame["market_price_usd"]
        frame["abs_error_usd"] = frame["error_usd"].abs()
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def summarize_candidate_errors(priced: pd.DataFrame) -> pd.DataFrame:
    """Compute comparable pricing diagnostics for all candidate models."""

    return (
        priced.groupby(["currency", "model"], as_index=False)
        .agg(
            options=("abs_error_usd", "size"),
            bias_usd=("error_usd", "mean"),
            mae_usd=("abs_error_usd", "mean"),
            median_ae_usd=("abs_error_usd", "median"),
            rmse_usd=("error_usd", lambda s: float(np.sqrt(np.mean(np.square(s))))),
            p90_ae_usd=("abs_error_usd", lambda s: float(s.quantile(0.90))),
        )
        .sort_values(["currency", "mae_usd"])
        .reset_index(drop=True)
    )

"""Comparable crypto option model candidates beyond Black-Scholes and Merton."""

from __future__ import annotations

from dataclasses import dataclass, replace

import enum
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
    xi: float = 0.5


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


# Model B - Market-Regime Implied Stochastic Volatility
def _surface_design(options: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    x = options.get("log_moneyness")
    if x is None:
        x = np.log(options["strike"].to_numpy(float) / options["underlying_price"].to_numpy(float))
    else:
        x = x.to_numpy(float)
    t = np.maximum(options["time_to_maturity"].to_numpy(float), 1e-8)
    put = options["option_type"].astype(str).str.lower().str.startswith("p").to_numpy(float)
    short = (t <= 30.0 / 365.25).astype(float)
    sqrt_t = np.sqrt(t)
    # Three extra terms added for MR-ISVM:
    #   log_moneyness_x_sqrt_T : SVI-style cross term - controls how skew
    #                             steepness varies with maturity (key for term
    #                             structure of the crypto smile).
    #   put_x_sqrt_T           : puts and calls have different term structures;
    #                             this lets the surface tilt the IV term slope
    #                             separately for each side.
    #   log_moneyness_cube     : cubic asymmetry - crypto OTM puts are more
    #                             expensive than the quadratic term alone can
    #                             capture, driving systematic under-pricing.
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
        "log_moneyness_x_sqrt_T",
        "put_x_sqrt_T",
        "log_moneyness_cube",
    )
    design = np.column_stack(
        [
            np.ones(len(options)),
            x,
            np.abs(x),
            x * x,
            sqrt_t,
            t,
            put,
            put * x,
            short,
            x * sqrt_t,          # SVI cross term
            put * sqrt_t,         # put-side term slope
            x * x * x,           # cubic skew asymmetry
        ]
    )
    return design, features


def fit_implied_surface(
    options: pd.DataFrame,
    *,
    max_options: int = 80,
    ridge: float = 1e-3,
) -> ImpliedSurfaceParams:
    """
    Fit a reduced MR-ISVM implied-volatility surface on a calibration sample.

    The surface is fitted directly in price space rather than treating implied
    volatility regression as the final objective.  The initial point is a flat
    smile at the median observed IV, and L-BFGS-B refines the coefficients using
    a vega-weighted relative RMSE objective.  This keeps the calibration aligned
    with the USD pricing-error metrics used in the report.
    """
    sample = calibration_sample(options, max_options=max_options)
    sample = sample.dropna(subset=["mark_iv_decimal"]).copy()
    sample = sample[sample["mark_iv_decimal"].between(0.01, 4.0)]
    if sample.empty:
        raise ValueError("No usable options for implied-surface calibration")

    X, feature_names = _surface_design(sample)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    S_arr   = sample["underlying_price"].to_numpy(float)
    K_arr   = sample["strike"].to_numpy(float)
    T_arr   = np.maximum(sample["time_to_maturity"].to_numpy(float), 1e-8)
    r_arr   = sample["rate"].fillna(0.0).to_numpy(float)
    opt_arr = sample["option_type"].to_numpy(object)
    T_days  = T_arr * 365.25

    market_price = sample["market_price_usd"].to_numpy(float)

    sigma_floor = 0.20
    sigma_cap   = 3.0

    # Approximate ATM sigma per option from mark_iv_decimal; fall back to
    # the sample median when the column is missing or NaN.
    sigma_atm = sample["mark_iv_decimal"].fillna(
        sample["mark_iv_decimal"].median()
    ).to_numpy(float)
    sigma_atm = np.clip(sigma_atm, 0.05, 3.0)

    # Black-Scholes vega approx S * phi(d1) * sqrtT  (independent of call/put)
    d1 = (
        np.log(S_arr / K_arr)
        + (r_arr + 0.5 * sigma_atm ** 2) * T_arr
    ) / (sigma_atm * np.sqrt(T_arr))
    vega_approx = S_arr * np.exp(-0.5 * d1 ** 2) / np.sqrt(2.0 * np.pi) * np.sqrt(T_arr)
    vega_approx = np.maximum(vega_approx, 1e-4)

    # Normalise so the average weight = 1 (keeps objective scale stable)
    weights = vega_approx / vega_approx.mean()
    weights = np.clip(weights, 0.05, 20.0)

    # ------------------------------------------------------------------
    # Helper: map coefficient vector -> clipped IV -> BS price array
    # ------------------------------------------------------------------
    def _price_from_coefs(c: np.ndarray) -> np.ndarray:
        sigma_iv = np.einsum("ij,j->i", X, c)
        adaptive_floor = np.maximum(
            sigma_floor,
            0.15 * np.sqrt(np.maximum(30.0 / T_days, 1.0)),
        )
        sigma_iv = np.clip(sigma_iv, adaptive_floor, sigma_cap)
        return np.asarray(bs_price(S_arr, K_arr, T_arr, r_arr, sigma_iv, opt_arr), dtype=float)

    # ------------------------------------------------------------------
    # Price-space objective (vega-weighted relative RMSE)
    # ------------------------------------------------------------------
    def objective(c: np.ndarray) -> float:
        model_px = _price_from_coefs(c)
        rel_err  = (model_px - market_price) / np.maximum(market_price, 0.5)
        return float(np.sqrt(np.average(rel_err ** 2, weights=weights)))

    # intercept = median mark IV of the calibration sample
    # all other coefficients = 0
    # This is a valid point in the price-space landscape (the surface
    # predicts a constant IV across all strikes and maturities equal to
    # the sample median) and avoids the IV-space local minimum that the
    # OLS warm-start introduces.
    atm_iv = float(np.median(sigma_atm))
    coefs_init = np.zeros(X.shape[1])
    coefs_init[0] = atm_iv          # intercept only

    if minimize is None:
        # scipy unavailable - return flat-smile solution
        model_px_flat = _price_from_coefs(coefs_init)
        price_rmse = float(np.sqrt(np.mean(
            ((model_px_flat - market_price) / np.maximum(market_price, 0.5)) ** 2
        )))
        return ImpliedSurfaceParams(
            tuple(float(v) for v in coefs_init),
            feature_names,
            sigma_floor=sigma_floor,
            fit_rmse=price_rmse,
        )

    # Ridge penalty applied inside the objective via a regularisation term
    # so the optimiser sees the penalised landscape from the first step.
    ridge_vec = np.full(X.shape[1], ridge)
    ridge_vec[0] = 0.0              # do not penalise the intercept

    def objective_regularised(c: np.ndarray) -> float:
        price_loss = objective(c)
        reg        = float(np.dot(ridge_vec, c ** 2))
        return price_loss + reg

    result = minimize(
        objective_regularised,
        coefs_init,
        method="L-BFGS-B",
        options={
            "maxiter": 2000,
            "ftol":    1e-12,
            "gtol":    1e-7,    # gradient tolerance
        },
    )
    coefs_opt = result.x

    # Final price RMSE (unpenalised, matches summarize_candidate_errors metric)
    model_px_final = _price_from_coefs(coefs_opt)
    price_rmse = float(np.sqrt(np.mean(
        ((model_px_final - market_price) / np.maximum(market_price, 0.5)) ** 2
    )))

    return ImpliedSurfaceParams(
        tuple(float(v) for v in coefs_opt),
        feature_names,
        sigma_floor=sigma_floor,
        fit_rmse=price_rmse,
    )


def predict_implied_surface_iv(options: pd.DataFrame, params: ImpliedSurfaceParams) -> np.ndarray:
    """Predict current-regime implied volatility for each option row.

    The adaptive floor replaces the fixed params.sigma_floor for very short
    maturities (T < 7 days). Near expiry, the polynomial surface can produce
    implausibly low IV in sparse strike regions; a time-scaled minimum of
    max(sigma_floor, 0.15 * sqrt(30/T_days)) prevents those artefacts from
    inflating pricing error on short-dated options without affecting longer
    maturities where the surface is well-identified.
    """

    X, _ = _surface_design(options)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = np.einsum("ij,j->i", X, np.asarray(params.coefficients, dtype=float))
    # Adaptive floor: tighter for short-dated options where the surface
    # extrapolates poorly, relaxed for longer maturities.
    T = np.maximum(options["time_to_maturity"].to_numpy(float), 1e-8)
    T_days = T * 365.25
    adaptive_floor = np.maximum(
        params.sigma_floor,
        0.15 * np.sqrt(np.maximum(30.0 / T_days, 1.0)),
    )
    return np.clip(sigma, adaptive_floor, params.sigma_cap)


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


class RegimeLabel(enum.Enum):
    """Discrete market regimes used by the multi-regime MR-ISVM surface."""

    CALM   = "calm"    # low realised vol, no strong directional bias
    STRESS = "stress"  # high realised vol or crash-skew environment
    TREND  = "trend"   # intermediate vol with a persistent directional drift

    def __str__(self) -> str:
        return self.value


# Regime-classification thresholds (tuned to BTC/ETH empirical distributions).
_HIGH_VOL_THRESH    : float = 0.65   # annualised realised vol above -> STRESS
_CRASH_SKEW_THRESH  : float = -0.7   # return skewness below         -> STRESS
_TREND_DRIFT_THRESH : float = 0.40   # |annualised drift| above      -> TREND
_MIN_OPTIONS_PER_REGIME: int = 25    # fewer usable options -> skip, use fallback


@dataclass(frozen=True)
class MultiRegimeSurfaceParams:
    """
    One ImpliedSurfaceParams per detected regime plus the active regime label.

    Attributes
    ----------
    regime_surfaces : dict[RegimeLabel, ImpliedSurfaceParams]
        Fitted surface for each regime that had enough options to calibrate.
        A missing regime falls back to the CALM surface.
    current_regime  : RegimeLabel
        The regime detected from the most-recent returns at calibration time.
    fallback_label  : RegimeLabel
        Which surface to use when the requested regime is absent (default CALM).
    """

    regime_surfaces: dict[RegimeLabel, ImpliedSurfaceParams]
    current_regime:  RegimeLabel
    fallback_label:  RegimeLabel = RegimeLabel.CALM

    @property
    def active_params(self) -> ImpliedSurfaceParams:
        """ImpliedSurfaceParams for the current regime (with CALM fallback)."""
        return self.regime_surfaces.get(
            self.current_regime,
            self.regime_surfaces[self.fallback_label],
        )

    def params_for(self, label: RegimeLabel) -> ImpliedSurfaceParams:
        """Return the surface for an explicit regime (with fallback)."""
        return self.regime_surfaces.get(label, self.active_params)

    @property
    def n_regimes(self) -> int:
        """Number of distinct regimes that were successfully calibrated."""
        return len(self.regime_surfaces)

    def __repr__(self) -> str:
        labels = ", ".join(r.value for r in self.regime_surfaces)
        return (
            f"MultiRegimeSurfaceParams("
            f"regimes=[{labels}], "
            f"current={self.current_regime.value}, "
            f"n_regimes={self.n_regimes})"
        )


def detect_market_regime(
    return_frame: pd.DataFrame,
    *,
    lookback: int = 20,
    high_vol_thresh: float    = _HIGH_VOL_THRESH,
    crash_skew_thresh: float  = _CRASH_SKEW_THRESH,
    trend_drift_thresh: float = _TREND_DRIFT_THRESH,
) -> RegimeLabel:
    """
    Classify the current market into one of three regimes from recent returns.

    Parameters
    ----------
    return_frame : pd.DataFrame
        Must contain a "log_return" column.  ``attrs["periods_per_year"]`` is
        used for annualisation (default 365.25 for crypto).
    lookback : int
        Number of most-recent observations used for classification (default 30).
    high_vol_thresh : float
        Annualised realised-vol threshold above which the regime is STRESS.
    crash_skew_thresh : float
        Return skewness threshold below which the regime is STRESS.
    trend_drift_thresh : float
        |Annualised drift| threshold above which the regime is TREND.

    Returns
    -------
    RegimeLabel
        Priority: STRESS > TREND > CALM.
    """
    returns = return_frame["log_return"].dropna()
    if len(returns) < max(lookback // 2, 5):
        return RegimeLabel.CALM   # not enough history - safest default

    ppy    = float(return_frame.attrs.get("periods_per_year", 365.25))
    recent = returns.iloc[-lookback:]

    realized_vol     = float(recent.std(ddof=1)) * np.sqrt(ppy)
    skewness         = float(recent.skew())
    annualized_drift = float(recent.mean()) * ppy

    # STRESS: high vol or crash-type negative skew
    if realized_vol > high_vol_thresh or skewness < crash_skew_thresh:
        return RegimeLabel.STRESS

    # TREND: strong directional bias
    if abs(annualized_drift) > trend_drift_thresh:
        return RegimeLabel.TREND

    return RegimeLabel.CALM


def _label_option_regimes(
    options: pd.DataFrame,
    return_frame: pd.DataFrame,
    *,
    regime_window: int = 30,
) -> pd.Series:
    """
    Assign a RegimeLabel to every option row from the returns that preceded it.

    If options have no parseable date information, every row receives the
    current (most-recent) regime, so the multi-regime fit degrades gracefully
    to a single-regime calibration.

    Returns
    -------
    pd.Series[RegimeLabel]  - same index as `options`.
    """
    valid_returns = return_frame.dropna(subset=["log_return"]).copy()
    ppy = float(return_frame.attrs.get("periods_per_year", 365.25))

    # Build a date -> regime mapping over the return time series.
    if "datetime" in valid_returns.columns:
        ret_dates = pd.to_datetime(valid_returns["datetime"], utc=True, errors="coerce")
    elif "timestamp" in valid_returns.columns:
        ret_dates = pd.to_datetime(valid_returns["timestamp"], unit="ms", utc=True, errors="coerce")
    elif isinstance(valid_returns.index, pd.DatetimeIndex):
        ret_dates = pd.Series(pd.to_datetime(valid_returns.index, utc=True), index=valid_returns.index)
    else:
        current = detect_market_regime(return_frame)
        return pd.Series(current, index=options.index)

    date_mask = pd.Series(ret_dates, index=valid_returns.index).notna()
    valid_returns = valid_returns.loc[date_mask]
    ret_dates = pd.Series(ret_dates, index=valid_returns.index)
    if valid_returns.empty:
        current = detect_market_regime(return_frame)
        return pd.Series(current, index=options.index)

    date_to_regime: dict[pd.Timestamp, RegimeLabel] = {}
    log_returns = valid_returns["log_return"].to_numpy(float)

    for i, dt in enumerate(ret_dates):
        if i < regime_window // 2:
            label = RegimeLabel.CALM
        else:
            window_ret = log_returns[max(0, i - regime_window): i]
            tmp = pd.DataFrame({"log_return": window_ret})
            tmp.attrs["periods_per_year"] = ppy
            label = detect_market_regime(tmp, lookback=len(tmp))
        date_to_regime[dt] = label

    if not date_to_regime:
        current = detect_market_regime(return_frame)
        return pd.Series(current, index=options.index)

    sorted_dates = sorted(date_to_regime.keys())

    current = detect_market_regime(return_frame)

    def _nearest(opt_date: pd.Timestamp) -> RegimeLabel:
        if pd.isna(opt_date):
            return current
        idx = np.searchsorted(sorted_dates, opt_date, side="right") - 1
        return date_to_regime[sorted_dates[max(idx, 0)]]

    # Try to extract an observation date from the options DataFrame.
    opt_dates: pd.Series | None = None
    for col in ("valuation_datetime", "datetime", "date", "timestamp", "valuation_timestamp", "expiration_datetime", "expiry_date"):
        if col in options.columns:
            try:
                if col.endswith("timestamp") and pd.api.types.is_numeric_dtype(options[col]):
                    opt_dates = pd.to_datetime(options[col], unit="ms", utc=True, errors="coerce")
                else:
                    opt_dates = pd.to_datetime(options[col], utc=True, errors="coerce")
                break
            except Exception:
                pass
    if opt_dates is None and isinstance(options.index, pd.DatetimeIndex):
        opt_dates = pd.Series(options.index, index=options.index)

    if opt_dates is None:
        return pd.Series(current, index=options.index)

    return opt_dates.map(_nearest)


def fit_implied_surface_multi_regime(
    options: pd.DataFrame,
    return_frame: pd.DataFrame,
    *,
    max_options_per_regime: int = 60,
    ridge: float = 1e-3,
    regime_window: int = 30,
) -> MultiRegimeSurfaceParams:
    """
    Fit one ImpliedSurfaceParams per detected market regime (multi-regime MR-ISVM).

    This is the regime-aware companion to ``fit_implied_surface``.  It:

    1. Detects the current regime via ``detect_market_regime``.
    2. Labels every option row with the regime that prevailed on its date via
       ``_label_option_regimes``.
    3. Calls the **existing** ``fit_implied_surface`` independently for each
       regime slice that has >= ``_MIN_OPTIONS_PER_REGIME`` usable rows.
    4. Guarantees the CALM regime always has a surface (falls back to fitting
       on all options if the CALM slice is too small).

    Parameters
    ----------
    options : pd.DataFrame
        Full option universe for one currency.
    return_frame : pd.DataFrame
        Historical log-returns for the same currency.
    max_options_per_regime : int
        Forwarded to ``fit_implied_surface`` for each regime slice.
    ridge : float
        Ridge penalty forwarded to ``fit_implied_surface``.
    regime_window : int
        Rolling lookback (in return observations) for per-date labelling.

    Returns
    -------
    MultiRegimeSurfaceParams
        Call ``.active_params`` to get the ImpliedSurfaceParams for the current
        regime - fully compatible with ``price_candidate_models``.
    """
    current_regime = detect_market_regime(return_frame)
    regime_labels  = _label_option_regimes(options, return_frame, regime_window=regime_window)

    regime_surfaces: dict[RegimeLabel, ImpliedSurfaceParams] = {}

    for regime in RegimeLabel:
        slice_opts = options[regime_labels == regime].copy()
        usable = slice_opts.dropna(subset=["mark_iv_decimal"])
        usable = usable[usable["mark_iv_decimal"].between(0.01, 4.0)]

        if len(usable) < _MIN_OPTIONS_PER_REGIME:
            continue   # not enough data; this regime will use the CALM fallback

        try:
            regime_surfaces[regime] = fit_implied_surface(
                slice_opts,
                max_options=max_options_per_regime,
                ridge=ridge,
            )
        except Exception:
            continue   # degenerate slice - skip silently

    # Always guarantee a CALM surface as the universal fallback.
    if RegimeLabel.CALM not in regime_surfaces:
        try:
            regime_surfaces[RegimeLabel.CALM] = fit_implied_surface(
                options,
                max_options=max_options_per_regime,
                ridge=ridge,
            )
        except Exception as exc:
            raise RuntimeError(
                "MR-ISVM multi-regime: could not fit even the fallback CALM surface. "
                "Ensure the options DataFrame contains usable mark_iv_decimal values."
            ) from exc

    return MultiRegimeSurfaceParams(
        regime_surfaces=regime_surfaces,
        current_regime=current_regime,
        fallback_label=RegimeLabel.CALM,
    )


def predict_implied_surface_iv_multi_regime(
    options: pd.DataFrame,
    params: MultiRegimeSurfaceParams,
    *,
    regime_override: RegimeLabel | None = None,
) -> np.ndarray:
    """
    Predict IV using the regime-specific surface.

    Delegates to the existing ``predict_implied_surface_iv`` after selecting
    the correct ImpliedSurfaceParams for the active (or overridden) regime.

    Parameters
    ----------
    regime_override : RegimeLabel or None
        Force prediction from a specific regime's surface (e.g. for stress
        testing).  When None (default), the current regime is used.
    """
    surface = params.params_for(regime_override) if regime_override is not None else params.active_params
    return predict_implied_surface_iv(options, surface)


def implied_surface_price_multi_regime(
    options: pd.DataFrame,
    params: MultiRegimeSurfaceParams,
    *,
    rate_override: float | None = None,
    regime_override: RegimeLabel | None = None,
) -> np.ndarray:
    """
    Price options using the regime-aware MR-ISVM surface.

    Drop-in multi-regime replacement for ``implied_surface_price``.  Selects
    the active regime's ImpliedSurfaceParams then delegates to the existing
    ``implied_surface_price``, preserving all IV-clamping and adaptive-floor
    logic exactly.

    Parameters
    ----------
    rate_override : float or None
        Overrides the "rate" column for all options when supplied.
    regime_override : RegimeLabel or None
        Force a specific regime surface (e.g. for scenario analysis).
    """
    surface = params.params_for(regime_override) if regime_override is not None else params.active_params
    return implied_surface_price(options, surface, rate_override=rate_override)


def summarize_regime_surfaces(params: MultiRegimeSurfaceParams) -> pd.DataFrame:
    """
    Tidy DataFrame comparing the fitted surfaces across all calibrated regimes.

    Columns: regime, n_coefficients, fit_rmse, sigma_floor, sigma_cap, active.
    Purely for inspection - no effect on pricing.
    """
    rows = []
    for label, surface in params.regime_surfaces.items():
        rows.append({
            "regime":         label.value,
            "n_coefficients": len(surface.coefficients),
            "fit_rmse":       round(surface.fit_rmse, 6),
            "sigma_floor":    round(surface.sigma_floor, 4),
            "sigma_cap":      round(surface.sigma_cap, 4),
            "active":         label == params.current_regime,
        })
    return (
        pd.DataFrame(rows)
        .sort_values("regime")
        .reset_index(drop=True)
    )

# Model C
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

# Model A
def _variance_state_from_returns(
    return_frame: pd.DataFrame,
    merton_params: MertonParams,
) -> SVCJMomentParams:
    r = return_frame["log_return"].dropna()
    ppy = float(return_frame.attrs.get("periods_per_year", 365.25))
    dt_approx = 1.0 / ppy

    rolling_var = r.rolling(20, min_periods=5).var() * ppy
    rolling_var = rolling_var.dropna()

    theta = float(np.clip(merton_params.sigma ** 2, 1e-6, 9.0))
    v0 = float(np.clip(
        rolling_var.iloc[-1] if not rolling_var.empty else theta,
        0.05 * theta, 9.0,
    ))

    acf = float(rolling_var.autocorr(lag=1)) if len(rolling_var) > 10 else 0.75
    kappa_v = float(np.clip(-np.log(np.clip(acf, 0.05, 0.98)) * ppy, 0.25, 12.0))

    jump_returns = return_frame.loc[
        return_frame.get("is_jump", pd.Series(False, index=return_frame.index)).astype(bool),
        "log_return",
    ]
    if len(jump_returns) > 0:
        variance_jump_mean = float(np.maximum(
            jump_returns.pow(2).mean() * ppy - theta,
            0.05 * theta,
        ))
    else:
        variance_jump_mean = 0.05 * theta
    variance_jump_mean = float(np.clip(variance_jump_mean, 1e-8, 4.0 * theta))

    if len(rolling_var) > 10:
        dv = rolling_var.diff().dropna()
        mean_v = float(rolling_var.mean())
        denom = np.sqrt(max(mean_v, 1e-8) * dt_approx * 20.0)
        xi = float(np.clip(dv.std() / denom, 0.05, 3.0))
    else:
        xi = 0.5

    variance_change = rolling_var.diff().dropna()
    aligned_returns = r.reindex(variance_change.index)
    if (
        len(variance_change) > 5
        and aligned_returns.std(ddof=1) > 0
        and variance_change.std(ddof=1) > 0
    ):
        leverage_rho = float(np.corrcoef(aligned_returns, variance_change)[0, 1])
    else:
        leverage_rho = -0.5

    return SVCJMomentParams(
        v0=v0,
        theta=theta,
        kappa_v=kappa_v,
        xi=float(np.clip(xi, 0.05, 3.0)),
        leverage_rho=float(np.clip(leverage_rho, -0.95, 0.95)),
        variance_jump_mean=variance_jump_mean,
        variance_multiplier=1.0,
        jump_intensity=merton_params.jump_intensity,
        jump_mean=merton_params.jump_mean,
        jump_vol=merton_params.jump_vol,
    )

def svcj_average_variance(
    params: SVCJMomentParams,
    T: np.ndarray | float,
) -> np.ndarray:
    """
    Compute the risk-neutral expected average variance E[1/T integral0T vt dt]
    for an SVCJ process, using the closed-form moment from the affine
    Heston + jump framework.

    Formula (Gatheral 2006, Ch. 2 + jump compensator)
    --------------------------------------------------
    For the CIR variance process with jumps:

        E[vt] = theta + (v0 - theta) e^{-kappat}  +  lambda mu_v / kappa (1 - e^{-kappat})

    Integrating over [0, T] and dividing by T:

        avg_var = theta
                 + (v0 - theta) (1 - e^{-kappaT}) / (kappaT)
                 + lambda mu_v / kappa [1 - (1 - e^{-kappaT})/(kappaT)]

    The vol-of-vol (xi) and leverage (leverage_rho) do not enter the
    *first* moment of the variance, but they do affect the variance of
    realised variance (the vol-of-vol risk premium).  We add a small
    convexity correction term  +1/2 xi^2 sigma / kappa  (derived from the second
    moment) so that higher xi produces a higher effective average variance,
    consistent with the SVCJ implied-vol surface being wider than Heston.

    Finally, variance_multiplier scales the result (calibrated to prices).
    """
    horizon = np.maximum(np.asarray(T, dtype=float), 1e-8)
    kappa   = max(params.kappa_v, 1e-8)
    lam     = max(params.jump_intensity, 0.0)
    mu_v    = max(params.variance_jump_mean, 0.0)
    xi      = params.xi
    v0      = params.v0
    theta   = params.theta

    # mean-reversion decay averaged over [0, T]
    decay_avg = (1.0 - np.exp(-kappa * horizon)) / (kappa * horizon)

    # CIR mean path (no jumps)
    base_avg = theta + (v0 - theta) * decay_avg

    # variance-jump contribution to the mean
    # lambda mu_v / kappa * [1 - (1 - e^{-kappaT})/(kappaT)]
    jump_avg = lam * mu_v / kappa * (1.0 - decay_avg)

    # vol-of-vol convexity correction:  +1/2 xi^2 * avg_var / kappa
    # This is the leading term of Var(realised vol) that shifts the
    # risk-neutral expectation of realised variance upward.
    base_for_correction = np.maximum(base_avg + jump_avg, 1e-10)
    xi_correction = 0.5 * xi ** 2 * base_for_correction / kappa

    avg_var = (base_avg + jump_avg + xi_correction) * params.variance_multiplier
    return np.maximum(avg_var, 1e-10)


def svcj_moment_price(
    options: pd.DataFrame,
    params: SVCJMomentParams,
    *,
    rate_override: float | None = None,
) -> np.ndarray:
    """
    Price options with a moment-matched SVCJ approximation.

    The effective diffusive volatility is taken from
    ``svcj_average_variance`` (which now properly accounts for xi, the
    variance-jump mean, and a convexity correction).  Jump risk is then
    layered on top via the Merton series, using the calibrated jump
    parameters - exactly the same structure as the original, but now
    with a better sigma_eff.
    """
    avg_var   = svcj_average_variance(params, options["time_to_maturity"].to_numpy(float))
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
    """
    Calibrate the SVCJ moment proxy to option prices.

    The price proxy depends on the average variance path, so the option
    calibration only refines parameters that enter that path directly:
    ``variance_multiplier`` and ``xi``. ``leverage_rho`` is still estimated
    from historical return/variance co-movement, but it is not optimized here
    because this moment-matched proxy does not use it in the pricing equation.
    """
    base = _variance_state_from_returns(return_frame, merton_params)
    sample = calibration_sample(options, max_options=max_options)
    if sample.empty or minimize is None:
        return base

    market = sample["market_price_usd"].to_numpy(float)
    weights = 1.0 / np.maximum(market, 0.5)

    def objective(x: np.ndarray) -> float:
        vm, xi = float(x[0]), float(x[1])
        params = replace(
            base,
            variance_multiplier=vm,
            xi=xi,
        )
        model = svcj_moment_price(sample, params)
        errors = (model - market) / np.maximum(market, 0.5)
        return float(np.sqrt(np.average(errors ** 2, weights=weights)))

    x0 = np.array([base.variance_multiplier, base.xi], dtype=float)
    bounds = [(0.25, 4.0), (0.05, 3.0)]
    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 300, "ftol": 1e-8},
    )
    vm_opt, xi_opt = (float(v) for v in result.x)
    return replace(
        base,
        variance_multiplier=vm_opt,
        xi=float(np.clip(xi_opt, 0.05, 3.0)),
    )

# Model D
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

# Summarizing the Parameter Values
def summarize_model_params(
    surface_params: ImpliedSurfaceParams,
    svcj_params: SVCJMomentParams,
    dynamic_params: DynamicJumpParams,
    garch_params: GarchParams,
) -> pd.DataFrame:
    """Return a tidy DataFrame of every calibrated parameter across all four candidate models.

    Each row is one parameter.  Columns are:
        model       : human-readable model label (matches price_candidate_models keys)
        parameter   : parameter name
        value       : calibrated scalar value
        description : plain-English explanation of what the parameter controls

    This is intended purely for inspection and reporting - it has no effect
    on pricing.  Call it after all four models have been calibrated and pass
    the resulting params objects directly.

    Example
    -------
    >>> param_table = summarize_model_params(surface_params, svcj_params,
    ...                                      dynamic_params, garch_params)
    >>> param_table.to_string(index=False)
    """

    rows: list[dict] = []

    # ------------------------------------------------------------------
    # Model B - MR-ISVM implied surface
    # ------------------------------------------------------------------
    _surface_descriptions = {
        "intercept":            "ATM IV level (baseline implied volatility)",
        "log_moneyness":        "linear skew slope (tilts smile left/right)",
        "abs_log_moneyness":    "symmetric smile width (V-shape lift)",
        "log_moneyness_sq":     "quadratic smile curvature (bowl depth)",
        "sqrt_T":               "term-structure level (IV change with sqrtT)",
        "T":                    "term-structure curvature (IV change with T)",
        "put":                  "put vs call IV offset at ATM",
        "put_x_log_moneyness":  "put-side skew slope (asymmetric tilt)",
        "short_maturity":       "short-dated IV premium (<=30 day options)",
        "log_moneyness_x_sqrt_T": "SVI cross-term: skew steepness vs maturity",
        "put_x_sqrt_T":         "put-side term slope (separate from calls)",
        "log_moneyness_cube":   "cubic skew asymmetry (OTM put wing steepness)",
    }
    for name, coef in zip(surface_params.feature_names, surface_params.coefficients):
        rows.append({
            "model":       "MR-ISVM surface",
            "parameter":   name,
            "value":       round(coef, 6),
            "description": _surface_descriptions.get(name, "surface coefficient"),
        })
    rows.append({
        "model":       "MR-ISVM surface",
        "parameter":   "sigma_floor",
        "value":       round(surface_params.sigma_floor, 4),
        "description": "minimum predicted IV (crypto lower bound)",
    })
    rows.append({
        "model":       "MR-ISVM surface",
        "parameter":   "sigma_cap",
        "value":       round(surface_params.sigma_cap, 4),
        "description": "maximum predicted IV (outlier cap)",
    })
    rows.append({
        "model":       "MR-ISVM surface",
        "parameter":   "fit_rmse",
        "value":       round(surface_params.fit_rmse, 6),
        "description": "in-sample relative price RMSE from surface calibration",
    })

    # ------------------------------------------------------------------
    # Model A - SVCJ moment proxy
    # ------------------------------------------------------------------
    _svcj_descriptions = {
        "v0":                 "initial variance v0 (spot variance at calibration date)",
        "theta":              "long-run variance theta (variance mean-reversion target)",
        "kappa_v":            "variance mean-reversion speed kappa (per year; higher = faster pull to theta)",
        "variance_jump_mean": "mean variance jump size mu_v (Exp scale; adds to v on each jump)",
        "leverage_rho":       "spot-vol correlation rho (negative = leverage effect / skew)",
        "variance_multiplier":"overall variance scale factor (calibrated to option prices)",
        "jump_intensity":     "jump arrival rate lambda (jumps per year)",
        "jump_mean":          "mean log spot-price jump mu_S (negative = crash bias)",
        "jump_vol":           "std of log spot-price jump sigma_S (uncertainty in jump size)",
    }
    svcj_field_order = [
        "v0", "theta", "kappa_v", "variance_jump_mean",
        "leverage_rho", "variance_multiplier",
        "jump_intensity", "jump_mean", "jump_vol",
    ]
    for field in svcj_field_order:
        val = getattr(svcj_params, field, None)
        if val is None:
            continue
        rows.append({
            "model":       "SVCJ proxy",
            "parameter":   field,
            "value":       round(float(val), 6),
            "description": _svcj_descriptions.get(field, "SVCJ parameter"),
        })
    if hasattr(svcj_params, "xi"):
        rows.append({
            "model":       "SVCJ proxy",
            "parameter":   "xi",
            "value":       round(float(svcj_params.xi), 6),
            "description": "vol-of-vol xi (controls variance-of-variance / smile width)",
        })

    # ------------------------------------------------------------------
    # Model D - Dynamic jump
    # ------------------------------------------------------------------
    _dj_descriptions = {
        "sigma":             "diffusive volatility sigma (non-jump component)",
        "base_intensity":    "long-run jump intensity lambda (unconditional arrival rate per year)",
        "current_intensity": "current jump intensity lambda_t (elevated after recent jumps)",
        "mean_reversion":    "intensity mean-reversion speed beta (per year; how fast lambda_t -> lambda)",
        "jump_mean":         "mean log spot-price jump mu_S",
        "jump_vol":          "std of log spot-price jump sigma_S",
    }
    for field, desc in _dj_descriptions.items():
        val = getattr(dynamic_params, field, None)
        if val is None:
            continue
        rows.append({
            "model":       "Dynamic jump",
            "parameter":   field,
            "value":       round(float(val), 6),
            "description": desc,
        })

    # ------------------------------------------------------------------
    # Model C - GARCH variance
    # ------------------------------------------------------------------
    _garch_descriptions = {
        "omega":           "GARCH constant omega (unconditional variance floor)",
        "alpha":           "ARCH coefficient alpha (weight on last squared return)",
        "beta":            "GARCH coefficient beta (weight on last variance)",
        "last_variance":   "conditional variance at last observation h_T",
        "mean_return":     "estimated mean log-return (subtracted before fitting)",
        "periods_per_year":"trading periods per year used for annualisation",
    }
    for field, desc in _garch_descriptions.items():
        val = getattr(garch_params, field, None)
        if val is None:
            continue
        rows.append({
            "model":       "GARCH variance",
            "parameter":   field,
            "value":       round(float(val), 6),
            "description": desc,
        })
    # Derived quantity - persistence is a property, not a stored field
    rows.append({
        "model":       "GARCH variance",
        "parameter":   "persistence (alpha + beta)",
        "value":       round(float(garch_params.persistence), 6),
        "description": "GARCH persistence (< 1 required for stationarity; crypto approx 0.95-0.99)",
    })
    long_var = garch_params.omega / max(1.0 - garch_params.persistence, 1e-8)
    rows.append({
        "model":       "GARCH variance",
        "parameter":   "long_run_variance (omega / (1 - alpha - beta))",
        "value":       round(float(long_var), 6),
        "description": "unconditional variance implied by GARCH parameters",
    })

    return pd.DataFrame(rows, columns=["model", "parameter", "value", "description"])

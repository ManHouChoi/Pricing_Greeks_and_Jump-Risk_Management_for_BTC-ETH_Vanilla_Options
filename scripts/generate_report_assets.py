"""Generate report figures, tables, and diagnostics.

This script is intentionally deterministic apart from the live Deribit snapshot
it consumes. It is the reproducible bridge between the quant code in ``src/``
and the written report in ``reports/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.binomial import convergence_table, crr_price
from src.black_scholes import bs_greeks, bs_price
from src.calibration import (
    calibrate_merton_to_options,
    compare_model_errors,
    estimate_historical_merton_params,
)
from src.candidate_models import (
    calibrate_dynamic_jump,
    calibrate_svcj_moment_proxy,
    dynamic_jump_price,
    fit_garch_params,
    fit_implied_surface,
    garch_price,
    implied_surface_price,
    price_candidate_models,
    summarize_candidate_errors,
    svcj_moment_price,
)
from src.data_deribit import DeribitClient, fetch_market_snapshot
from src.merton_jump import merton_greeks, merton_price
from src.plots import (
    plot_hedge_errors,
    plot_model_errors,
    plot_return_jumps,
    plot_vol_smile,
    pricing_error_summary,
    set_project_style,
)
from src.preprocessing import combine_currency_frames, filter_liquid_options, normalize_option_chain
from src.risk import simulate_jump_hedge_experiment


FIG_DIR = ROOT / "reports" / "figures"
TAB_DIR = ROOT / "reports" / "tables"
JUMP_Z_THRESHOLD = 4.0
CALIBRATION_OPTIONS = 80


def _write_table(name: str, frame: pd.DataFrame) -> None:
    frame.to_csv(TAB_DIR / f"{name}.csv", index=False)
    (TAB_DIR / f"{name}.md").write_text(frame.to_markdown(index=False, floatfmt=".6g"))


def _load_market_data():
    client = DeribitClient(cache_dir=ROOT / "data" / "cache", ttl_seconds=300)
    snapshot = fetch_market_snapshot(("BTC", "ETH"), client=client, use_cache=True)

    option_frames = []
    raw_counts = {}
    usable_counts = {}
    for currency, payload in snapshot.items():
        options = normalize_option_chain(payload["book"], payload["instruments"])
        options["currency"] = currency
        filtered = filter_liquid_options(options)
        raw_counts[currency] = len(options)
        usable_counts[currency] = len(filtered)
        option_frames.append(filtered)
    return snapshot, combine_currency_frames(option_frames), raw_counts, usable_counts


def _fit_models(snapshot, options_all):
    historical_params = {}
    calibration_results = {}
    return_frames = {}
    priced_frames = []

    for currency, payload in snapshot.items():
        params, returns = estimate_historical_merton_params(payload["index"], z_threshold=JUMP_Z_THRESHOLD)
        historical_params[currency] = params
        return_frames[currency] = returns

        currency_options = options_all[options_all["currency"] == currency].copy()
        result = calibrate_merton_to_options(
            currency_options,
            params,
            max_options=CALIBRATION_OPTIONS,
        )
        calibration_results[currency] = result
        priced_frames.append(compare_model_errors(currency_options, result.params, bs_sigma=params.sigma))

    return historical_params, calibration_results, return_frames, combine_currency_frames(priced_frames)


def _plot_core_figures(options_all, return_frames, priced_all, historical_params, calibration_results):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
    for ax, currency in zip(axes, ["BTC", "ETH"]):
        plot_return_jumps(
            return_frames[currency],
            ax=ax,
            title=f"{currency} robust jump detection, Deribit index returns",
        )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_return_jump_detection.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, currency in zip(axes, ["BTC", "ETH"]):
        plot_vol_smile(
            options_all[options_all["currency"] == currency],
            ax=ax,
            title=f"{currency} Deribit mark-IV smile",
        )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_deribit_iv_smiles.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, currency in zip(axes, ["BTC", "ETH"]):
        plot_model_errors(
            priced_all[priced_all["currency"] == currency],
            ax=ax,
            title=f"{currency} pricing error: Black-Scholes vs Merton",
        )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_pricing_error_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    convergence_frames = []
    for currency, marker in [("BTC", "o"), ("ETH", "s")]:
        spot = float(options_all.loc[options_all["currency"] == currency, "underlying_price"].median())
        sigma = historical_params[currency].sigma
        convergence = convergence_table(
            spot,
            spot,
            30 / 365.25,
            0.01,
            sigma,
            "call",
            steps_grid=(25, 50, 100, 250, 500, 1000),
        )
        convergence["currency"] = currency
        convergence_frames.append(convergence)
        ax.plot(convergence["steps"], convergence["absolute_error"], marker=marker, label=currency)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("CRR convergence toward Black-Scholes, 30-day ATM call")
    ax.set_xlabel("Tree steps, log scale")
    ax.set_ylabel("Absolute pricing error, USD log scale")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_btm_convergence.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    m_grid = np.linspace(0.75, 1.25, 81)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for row, currency in enumerate(["BTC", "ETH"]):
        spot = float(options_all.loc[options_all["currency"] == currency, "underlying_price"].median())
        sigma = historical_params[currency].sigma
        strikes = spot / m_grid
        for option_type in ["call", "put"]:
            greeks = bs_greeks(spot, strikes, 30 / 365.25, 0.01, sigma, option_type)
            axes[row, 0].plot(np.log(strikes / spot), greeks["delta"], label=option_type)
            axes[row, 1].plot(np.log(strikes / spot), greeks["gamma"], label=option_type)
        axes[row, 0].set_title(f"{currency} Delta profile")
        axes[row, 1].set_title(f"{currency} Gamma profile")
        axes[row, 0].set_ylabel(currency)
    for ax in axes.flat:
        ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("log(K / S)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_greek_profiles.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    return pd.concat(convergence_frames, ignore_index=True)


def _hedge_analysis(options_all, calibration_results):
    hedge_frames = []
    for currency in ["BTC", "ETH"]:
        spot = float(options_all.loc[options_all["currency"] == currency, "underlying_price"].median())
        params = calibration_results[currency].params
        hedges = simulate_jump_hedge_experiment(
            spot,
            K=0.95 * spot,
            T=30 / 365.25,
            r=0.01,
            sigma=params.sigma,
            jump_intensity=params.jump_intensity,
            jump_mean=params.jump_mean,
            jump_vol=params.jump_vol,
            steps=30,
            paths=100,
            seed=4331 if currency == "BTC" else 4332,
            option_type="put",
        )
        hedges["currency"] = currency
        hedge_frames.append(hedges)
    hedges_all = pd.concat(hedge_frames, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for ax, currency in zip(axes, ["BTC", "ETH"]):
        plot_hedge_errors(
            hedges_all[hedges_all["currency"] == currency],
            ax=ax,
            title=f"{currency} 30-day 5% OTM put hedge error",
        )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_jump_hedge_error_distribution.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    return hedges_all.groupby(["currency", "model"]).agg(
        mean_pnl=("pnl", "mean"),
        median_pnl=("pnl", "median"),
        std_pnl=("pnl", "std"),
        mean_abs_pnl=("absolute_pnl", "mean"),
        p05=("pnl", lambda s: s.quantile(0.05)),
        p95=("pnl", lambda s: s.quantile(0.95)),
        mean_turnover=("turnover", "mean"),
    ).reset_index()


def _return_diagnostics(snapshot, return_frames):
    rows = []
    for currency in ["BTC", "ETH"]:
        returns = return_frames[currency]
        ppy = returns.attrs["periods_per_year"]
        r = returns["log_return"]
        jb = stats.jarque_bera(r)
        rows.append(
            {
                "Currency": currency,
                "Observations": len(r),
                "Annualized mean": float(r.mean() * ppy),
                "Annualized vol": float(r.std(ddof=1) * np.sqrt(ppy)),
                "Skew": float(stats.skew(r, bias=False)),
                "Excess kurtosis": float(stats.kurtosis(r, fisher=True, bias=False)),
                "Jarque-Bera statistic": float(jb.statistic),
                "Jarque-Bera p-value": float(jb.pvalue),
                "Jump share": float(returns["is_jump"].mean()),
                "Index rows": len(snapshot[currency]["index"]),
            }
        )
    return pd.DataFrame(rows)


def _pricing_error_tests(priced_all):
    rows = []
    for currency in ["BTC", "ETH"]:
        frame = priced_all[priced_all["currency"] == currency].dropna(
            subset=["abs_bs_error_usd", "abs_merton_error_usd"]
        )
        improvement = frame["abs_bs_error_usd"] - frame["abs_merton_error_usd"]
        positive = int((improvement > 0).sum())
        t_test = stats.ttest_1samp(improvement, 0.0, alternative="greater")
        try:
            wilcoxon = stats.wilcoxon(improvement, alternative="greater", zero_method="wilcox")
            wilcoxon_stat = float(wilcoxon.statistic)
            wilcoxon_p = float(wilcoxon.pvalue)
        except ValueError:
            wilcoxon_stat = np.nan
            wilcoxon_p = np.nan
        sign = stats.binomtest(positive, len(improvement), 0.5, alternative="greater")
        rows.append(
            {
                "Currency": currency,
                "Paired options": len(improvement),
                "Mean MAE improvement USD": float(improvement.mean()),
                "Median MAE improvement USD": float(improvement.median()),
                "Share improved": positive / len(improvement),
                "Paired t-stat": float(t_test.statistic),
                "Paired t-test p": float(t_test.pvalue),
                "Wilcoxon statistic": wilcoxon_stat,
                "Wilcoxon p": wilcoxon_p,
                "Sign-test p": float(sign.pvalue),
            }
        )
    return pd.DataFrame(rows)


def _sensitivity_analysis(options_all, calibration_results):
    rows = []
    lambda_multipliers = [0.0, 0.5, 1.0, 1.5, 2.0]
    jump_vol_multipliers = [0.5, 1.0, 1.5, 2.0]

    for currency in ["BTC", "ETH"]:
        spot = float(options_all.loc[options_all["currency"] == currency, "underlying_price"].median())
        params = calibration_results[currency].params
        strike = 0.95 * spot
        bs_put = float(bs_price(spot, strike, 30 / 365.25, 0.01, params.sigma, "put"))
        for lam_mult in lambda_multipliers:
            for vol_mult in jump_vol_multipliers:
                price = float(
                    merton_price(
                        spot,
                        strike,
                        30 / 365.25,
                        0.01,
                        "put",
                        sigma=params.sigma,
                        jump_intensity=params.jump_intensity * lam_mult,
                        jump_mean=params.jump_mean,
                        jump_vol=params.jump_vol * vol_mult,
                    )
                )
                rows.append(
                    {
                        "Currency": currency,
                        "lambda_multiplier": lam_mult,
                        "jump_vol_multiplier": vol_mult,
                        "BS put USD": bs_put,
                        "Merton put USD": price,
                        "Jump premium USD": price - bs_put,
                    }
                )

    sensitivity = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for ax, currency in zip(axes, ["BTC", "ETH"]):
        pivot = sensitivity[sensitivity["Currency"] == currency].pivot(
            index="lambda_multiplier",
            columns="jump_vol_multiplier",
            values="Jump premium USD",
        )
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="mako", cbar_kws={"label": "USD"}, ax=ax)
        ax.set_title(f"{currency} 30-day 5% OTM put jump premium")
        ax.set_xlabel("Jump-volatility multiplier")
        ax.set_ylabel("Jump-intensity multiplier")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_jump_parameter_sensitivity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return sensitivity


def _candidate_analysis(options_all, return_frames, historical_params, calibration_results):
    priced_frames = []
    parameter_rows = []
    stress_rows = []

    for currency in ["BTC", "ETH"]:
        options = options_all[options_all["currency"] == currency].copy()
        merton_params = calibration_results[currency].params
        surface_params = fit_implied_surface(options, max_options=CALIBRATION_OPTIONS)
        svcj_params = calibrate_svcj_moment_proxy(
            options,
            return_frames[currency],
            merton_params,
            max_options=CALIBRATION_OPTIONS,
        )
        dynamic_params = calibrate_dynamic_jump(
            options,
            return_frames[currency],
            merton_params,
            max_options=CALIBRATION_OPTIONS,
        )
        garch_params = fit_garch_params(return_frames[currency])

        priced_frames.append(
            price_candidate_models(
                options,
                historical_sigma=historical_params[currency].sigma,
                merton_params=merton_params,
                surface_params=surface_params,
                svcj_params=svcj_params,
                dynamic_params=dynamic_params,
                garch_params=garch_params,
            )
        )

        parameter_rows.extend(
            [
                {
                    "Currency": currency,
                    "Model": "Black-Scholes",
                    "Calibration object": "Historical log-return variance",
                    "Pricing method": "Closed-form diffusion formula",
                    "Key fitted state": f"sigma={historical_params[currency].sigma:.4f}",
                },
                {
                    "Currency": currency,
                    "Model": "Merton",
                    "Calibration object": "Weighted option-price error",
                    "Pricing method": "Poisson mixture of Black-Scholes prices",
                    "Key fitted state": (
                        f"sigma={merton_params.sigma:.4f}; lambda={merton_params.jump_intensity:.2f}; "
                        f"muJ={merton_params.jump_mean:.4f}; deltaJ={merton_params.jump_vol:.4f}"
                    ),
                },
                {
                    "Currency": currency,
                    "Model": "SVCJ proxy",
                    "Calibration object": "Historical variance state plus option variance multiplier",
                    "Pricing method": "Moment-matched stochastic variance with Merton jumps",
                    "Key fitted state": (
                        f"v0={svcj_params.v0:.4f}; theta={svcj_params.theta:.4f}; "
                        f"kappaV={svcj_params.kappa_v:.2f}; vJump={svcj_params.variance_jump_mean:.4f}; "
                        f"rho={svcj_params.leverage_rho:.3f}; mult={svcj_params.variance_multiplier:.3f}"
                    ),
                },
                {
                    "Currency": currency,
                    "Model": "MR-ISVM surface",
                    "Calibration object": "Current-regime Deribit mark-IV cross section",
                    "Pricing method": "Black-Scholes with fitted regime IV surface",
                    "Key fitted state": f"surface_rmse_iv={surface_params.fit_rmse:.4f}; features={len(surface_params.coefficients)}",
                },
                {
                    "Currency": currency,
                    "Model": "Dynamic jump",
                    "Calibration object": "Weighted option-price error with mean-reverting lambda",
                    "Pricing method": "Merton mixture with horizon-dependent effective intensity",
                    "Key fitted state": (
                        f"sigma={dynamic_params.sigma:.4f}; baseLambda={dynamic_params.base_intensity:.2f}; "
                        f"currentLambda={dynamic_params.current_intensity:.2f}; beta={dynamic_params.mean_reversion:.2f}"
                    ),
                },
                {
                    "Currency": currency,
                    "Model": "GARCH variance",
                    "Calibration object": "Historical conditional return variance",
                    "Pricing method": "Black-Scholes with GARCH forecast average variance",
                    "Key fitted state": (
                        f"omega={garch_params.omega:.6g}; alpha={garch_params.alpha:.3f}; "
                        f"beta={garch_params.beta:.3f}; persistence={garch_params.persistence:.3f}"
                    ),
                },
            ]
        )

        spot = float(options["underlying_price"].median())
        stress_option = pd.DataFrame(
            [
                {
                    "currency": currency,
                    "instrument_name": f"{currency}-30D-95P-STRESS",
                    "option_type": "put",
                    "underlying_price": spot,
                    "strike": 0.95 * spot,
                    "time_to_maturity": 30.0 / 365.25,
                    "rate": 0.01,
                    "market_price_usd": np.nan,
                    "log_moneyness": np.log(0.95),
                    "open_interest": 0.0,
                    "volume": 0.0,
                    "mark_iv_decimal": np.nan,
                }
            ]
        )
        stress_prices = {
            "Black-Scholes": float(
                bs_price(spot, 0.95 * spot, 30.0 / 365.25, 0.01, historical_params[currency].sigma, "put")
            ),
            "Merton": float(
                merton_price(
                    spot,
                    0.95 * spot,
                    30.0 / 365.25,
                    0.01,
                    "put",
                    sigma=merton_params.sigma,
                    jump_intensity=merton_params.jump_intensity,
                    jump_mean=merton_params.jump_mean,
                    jump_vol=merton_params.jump_vol,
                )
            ),
            "SVCJ proxy": float(svcj_moment_price(stress_option, svcj_params)[0]),
            "MR-ISVM surface": float(implied_surface_price(stress_option, surface_params)[0]),
            "Dynamic jump": float(dynamic_jump_price(stress_option, dynamic_params)[0]),
            "GARCH variance": float(garch_price(stress_option, garch_params)[0]),
        }
        bs_stress = stress_prices["Black-Scholes"]
        stress_rows.extend(
            [
                {
                    "Currency": currency,
                    "Model": model,
                    "30d 5pct OTM put USD": price,
                    "Premium over BS USD": price - bs_stress,
                    "Premium over BS pct": (price - bs_stress) / max(bs_stress, 1e-12),
                }
                for model, price in stress_prices.items()
            ]
        )

    candidate_priced = pd.concat(priced_frames, ignore_index=True)
    candidate_errors = summarize_candidate_errors(candidate_priced)
    candidate_parameters = pd.DataFrame(parameter_rows)
    candidate_stress = pd.DataFrame(stress_rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, currency in zip(axes, ["BTC", "ETH"]):
        plot_df = candidate_errors[candidate_errors["currency"] == currency].copy()
        sns.barplot(data=plot_df, x="model", y="mae_usd", ax=ax, color="#386fa4")
        ax.set_title(f"{currency} candidate model MAE")
        ax.set_xlabel("")
        ax.set_ylabel("Mean absolute pricing error, USD")
        ax.tick_params(axis="x", labelrotation=32)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_candidate_model_mae.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, currency in zip(axes, ["BTC", "ETH"]):
        plot_df = candidate_stress[candidate_stress["Currency"] == currency].copy()
        sns.barplot(data=plot_df, x="Model", y="30d 5pct OTM put USD", ax=ax, color="#4f8a5b")
        ax.set_title(f"{currency} 30-day 5% OTM put model value")
        ax.set_xlabel("")
        ax.set_ylabel("Model price, USD")
        ax.tick_params(axis="x", labelrotation=32)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_candidate_stress_puts.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "candidate_model_errors": candidate_errors,
        "candidate_model_parameters": candidate_parameters,
        "candidate_stress_prices": candidate_stress,
    }


def _summary_tables(snapshot, options_all, raw_counts, usable_counts, historical_params, calibration_results, return_frames, priced_all):
    data_summary = pd.DataFrame(
        [
            {
                "Currency": currency,
                "Raw option rows": raw_counts[currency],
                "Usable rows": usable_counts[currency],
                "Median spot (USD)": float(
                    options_all.loc[options_all["currency"] == currency, "underlying_price"].median()
                ),
                "Median mark IV": float(
                    options_all.loc[options_all["currency"] == currency, "mark_iv_decimal"].median()
                ),
                "Min maturity (days)": float(
                    options_all.loc[options_all["currency"] == currency, "time_to_maturity"].min() * 365.25
                ),
                "Max maturity (days)": float(
                    options_all.loc[options_all["currency"] == currency, "time_to_maturity"].max() * 365.25
                ),
            }
            for currency in ["BTC", "ETH"]
        ]
    )

    jump_summary = pd.DataFrame(
        [
            {
                "Currency": currency,
                "Index observations": len(snapshot[currency]["index"]),
                "Return observations": len(return_frames[currency]),
                "Jump observations": int(return_frames[currency]["is_jump"].sum()),
                "Diffusive sigma": historical_params[currency].sigma,
                "Jump intensity": historical_params[currency].jump_intensity,
                "Jump mean": historical_params[currency].jump_mean,
                "Jump vol": historical_params[currency].jump_vol,
                "Worst log return": float(return_frames[currency]["log_return"].min()),
                "Best log return": float(return_frames[currency]["log_return"].max()),
            }
            for currency in ["BTC", "ETH"]
        ]
    )

    calibration_rows = []
    for currency in ["BTC", "ETH"]:
        result = calibration_results[currency]
        params = result.params
        errors = pricing_error_summary(priced_all[priced_all["currency"] == currency])
        bs_mae = float(errors.loc[errors["model"] == "Black-Scholes", "MAE_USD"].iloc[0])
        merton_mae = float(errors.loc[errors["model"] == "Merton", "MAE_USD"].iloc[0])
        bs_rmse = float(errors.loc[errors["model"] == "Black-Scholes", "RMSE_USD"].iloc[0])
        merton_rmse = float(errors.loc[errors["model"] == "Merton", "RMSE_USD"].iloc[0])
        calibration_rows.append(
            {
                "Currency": currency,
                "Calibrated sigma": params.sigma,
                "Calibrated lambda": params.jump_intensity,
                "Calibrated jump mean": params.jump_mean,
                "Calibrated jump vol": params.jump_vol,
                "Calibration options": result.n_options,
                "BS MAE USD": bs_mae,
                "Merton MAE USD": merton_mae,
                "MAE improvement": (bs_mae - merton_mae) / bs_mae,
                "BS RMSE USD": bs_rmse,
                "Merton RMSE USD": merton_rmse,
            }
        )
    calibration_summary = pd.DataFrame(calibration_rows)

    rate_rows = []
    greek_rows = []
    american_rows = []
    for currency in ["BTC", "ETH"]:
        spot = float(options_all.loc[options_all["currency"] == currency, "underlying_price"].median())
        sigma = historical_params[currency].sigma
        merton_params = calibration_results[currency].params
        for option_type in ["call", "put"]:
            for rate in [0.01, 0.05]:
                rate_rows.append(
                    {
                        "Currency": currency,
                        "Option": option_type,
                        "Rate": rate,
                        "BS price USD": float(bs_price(spot, spot, 30 / 365.25, rate, sigma, option_type)),
                        "Merton price USD": float(
                            merton_price(
                                spot,
                                spot,
                                30 / 365.25,
                                rate,
                                option_type,
                                sigma=merton_params.sigma,
                                jump_intensity=merton_params.jump_intensity,
                                jump_mean=merton_params.jump_mean,
                                jump_vol=merton_params.jump_vol,
                            )
                        ),
                    }
                )

            bsg = bs_greeks(spot, spot, 30 / 365.25, 0.01, sigma, option_type)
            mg = merton_greeks(
                spot,
                spot,
                30 / 365.25,
                0.01,
                option_type,
                sigma=merton_params.sigma,
                jump_intensity=merton_params.jump_intensity,
                jump_mean=merton_params.jump_mean,
                jump_vol=merton_params.jump_vol,
            )
            greek_rows.append(
                {
                    "Currency": currency,
                    "Option": option_type,
                    "BS delta": float(bsg["delta"]),
                    "Merton delta": float(mg["delta"]),
                    "BS gamma": float(bsg["gamma"]),
                    "Merton gamma": float(mg["gamma"]),
                    "BS vega": float(bsg["vega"]),
                    "Merton vega": float(mg["vega"]),
                }
            )

            bs_eur = float(bs_price(spot, spot, 30 / 365.25, 0.01, sigma, option_type))
            crr_eur = crr_price(spot, spot, 30 / 365.25, 0.01, sigma, option_type, steps=500, style="european")
            crr_am = crr_price(spot, spot, 30 / 365.25, 0.01, sigma, option_type, steps=500, style="american")
            american_rows.append(
                {
                    "Currency": currency,
                    "Option": option_type,
                    "Spot/Strike USD": spot,
                    "BS European USD": bs_eur,
                    "CRR European USD": crr_eur,
                    "CRR American USD": crr_am,
                    "American premium USD": crr_am - crr_eur,
                }
            )

    return {
        "data_summary": data_summary,
        "jump_summary": jump_summary,
        "calibration_summary": calibration_summary,
        "rate_summary": pd.DataFrame(rate_rows),
        "greek_summary": pd.DataFrame(greek_rows),
        "american_european_summary": pd.DataFrame(american_rows),
    }


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    set_project_style()

    snapshot, options_all, raw_counts, usable_counts = _load_market_data()
    historical_params, calibration_results, return_frames, priced_all = _fit_models(snapshot, options_all)

    convergence_summary = _plot_core_figures(
        options_all,
        return_frames,
        priced_all,
        historical_params,
        calibration_results,
    )
    hedge_summary = _hedge_analysis(options_all, calibration_results)
    return_diagnostics = _return_diagnostics(snapshot, return_frames)
    statistical_tests = _pricing_error_tests(priced_all)
    sensitivity_summary = _sensitivity_analysis(options_all, calibration_results)
    candidate_tables = _candidate_analysis(options_all, return_frames, historical_params, calibration_results)
    tables = _summary_tables(
        snapshot,
        options_all,
        raw_counts,
        usable_counts,
        historical_params,
        calibration_results,
        return_frames,
        priced_all,
    )
    tables.update(
        {
            "convergence_summary": convergence_summary,
            "hedge_summary": hedge_summary,
            "return_diagnostics": return_diagnostics,
            "statistical_tests": statistical_tests,
            "jump_sensitivity": sensitivity_summary,
        }
    )
    tables.update(candidate_tables)

    for name, table in tables.items():
        _write_table(name, table)

    metrics = {"snapshot_time": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")}
    metrics.update({name: table.to_dict(orient="records") for name, table in tables.items()})
    (TAB_DIR / "report_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"Generated report assets at {pd.Timestamp.utcnow():%Y-%m-%d %H:%M UTC}")
    for name in [
        "return_diagnostics",
        "statistical_tests",
        "calibration_summary",
        "jump_sensitivity",
        "candidate_model_errors",
        "candidate_stress_prices",
    ]:
        print(f"\n{name.upper()}")
        print(tables[name].to_markdown(index=False, floatfmt=".6g"))


if __name__ == "__main__":
    main()

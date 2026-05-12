"""Plot helpers for the academic notebook."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def set_project_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")


def plot_return_jumps(returns: pd.DataFrame, *, ax=None, title: str = "Detected Return Jumps"):
    ax = ax or plt.gca()
    normal = returns[~returns["is_jump"]]
    jumps = returns[returns["is_jump"]]
    ax.plot(normal["datetime"], normal["log_return"], ".", alpha=0.45, label="Regular return")
    ax.scatter(jumps["datetime"], jumps["log_return"], color="crimson", s=28, label="Jump")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Log return")
    ax.legend()
    return ax


def plot_vol_smile(options: pd.DataFrame, *, ax=None, title: str = "Deribit Implied Volatility Smile"):
    ax = ax or plt.gca()
    sns.scatterplot(
        data=options,
        x="log_moneyness",
        y="mark_iv_decimal",
        hue="option_type",
        size="time_to_maturity",
        sizes=(20, 120),
        alpha=0.7,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("log(K / S)")
    ax.set_ylabel("Mark IV")
    return ax


def plot_model_errors(options: pd.DataFrame, *, ax=None, title: str = "Model Pricing Error"):
    ax = ax or plt.gca()
    plot_df = options.melt(
        id_vars=["instrument_name", "log_moneyness", "option_type"],
        value_vars=["bs_error_usd", "merton_error_usd"],
        var_name="model",
        value_name="error_usd",
    )
    plot_df["model"] = plot_df["model"].map({"bs_error_usd": "Black-Scholes", "merton_error_usd": "Merton"})
    sns.scatterplot(data=plot_df, x="log_moneyness", y="error_usd", hue="model", style="option_type", alpha=0.75, ax=ax)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("log(K / S)")
    ax.set_ylabel("Model - market price (USD)")
    return ax


def plot_hedge_errors(hedge_results: pd.DataFrame, *, ax=None, title: str = "Delta-Hedge Error Distribution"):
    ax = ax or plt.gca()
    sns.boxplot(data=hedge_results, x="model", y="pnl", ax=ax)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Short-option hedge P&L (USD)")
    return ax


def pricing_error_summary(options: pd.DataFrame) -> pd.DataFrame:
    """Return compact model-error diagnostics for the report."""

    return pd.DataFrame(
        {
            "model": ["Black-Scholes", "Merton"],
            "MAE_USD": [options["abs_bs_error_usd"].mean(), options["abs_merton_error_usd"].mean()],
            "RMSE_USD": [
                (options["bs_error_usd"].pow(2).mean()) ** 0.5,
                (options["merton_error_usd"].pow(2).mean()) ** 0.5,
            ],
        }
    )


"""Data cleaning and feature engineering for crypto option chains."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


MS_PER_YEAR = 365.25 * 24 * 60 * 60 * 1000


def _parse_instrument_name(name: str) -> tuple[float, str]:
    parts = str(name).split("-")
    if len(parts) < 4:
        return math.nan, ""
    strike = float(parts[-2].replace("d", "."))
    option_type = "call" if parts[-1].upper() == "C" else "put"
    return strike, option_type


def _normalise_percent(value: pd.Series) -> pd.Series:
    values = pd.to_numeric(value, errors="coerce")
    return np.where(values.abs() > 3.0, values / 100.0, values)


def normalize_option_chain(
    book: pd.DataFrame,
    instruments: pd.DataFrame,
    *,
    valuation_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Merge Deribit book summaries with instrument metadata and add features."""

    if book.empty:
        return pd.DataFrame()

    valuation_time = valuation_time or pd.Timestamp.utcnow()
    valuation_ms = int(valuation_time.timestamp() * 1000)
    instrument_cols = [
        col
        for col in [
            "instrument_name",
            "strike",
            "option_type",
            "expiration_timestamp",
            "contract_size",
            "settlement_currency",
            "quote_currency",
        ]
        if col in instruments.columns
    ]
    df = book.copy()
    if instrument_cols:
        df = df.merge(instruments[instrument_cols], on="instrument_name", how="left")

    parsed = df["instrument_name"].map(_parse_instrument_name)
    parsed_strike = parsed.map(lambda item: item[0])
    parsed_type = parsed.map(lambda item: item[1])
    df["strike"] = pd.to_numeric(df.get("strike", parsed_strike), errors="coerce").fillna(parsed_strike)
    df["option_type"] = df.get("option_type", parsed_type).fillna(parsed_type).str.lower()
    if "expiration_timestamp" not in df:
        df["expiration_timestamp"] = np.nan

    numeric_cols = [
        "bid_price",
        "ask_price",
        "mid_price",
        "mark_price",
        "underlying_price",
        "interest_rate",
        "open_interest",
        "volume",
        "mark_iv",
        "expiration_timestamp",
    ]
    for col in numeric_cols:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["valuation_timestamp"] = valuation_ms
    df["valuation_datetime"] = pd.to_datetime(valuation_ms, unit="ms", utc=True)
    df["expiration_datetime"] = pd.to_datetime(df["expiration_timestamp"], unit="ms", utc=True)
    df["time_to_maturity"] = (df["expiration_timestamp"] - valuation_ms) / MS_PER_YEAR
    df["rate"] = pd.Series(_normalise_percent(df.get("interest_rate", 0.0)), index=df.index).fillna(0.0)
    df["mark_iv_decimal"] = pd.Series(_normalise_percent(df.get("mark_iv", np.nan)), index=df.index)

    for col in ["bid_price", "ask_price", "mid_price", "mark_price"]:
        if col in df:
            df[f"{col}_usd"] = df[col] * df["underlying_price"]

    mid = df.get("mid_price")
    if mid is None:
        mid = (df["bid_price"] + df["ask_price"]) / 2.0
    derived_mid = (df["bid_price"] + df["ask_price"]) / 2.0
    df["market_price_coin"] = np.where(mid > 0.0, mid, df["mark_price"])
    df["market_price_coin"] = np.where(df["market_price_coin"] > 0.0, df["market_price_coin"], derived_mid)
    df["market_price_usd"] = df["market_price_coin"] * df["underlying_price"]
    df["spread_coin"] = df["ask_price"] - df["bid_price"]
    df["spread_usd"] = df["spread_coin"] * df["underlying_price"]
    df["moneyness"] = df["underlying_price"] / df["strike"]
    df["log_moneyness"] = np.log(df["strike"] / df["underlying_price"])
    df["liquidity_score"] = df[["open_interest", "volume"]].fillna(0.0).sum(axis=1)
    return df.sort_values(["expiration_timestamp", "strike", "option_type"]).reset_index(drop=True)


def filter_liquid_options(
    options: pd.DataFrame,
    *,
    moneyness_range: tuple[float, float] = (0.5, 1.5),
    min_ttm: float = 1.0 / 365.25,
    max_ttm: float = 2.0,
    require_two_sided: bool = False,
) -> pd.DataFrame:
    """Keep options suitable for calibration and plotting."""

    if options.empty:
        return options.copy()
    mask = (
        options["underlying_price"].gt(0.0)
        & options["strike"].gt(0.0)
        & options["time_to_maturity"].between(min_ttm, max_ttm)
        & options["market_price_usd"].gt(0.0)
        & options["mark_iv_decimal"].between(0.01, 4.0)
        & options["moneyness"].between(*moneyness_range)
        & options["option_type"].isin(["call", "put"])
    )
    if require_two_sided:
        mask &= options["bid_price"].gt(0.0) & options["ask_price"].gt(0.0)
    return options.loc[mask].copy().reset_index(drop=True)


def compute_log_returns(index_history: pd.DataFrame) -> pd.DataFrame:
    """Compute timestamped log returns from a Deribit index history frame."""

    df = index_history.sort_values("timestamp").copy()
    df["log_price"] = np.log(df["price"])
    df["log_return"] = df["log_price"].diff()
    return df.dropna(subset=["log_return"]).reset_index(drop=True)


def detect_jumps(
    returns: pd.DataFrame,
    *,
    z_threshold: float = 3.0,
    return_col: str = "log_return",
) -> pd.DataFrame:
    """Flag large absolute return shocks using a robust z-score."""

    df = returns.copy()
    median = df[return_col].median()
    mad = (df[return_col] - median).abs().median()
    robust_sigma = 1.4826 * mad if mad > 0.0 else df[return_col].std(ddof=1)
    df["jump_z"] = (df[return_col] - median) / max(robust_sigma, 1e-12)
    df["is_jump"] = df["jump_z"].abs() >= z_threshold
    return df


def combine_currency_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate non-empty frames while preserving an empty fallback."""

    usable = [frame for frame in frames if frame is not None and not frame.empty]
    return pd.concat(usable, ignore_index=True) if usable else pd.DataFrame()


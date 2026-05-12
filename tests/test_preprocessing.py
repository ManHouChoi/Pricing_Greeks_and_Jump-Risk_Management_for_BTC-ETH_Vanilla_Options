import pandas as pd

from src.preprocessing import filter_liquid_options, normalize_option_chain


def test_normalize_option_chain_converts_coin_quote_to_usd():
    book = pd.DataFrame(
        [
            {
                "instrument_name": "BTC-30JUN26-100000-C",
                "bid_price": 0.01,
                "ask_price": 0.02,
                "mid_price": 0.015,
                "mark_price": 0.016,
                "underlying_price": 80000.0,
                "interest_rate": 0.01,
                "open_interest": 5,
                "volume": 1,
                "mark_iv": 65.0,
            }
        ]
    )
    instruments = pd.DataFrame(
        [
            {
                "instrument_name": "BTC-30JUN26-100000-C",
                "strike": 100000.0,
                "option_type": "call",
                "expiration_timestamp": 1782777600000,
            }
        ]
    )
    options = normalize_option_chain(book, instruments, valuation_time=pd.Timestamp("2026-05-12", tz="UTC"))
    assert options.loc[0, "market_price_usd"] == 1200.0
    assert options.loc[0, "mark_iv_decimal"] == 0.65
    assert not filter_liquid_options(options).empty


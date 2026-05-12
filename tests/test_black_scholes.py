import math

from src.black_scholes import bs_price, implied_volatility


def test_put_call_parity():
    S, K, T, r, sigma = 100.0, 105.0, 0.75, 0.03, 0.42
    call = bs_price(S, K, T, r, sigma, "call")
    put = bs_price(S, K, T, r, sigma, "put")
    parity_gap = call - put - (S - K * math.exp(-r * T))
    assert abs(parity_gap) < 1e-8


def test_implied_volatility_reprices_market_price():
    S, K, T, r, sigma = 100.0, 95.0, 0.5, 0.02, 0.55
    price = bs_price(S, K, T, r, sigma, "call")
    recovered = implied_volatility(price, S, K, T, r, "call")
    assert abs(recovered - sigma) < 1e-6


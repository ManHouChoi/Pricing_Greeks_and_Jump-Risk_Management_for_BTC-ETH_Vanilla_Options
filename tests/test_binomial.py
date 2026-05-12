from src.binomial import crr_price
from src.black_scholes import bs_price


def test_crr_converges_to_black_scholes():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.04, 0.25
    tree = crr_price(S, K, T, r, sigma, "call", steps=1200)
    analytic = bs_price(S, K, T, r, sigma, "call")
    assert abs(tree - analytic) < 0.03


def test_american_put_is_at_least_european_put():
    S, K, T, r, sigma = 90.0, 100.0, 1.0, 0.05, 0.3
    european = crr_price(S, K, T, r, sigma, "put", steps=300, style="european")
    american = crr_price(S, K, T, r, sigma, "put", steps=300, style="american")
    assert american >= european


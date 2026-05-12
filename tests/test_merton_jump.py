from src.black_scholes import bs_price
from src.merton_jump import merton_price


def test_merton_reduces_to_black_scholes_when_jump_intensity_zero():
    args = dict(S=100.0, K=110.0, T=0.8, r=0.025, option_type="put")
    bs = bs_price(args["S"], args["K"], args["T"], args["r"], 0.35, args["option_type"])
    merton = merton_price(
        args["S"],
        args["K"],
        args["T"],
        args["r"],
        args["option_type"],
        sigma=0.35,
        jump_intensity=0.0,
        jump_mean=-0.1,
        jump_vol=0.2,
    )
    assert abs(merton - bs) < 1e-10


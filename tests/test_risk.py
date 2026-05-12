import numpy as np

from src.risk import delta_hedge_short_option, make_gap_stress_path


def test_delta_hedge_returns_finite_pnl():
    path = np.linspace(100.0, 92.0, 20)
    result = delta_hedge_short_option(path, K=100.0, T=30 / 365.25, r=0.01, sigma=0.5, option_type="put")
    assert np.isfinite(result["pnl"])
    assert result["model"] == "black_scholes"


def test_gap_stress_path_applies_gap_to_tail():
    path = np.array([100.0, 101.0, 102.0, 103.0])
    stressed = make_gap_stress_path(path, shock_log_return=-0.1, shock_index=2)
    assert stressed[0] == path[0]
    assert stressed[2] < path[2]
    assert stressed[3] < path[3]


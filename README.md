# IEDA4331 Crypto Options: Merton Jump-Diffusion

This project prices BTC/ETH vanilla options with Black-Scholes and CRR binomial trees, then adds a Merton jump-diffusion model to explain crypto jump risk, volatility smiles, Greeks, and hedge errors.

## Quick Start

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest
.venv/bin/jupyter notebook notebooks/IEDA4331_crypto_options_merton.ipynb
```

The notebook fetches public Deribit data, caches responses in `data/cache/`, calibrates BTC/ETH jump parameters, and generates report-ready figures.

## Structure

- `src/data_deribit.py`: public Deribit API client and caching.
- `src/preprocessing.py`: option-chain cleaning, quote conversion, returns, and jump flags.
- `src/black_scholes.py`: analytic prices, Greeks, and implied volatility.
- `src/binomial.py`: European/American CRR tree and convergence diagnostics.
- `src/merton_jump.py`: Merton pricing, finite-difference Greeks, and path simulation.
- `src/calibration.py`: historical jump initialization and market calibration.
- `src/risk.py`: jump stress paths and delta-hedging experiments.
- `src/plots.py`: notebook plotting helpers and pricing-error summary tables.
- `tests/`: unit tests for the pricing identities and core risk utilities.


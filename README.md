# IEDA4331 Crypto Options: Candidate Model Comparison

This project prices BTC/ETH vanilla options with Black-Scholes and CRR binomial trees, then compares Merton jump-diffusion, SVCJ, MR-ISVM, dynamic jump intensity, and GARCH-style variance models at the same empirical level. The goal is to explain crypto jump risk, volatility smiles, Greeks, stress pricing, and hedge errors without collapsing the work into a single beginner-style notebook.

## Quick Start

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest
.venv/bin/jupyter notebook notebooks/IEDA4331_crypto_options_merton.ipynb
```

The notebook fetches public Deribit data, caches responses in `data/cache/`, calibrates BTC/ETH model states, and generates report-ready figures. The reproducible report asset pipeline is:

```bash
.venv/bin/python scripts/generate_report_assets.py
```

## Structure

- `src/data_deribit.py`: public Deribit API client and caching.
- `src/preprocessing.py`: option-chain cleaning, quote conversion, returns, and jump flags.
- `src/black_scholes.py`: analytic prices, Greeks, and implied volatility.
- `src/binomial.py`: European/American CRR tree and convergence diagnostics.
- `src/merton_jump.py`: Merton pricing, finite-difference Greeks, and path simulation.
- `src/candidate_models.py`: comparable SVCJ proxy, MR-ISVM surface, dynamic jump intensity, and GARCH variance models.
- `src/calibration.py`: historical jump initialization and market calibration.
- `src/risk.py`: jump stress paths and delta-hedging experiments.
- `src/plots.py`: notebook plotting helpers and pricing-error summary tables.
- `tests/`: unit tests for the pricing identities and core risk utilities.

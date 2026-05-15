| Currency   | model           | parameter                                      |       value | description                                                                   |
|:-----------|:----------------|:-----------------------------------------------|------------:|:------------------------------------------------------------------------------|
| BTC        | MR-ISVM surface | intercept                                      |    0.335961 | ATM IV level (baseline implied volatility)                                    |
| BTC        | MR-ISVM surface | log_moneyness                                  |   -0.036263 | linear skew slope (tilts smile left/right)                                    |
| BTC        | MR-ISVM surface | abs_log_moneyness                              |    0.315198 | symmetric smile width (V-shape lift)                                          |
| BTC        | MR-ISVM surface | log_moneyness_sq                               |   -0.268759 | quadratic smile curvature (bowl depth)                                        |
| BTC        | MR-ISVM surface | sqrt_T                                         |   -0.020705 | term-structure level (IV change with sqrtT)                                   |
| BTC        | MR-ISVM surface | T                                              |    0.170945 | term-structure curvature (IV change with T)                                   |
| BTC        | MR-ISVM surface | put                                            |    0.121891 | put vs call IV offset at ATM                                                  |
| BTC        | MR-ISVM surface | put_x_log_moneyness                            |   -0.131368 | put-side skew slope (asymmetric tilt)                                         |
| BTC        | MR-ISVM surface | short_maturity                                 |   -0.01459  | short-dated IV premium (<=30 day options)                                     |
| BTC        | MR-ISVM surface | log_moneyness_x_sqrt_T                         |   -0.352305 | SVI cross-term: skew steepness vs maturity                                    |
| BTC        | MR-ISVM surface | put_x_sqrt_T                                   |   -0.304671 | put-side term slope (separate from calls)                                     |
| BTC        | MR-ISVM surface | log_moneyness_cube                             |    0.533687 | cubic skew asymmetry (OTM put wing steepness)                                 |
| BTC        | MR-ISVM surface | sigma_floor                                    |    0.2      | minimum predicted IV (crypto lower bound)                                     |
| BTC        | MR-ISVM surface | sigma_cap                                      |    3        | maximum predicted IV (outlier cap)                                            |
| BTC        | MR-ISVM surface | fit_rmse                                       |    0.260998 | in-sample relative price RMSE from surface calibration                        |
| BTC        | SVCJ proxy      | v0                                             |    0.060814 | initial variance v0 (spot variance at calibration date)                       |
| BTC        | SVCJ proxy      | theta                                          |    0.022314 | long-run variance theta (variance mean-reversion target)                      |
| BTC        | SVCJ proxy      | kappa_v                                        |   12        | variance mean-reversion speed kappa (per year; higher = faster pull to theta) |
| BTC        | SVCJ proxy      | variance_jump_mean                             |    0.089256 | mean variance jump size mu_v (Exp scale; adds to v on each jump)              |
| BTC        | SVCJ proxy      | leverage_rho                                   |   -0.104058 | spot-vol correlation rho (negative = leverage effect / skew)                  |
| BTC        | SVCJ proxy      | variance_multiplier                            |    0.305776 | overall variance scale factor (calibrated to option prices)                   |
| BTC        | SVCJ proxy      | jump_intensity                                 |   39.0532   | jump arrival rate lambda (jumps per year)                                     |
| BTC        | SVCJ proxy      | jump_mean                                      |   -0.031664 | mean log spot-price jump mu_S (negative = crash bias)                         |
| BTC        | SVCJ proxy      | jump_vol                                       |    0.05617  | std of log spot-price jump sigma_S (uncertainty in jump size)                 |
| BTC        | SVCJ proxy      | xi                                             |    0.28697  | vol-of-vol xi (controls variance-of-variance / smile width)                   |
| BTC        | Dynamic jump    | sigma                                          |    0.203157 | diffusive volatility sigma (non-jump component)                               |
| BTC        | Dynamic jump    | base_intensity                                 |   32.6059   | long-run jump intensity lambda (unconditional arrival rate per year)          |
| BTC        | Dynamic jump    | current_intensity                              |   32.606    | current jump intensity lambda_t (elevated after recent jumps)                 |
| BTC        | Dynamic jump    | mean_reversion                                 |    1.59986  | intensity mean-reversion speed beta (per year; how fast lambda_t -> lambda)   |
| BTC        | Dynamic jump    | jump_mean                                      |   -0.031664 | mean log spot-price jump mu_S                                                 |
| BTC        | Dynamic jump    | jump_vol                                       |    0.05617  | std of log spot-price jump sigma_S                                            |
| BTC        | GARCH variance  | omega                                          |    2e-06    | GARCH constant omega (unconditional variance floor)                           |
| BTC        | GARCH variance  | alpha                                          |    0.061262 | ARCH coefficient alpha (weight on last squared return)                        |
| BTC        | GARCH variance  | beta                                           |    0.918936 | GARCH coefficient beta (weight on last variance)                              |
| BTC        | GARCH variance  | last_variance                                  |    5.4e-05  | conditional variance at last observation h_T                                  |
| BTC        | GARCH variance  | mean_return                                    |   -0.000163 | estimated mean log-return (subtracted before fitting)                         |
| BTC        | GARCH variance  | periods_per_year                               | 1461        | trading periods per year used for annualisation                               |
| BTC        | GARCH variance  | persistence (alpha + beta)                     |    0.980198 | GARCH persistence (< 1 required for stationarity; crypto approx 0.95-0.99)    |
| BTC        | GARCH variance  | long_run_variance (omega / (1 - alpha - beta)) |    8.9e-05  | unconditional variance implied by GARCH parameters                            |
| ETH        | MR-ISVM surface | intercept                                      |    0.40276  | ATM IV level (baseline implied volatility)                                    |
| ETH        | MR-ISVM surface | log_moneyness                                  |   -0.209701 | linear skew slope (tilts smile left/right)                                    |
| ETH        | MR-ISVM surface | abs_log_moneyness                              |    0.708691 | symmetric smile width (V-shape lift)                                          |
| ETH        | MR-ISVM surface | log_moneyness_sq                               |   -0.411043 | quadratic smile curvature (bowl depth)                                        |
| ETH        | MR-ISVM surface | sqrt_T                                         |    0.18915  | term-structure level (IV change with sqrtT)                                   |
| ETH        | MR-ISVM surface | T                                              |    0.009252 | term-structure curvature (IV change with T)                                   |
| ETH        | MR-ISVM surface | put                                            |    0.081577 | put vs call IV offset at ATM                                                  |
| ETH        | MR-ISVM surface | put_x_log_moneyness                            |    0.271924 | put-side skew slope (asymmetric tilt)                                         |
| ETH        | MR-ISVM surface | short_maturity                                 |   -0.023394 | short-dated IV premium (<=30 day options)                                     |
| ETH        | MR-ISVM surface | log_moneyness_x_sqrt_T                         |   -0.372417 | SVI cross-term: skew steepness vs maturity                                    |
| ETH        | MR-ISVM surface | put_x_sqrt_T                                   |   -0.24216  | put-side term slope (separate from calls)                                     |
| ETH        | MR-ISVM surface | log_moneyness_cube                             |    0.310878 | cubic skew asymmetry (OTM put wing steepness)                                 |
| ETH        | MR-ISVM surface | sigma_floor                                    |    0.2      | minimum predicted IV (crypto lower bound)                                     |
| ETH        | MR-ISVM surface | sigma_cap                                      |    3        | maximum predicted IV (outlier cap)                                            |
| ETH        | MR-ISVM surface | fit_rmse                                       |    0.294461 | in-sample relative price RMSE from surface calibration                        |
| ETH        | SVCJ proxy      | v0                                             |    0.081814 | initial variance v0 (spot variance at calibration date)                       |
| ETH        | SVCJ proxy      | theta                                          |    0.072387 | long-run variance theta (variance mean-reversion target)                      |
| ETH        | SVCJ proxy      | kappa_v                                        |   12        | variance mean-reversion speed kappa (per year; higher = faster pull to theta) |
| ETH        | SVCJ proxy      | variance_jump_mean                             |    0.289548 | mean variance jump size mu_v (Exp scale; adds to v on each jump)              |
| ETH        | SVCJ proxy      | leverage_rho                                   |   -0.107822 | spot-vol correlation rho (negative = leverage effect / skew)                  |
| ETH        | SVCJ proxy      | variance_multiplier                            |    0.342874 | overall variance scale factor (calibrated to option prices)                   |
| ETH        | SVCJ proxy      | jump_intensity                                 |   40.054    | jump arrival rate lambda (jumps per year)                                     |
| ETH        | SVCJ proxy      | jump_mean                                      |   -0.031852 | mean log spot-price jump mu_S (negative = crash bias)                         |
| ETH        | SVCJ proxy      | jump_vol                                       |    0.068614 | std of log spot-price jump sigma_S (uncertainty in jump size)                 |
| ETH        | SVCJ proxy      | xi                                             |    0.352909 | vol-of-vol xi (controls variance-of-variance / smile width)                   |
| ETH        | Dynamic jump    | sigma                                          |    0.321076 | diffusive volatility sigma (non-jump component)                               |
| ETH        | Dynamic jump    | base_intensity                                 |   33.3241   | long-run jump intensity lambda (unconditional arrival rate per year)          |
| ETH        | Dynamic jump    | current_intensity                              |   33.3274   | current jump intensity lambda_t (elevated after recent jumps)                 |
| ETH        | Dynamic jump    | mean_reversion                                 |    0.833751 | intensity mean-reversion speed beta (per year; how fast lambda_t -> lambda)   |
| ETH        | Dynamic jump    | jump_mean                                      |   -0.031852 | mean log spot-price jump mu_S                                                 |
| ETH        | Dynamic jump    | jump_vol                                       |    0.068614 | std of log spot-price jump sigma_S                                            |
| ETH        | GARCH variance  | omega                                          |    8e-06    | GARCH constant omega (unconditional variance floor)                           |
| ETH        | GARCH variance  | alpha                                          |    0.059191 | ARCH coefficient alpha (weight on last squared return)                        |
| ETH        | GARCH variance  | beta                                           |    0.899803 | GARCH coefficient beta (weight on last variance)                              |
| ETH        | GARCH variance  | last_variance                                  |    0.000116 | conditional variance at last observation h_T                                  |
| ETH        | GARCH variance  | mean_return                                    |   -7.9e-05  | estimated mean log-return (subtracted before fitting)                         |
| ETH        | GARCH variance  | periods_per_year                               | 1461        | trading periods per year used for annualisation                               |
| ETH        | GARCH variance  | persistence (alpha + beta)                     |    0.958993 | GARCH persistence (< 1 required for stationarity; crypto approx 0.95-0.99)    |
| ETH        | GARCH variance  | long_run_variance (omega / (1 - alpha - beta)) |    0.000199 | unconditional variance implied by GARCH parameters                            |
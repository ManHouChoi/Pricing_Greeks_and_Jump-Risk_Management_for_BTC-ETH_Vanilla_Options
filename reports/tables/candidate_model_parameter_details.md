| Currency   | model           | parameter                                      |       value | description                                                                   |
|:-----------|:----------------|:-----------------------------------------------|------------:|:------------------------------------------------------------------------------|
| BTC        | MR-ISVM surface | intercept                                      |    0.327187 | ATM IV level (baseline implied volatility)                                    |
| BTC        | MR-ISVM surface | log_moneyness                                  |   -0.135231 | linear skew slope (tilts smile left/right)                                    |
| BTC        | MR-ISVM surface | abs_log_moneyness                              |    0.286776 | symmetric smile width (V-shape lift)                                          |
| BTC        | MR-ISVM surface | log_moneyness_sq                               |   -0.189591 | quadratic smile curvature (bowl depth)                                        |
| BTC        | MR-ISVM surface | sqrt_T                                         |    0.050684 | term-structure level (IV change with sqrtT)                                   |
| BTC        | MR-ISVM surface | T                                              |    0.09015  | term-structure curvature (IV change with T)                                   |
| BTC        | MR-ISVM surface | put                                            |    0.112418 | put vs call IV offset at ATM                                                  |
| BTC        | MR-ISVM surface | put_x_log_moneyness                            |   -0.040236 | put-side skew slope (asymmetric tilt)                                         |
| BTC        | MR-ISVM surface | short_maturity                                 |    0.001482 | short-dated IV premium (<=30 day options)                                     |
| BTC        | MR-ISVM surface | log_moneyness_x_sqrt_T                         |   -0.1955   | SVI cross-term: skew steepness vs maturity                                    |
| BTC        | MR-ISVM surface | put_x_sqrt_T                                   |   -0.247428 | put-side term slope (separate from calls)                                     |
| BTC        | MR-ISVM surface | log_moneyness_cube                             |    0.409222 | cubic skew asymmetry (OTM put wing steepness)                                 |
| BTC        | MR-ISVM surface | sigma_floor                                    |    0.2      | minimum predicted IV (crypto lower bound)                                     |
| BTC        | MR-ISVM surface | sigma_cap                                      |    3        | maximum predicted IV (outlier cap)                                            |
| BTC        | MR-ISVM surface | fit_rmse                                       |    0.369943 | in-sample relative price RMSE from surface calibration                        |
| BTC        | SVCJ proxy      | v0                                             |    0.034459 | initial variance v0 (spot variance at calibration date)                       |
| BTC        | SVCJ proxy      | theta                                          |    0.0009   | long-run variance theta (variance mean-reversion target)                      |
| BTC        | SVCJ proxy      | kappa_v                                        |   12        | variance mean-reversion speed kappa (per year; higher = faster pull to theta) |
| BTC        | SVCJ proxy      | variance_jump_mean                             |    0.0036   | mean variance jump size mu_v (Exp scale; adds to v on each jump)              |
| BTC        | SVCJ proxy      | leverage_rho                                   |   -0.103542 | spot-vol correlation rho (negative = leverage effect / skew)                  |
| BTC        | SVCJ proxy      | variance_multiplier                            |    0.25     | overall variance scale factor (calibrated to option prices)                   |
| BTC        | SVCJ proxy      | jump_intensity                                 |   40.0542   | jump arrival rate lambda (jumps per year)                                     |
| BTC        | SVCJ proxy      | jump_mean                                      |   -0.025168 | mean log spot-price jump mu_S (negative = crash bias)                         |
| BTC        | SVCJ proxy      | jump_vol                                       |    0.063548 | std of log spot-price jump sigma_S (uncertainty in jump size)                 |
| BTC        | SVCJ proxy      | xi                                             |    0.05     | vol-of-vol xi (controls variance-of-variance / smile width)                   |
| BTC        | Dynamic jump    | sigma                                          |    0.15855  | diffusive volatility sigma (non-jump component)                               |
| BTC        | Dynamic jump    | base_intensity                                 |   31.7627   | long-run jump intensity lambda (unconditional arrival rate per year)          |
| BTC        | Dynamic jump    | current_intensity                              |   31.7628   | current jump intensity lambda_t (elevated after recent jumps)                 |
| BTC        | Dynamic jump    | mean_reversion                                 |    2.04918  | intensity mean-reversion speed beta (per year; how fast lambda_t -> lambda)   |
| BTC        | Dynamic jump    | jump_mean                                      |   -0.025168 | mean log spot-price jump mu_S                                                 |
| BTC        | Dynamic jump    | jump_vol                                       |    0.063548 | std of log spot-price jump sigma_S                                            |
| BTC        | GARCH variance  | omega                                          |    2e-06    | GARCH constant omega (unconditional variance floor)                           |
| BTC        | GARCH variance  | alpha                                          |    0.061311 | ARCH coefficient alpha (weight on last squared return)                        |
| BTC        | GARCH variance  | beta                                           |    0.919672 | GARCH coefficient beta (weight on last variance)                              |
| BTC        | GARCH variance  | last_variance                                  |    4.4e-05  | conditional variance at last observation h_T                                  |
| BTC        | GARCH variance  | mean_return                                    |   -0.000223 | estimated mean log-return (subtracted before fitting)                         |
| BTC        | GARCH variance  | periods_per_year                               | 1461        | trading periods per year used for annualisation                               |
| BTC        | GARCH variance  | persistence (alpha + beta)                     |    0.980984 | GARCH persistence (< 1 required for stationarity; crypto approx 0.95-0.99)    |
| BTC        | GARCH variance  | long_run_variance (omega / (1 - alpha - beta)) |    8.9e-05  | unconditional variance implied by GARCH parameters                            |
| ETH        | MR-ISVM surface | intercept                                      |    0.473286 | ATM IV level (baseline implied volatility)                                    |
| ETH        | MR-ISVM surface | log_moneyness                                  |   -0.104952 | linear skew slope (tilts smile left/right)                                    |
| ETH        | MR-ISVM surface | abs_log_moneyness                              |    0.442645 | symmetric smile width (V-shape lift)                                          |
| ETH        | MR-ISVM surface | log_moneyness_sq                               |   -0.243731 | quadratic smile curvature (bowl depth)                                        |
| ETH        | MR-ISVM surface | sqrt_T                                         |   -0.109534 | term-structure level (IV change with sqrtT)                                   |
| ETH        | MR-ISVM surface | T                                              |    0.285077 | term-structure curvature (IV change with T)                                   |
| ETH        | MR-ISVM surface | put                                            |    0.107284 | put vs call IV offset at ATM                                                  |
| ETH        | MR-ISVM surface | put_x_log_moneyness                            |    0.112723 | put-side skew slope (asymmetric tilt)                                         |
| ETH        | MR-ISVM surface | short_maturity                                 |   -0.034009 | short-dated IV premium (<=30 day options)                                     |
| ETH        | MR-ISVM surface | log_moneyness_x_sqrt_T                         |   -0.336623 | SVI cross-term: skew steepness vs maturity                                    |
| ETH        | MR-ISVM surface | put_x_sqrt_T                                   |   -0.219018 | put-side term slope (separate from calls)                                     |
| ETH        | MR-ISVM surface | log_moneyness_cube                             |    0.292505 | cubic skew asymmetry (OTM put wing steepness)                                 |
| ETH        | MR-ISVM surface | sigma_floor                                    |    0.2      | minimum predicted IV (crypto lower bound)                                     |
| ETH        | MR-ISVM surface | sigma_cap                                      |    3        | maximum predicted IV (outlier cap)                                            |
| ETH        | MR-ISVM surface | fit_rmse                                       |    0.278342 | in-sample relative price RMSE from surface calibration                        |
| ETH        | SVCJ proxy      | v0                                             |    0.09893  | initial variance v0 (spot variance at calibration date)                       |
| ETH        | SVCJ proxy      | theta                                          |    0.10821  | long-run variance theta (variance mean-reversion target)                      |
| ETH        | SVCJ proxy      | kappa_v                                        |   12        | variance mean-reversion speed kappa (per year; higher = faster pull to theta) |
| ETH        | SVCJ proxy      | variance_jump_mean                             |    0.432841 | mean variance jump size mu_v (Exp scale; adds to v on each jump)              |
| ETH        | SVCJ proxy      | leverage_rho                                   |   -0.093343 | spot-vol correlation rho (negative = leverage effect / skew)                  |
| ETH        | SVCJ proxy      | variance_multiplier                            |    0.25     | overall variance scale factor (calibrated to option prices)                   |
| ETH        | SVCJ proxy      | jump_intensity                                 |   40.0542   | jump arrival rate lambda (jumps per year)                                     |
| ETH        | SVCJ proxy      | jump_mean                                      |   -0.03494  | mean log spot-price jump mu_S (negative = crash bias)                         |
| ETH        | SVCJ proxy      | jump_vol                                       |    0.060987 | std of log spot-price jump sigma_S (uncertainty in jump size)                 |
| ETH        | SVCJ proxy      | xi                                             |    0.05     | vol-of-vol xi (controls variance-of-variance / smile width)                   |
| ETH        | Dynamic jump    | sigma                                          |    0.360187 | diffusive volatility sigma (non-jump component)                               |
| ETH        | Dynamic jump    | base_intensity                                 |   33.0129   | long-run jump intensity lambda (unconditional arrival rate per year)          |
| ETH        | Dynamic jump    | current_intensity                              |   34.5101   | current jump intensity lambda_t (elevated after recent jumps)                 |
| ETH        | Dynamic jump    | mean_reversion                                 |    0.25     | intensity mean-reversion speed beta (per year; how fast lambda_t -> lambda)   |
| ETH        | Dynamic jump    | jump_mean                                      |   -0.03494  | mean log spot-price jump mu_S                                                 |
| ETH        | Dynamic jump    | jump_vol                                       |    0.060987 | std of log spot-price jump sigma_S                                            |
| ETH        | GARCH variance  | omega                                          |    8e-06    | GARCH constant omega (unconditional variance floor)                           |
| ETH        | GARCH variance  | alpha                                          |    0.059041 | ARCH coefficient alpha (weight on last squared return)                        |
| ETH        | GARCH variance  | beta                                           |    0.899766 | GARCH coefficient beta (weight on last variance)                              |
| ETH        | GARCH variance  | last_variance                                  |    0.000125 | conditional variance at last observation h_T                                  |
| ETH        | GARCH variance  | mean_return                                    |   -0.00013  | estimated mean log-return (subtracted before fitting)                         |
| ETH        | GARCH variance  | periods_per_year                               | 1461        | trading periods per year used for annualisation                               |
| ETH        | GARCH variance  | persistence (alpha + beta)                     |    0.958807 | GARCH persistence (< 1 required for stationarity; crypto approx 0.95-0.99)    |
| ETH        | GARCH variance  | long_run_variance (omega / (1 - alpha - beta)) |    0.000196 | unconditional variance implied by GARCH parameters                            |
# Gamma-Pareto Distribution

[![PyPI version](https://img.shields.io/pypi/v/gamma-pareto-mixture.svg)](https://pypi.org/project/gamma-pareto-mixture/)
[![Python versions](https://img.shields.io/pypi/pyversions/gamma-pareto-mixture.svg)](https://pypi.org/project/gamma-pareto-mixture/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python implementation of the Gamma-Pareto mixture distribution for modeling heavy-tailed data.

## Features

- **Full distribution support**: PDF, CDF, hazard function, moments, mode
- **Parameter estimation**: Maximum Likelihood Estimation (MLE) with two methods
- **Multiple dataframe support**: Works with both Pandas and Polars
- **Built-in datasets**: Fatigue, Flood, and Tribolium example datasets
- **Visualization**: PDF/CDF fitting plots, Q-Q plots
- **Simulation**: Generate random samples of Gamma-Pareto mixture distribution
- **Goodness-of-fit**: Kolmogorov-Smirnov test

## Installation

```bash
pip install gamma_pareto_mixture
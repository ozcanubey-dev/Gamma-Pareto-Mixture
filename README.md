# Gamma-Pareto Distribution

[![PyPI version](https://img.shields.io/pypi/v/gamma-pareto-distribution.svg)](https://pypi.org/project/gamma-pareto-distribution/)
[![Python versions](https://img.shields.io/pypi/pyversions/gamma-pareto-distribution.svg)](https://pypi.org/project/gamma-pareto-distribution/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/yourusername/gamma-pareto-distribution/actions/workflows/tests.yml/badge.svg)](https://github.com/yourusername/gamma-pareto-distribution/actions/workflows/tests.yml)

A Python implementation of the Gamma-Pareto mixture distribution for modeling heavy-tailed data.

## Features

- **Full distribution support**: PDF, CDF, hazard function, moments, mode
- **Parameter estimation**: Maximum Likelihood Estimation (MLE) with two methods
- **Multiple dataframe support**: Works with both Pandas and Polars
- **Built-in datasets**: Fatigue, Flood, and Tribolium example datasets
- **Visualization**: PDF/CDF fitting plots, Q-Q plots
- **Simulation**: Generate random samples from the distribution
- **Goodness-of-fit**: Kolmogorov-Smirnov test

## Installation

```bash
pip install gamma-pareto-distribution
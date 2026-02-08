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
pip install gamma-pareto-mixture
```



```python
from gamma_pareto_mixture import GammaParetoMixture, simulate_gamma_pareto_mixture, ks_test_gpm
import matplotlib.pyplot as plt

# For standalone execution and testing
if __name__ == "__main__":
    # Example usage
    print("Gamma-Pareto Distribution Package")
    print("=" * 50)

    # Create example instance
    gp = GammaParetoMixture.do_example("Fatigue")

    # Fit distribution
    print("\nFitting Gamma-Pareto to Fatigue dataset...")
    result = gp.fit(method='hybrid')

    # Display results
    print(f"\nFit Results:")
    print(f"Alpha (shape): {result['alpha']:.4f} ± {result['alpha_se']:.4f}")
    print(f"c (scale): {result['c']:.4f} ± {result['c_se']:.4f}")
    print(f"Theta (min): {result['theta']:.4f}")
    print(f"Log-likelihood: {result['loglik']:.4f}")
    print(f"AIC: {result['AIC']:.4f}")

    # Create plots
    print("\nCreating diagnostic plots...")
    ax1, ax2, ax3 = gp.plot_fit(
        alpha=result['alpha'],
        c=result['c'],
        theta=result['theta']
    )
    plt.show()
    print("Done! Check 'plot_result_fatigue_example.png' for diagnostic plots.")
```
"""
Gamma-Pareto Distribution Package.

A Python implementation of the Gamma-Pareto mixture distribution for modeling
heavy-tailed data.
"""

from .core import (
    GammaParetoMixture,
    simulate_gamma_pareto_mixture,
    ks_test_gpm
)

__version__ = "1.0.0"
__author__ = "Ozcan Ubeydullah"
__email__ = "ozcanubey-dev@gmail.com"

__all__ = [
    "GammaParetoMixture",
    "simulate_gamma_pareto_mixture",
    "ks_test_gpm"
]
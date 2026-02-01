"""
Configuration and fixtures for Gamma-Pareto tests.
"""
import pytest
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from typing import Generator

from GaPaDist.mixtureGPDist import GammaPareto, simulate_gamma_pareto_mixture, ks_test_gpm


@pytest.fixture
def sample_params() -> dict:
    """Return sample parameters for Gamma-Pareto distribution."""
    return {"alpha": 2.5, "c": 0.3, "theta": 10.0}


@pytest.fixture
def sample_data(sample_params: dict) -> np.ndarray:
    """Generate sample Gamma-Pareto data."""
    return simulate_gamma_pareto_mixture(
        n_sim=1000,
        alpha=sample_params["alpha"],
        c=sample_params["c"],
        theta=sample_params["theta"],
        random_state=42
    )


@pytest.fixture
def pandas_dataframe(sample_data: np.ndarray) -> pd.DataFrame:
    """Return sample data in pandas DataFrame."""
    return pd.DataFrame({"values": sample_data})


@pytest.fixture
def polars_dataframe(sample_data: np.ndarray) -> pl.DataFrame:
    """Return sample data in polars DataFrame."""
    return pl.DataFrame({"values": sample_data})


@pytest.fixture
def small_dataset() -> pd.DataFrame:
    """Return a small dataset for edge case testing."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    return pd.DataFrame({"small_data": data})


@pytest.fixture
def example_gamma_pareto(pandas_dataframe: pd.DataFrame) -> GammaPareto:
    """Return GammaParetoMixture instance with sample data."""
    return GammaPareto(data=pandas_dataframe, colname="values")


@pytest.fixture
def gamma_pareto_with_polars(polars_dataframe: pl.DataFrame) -> GammaPareto:
    """Return GammaParetoMixture instance with polars DataFrame."""
    return GammaPareto(data=polars_dataframe, colname="values")


@pytest.fixture
def fitted_gamma_pareto(example_gamma_pareto: GammaPareto) -> dict:
    """Return fitted Gamma-Pareto parameters."""
    return example_gamma_pareto.fit(method="hybrid")
"""
Test Gamma-Pareto distribution properties and statistical methods.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from GaPaDist.mixtureGPDist import GammaPareto, simulate_gamma_pareto_mixture, ks_test_gpm


class TestGammaParetoProperties:
    """Test distribution properties and methods."""


    def test_initialization_invalid_column(self, pandas_dataframe):
        """Test initialization with invalid column name."""
        with pytest.raises(ValueError, match="Column"):
            GammaPareto(data=pandas_dataframe, colname="nonexistent")


    def test_get_datatype_pandas(self, pandas_dataframe):
        """Test get_datatype for pandas DataFrame."""
        gp = GammaPareto(data=pandas_dataframe, colname="values")
        assert gp.get_datatype() == "Pandas"

    def test_get_datatype_polars(self, polars_dataframe):
        """Test get_datatype for polars DataFrame."""
        gp = GammaPareto(data=polars_dataframe, colname="values")
        assert gp.get_datatype() == "Polars"

    def test_get_data_vector_pandas(self, pandas_dataframe, sample_data):
        """Test get_data_vector for pandas DataFrame."""
        gp = GammaPareto(data=pandas_dataframe, colname="values")
        data_vector = gp.get_data_vector()

        assert isinstance(data_vector, np.ndarray)
        assert_array_equal(data_vector, sample_data)
        assert gp._data_vector is not None  # Should be cached

    def test_get_data_vector_polars(self, polars_dataframe, sample_data):
        """Test get_data_vector for polars DataFrame."""
        gp = GammaPareto(data=polars_dataframe, colname="values")
        data_vector = gp.get_data_vector()

        assert isinstance(data_vector, np.ndarray)
        assert_array_equal(data_vector, sample_data)
        assert gp._data_vector is not None  # Should be cached


    def test_cdf_monotonic(self, example_gamma_pareto, sample_params):
        """Test that CDF is monotonically increasing."""
        cdf_result = example_gamma_pareto.cdf(**sample_params)
        diffs = np.diff(cdf_result["cdf"])
        assert np.all(diffs >= 0)  # Should be non-decreasing

    def test_cdf_at_theta(self, example_gamma_pareto, sample_params):
        """Test CDF at theta boundary."""
        cdf_result = example_gamma_pareto.cdf(**sample_params)
        theta = sample_params["theta"]

        # Find index where x <= theta
        mask = cdf_result["x"] <= theta
        assert np.all(cdf_result["cdf"][mask] == 0)

    def test_hazard_function(self, example_gamma_pareto, sample_params):
        """Test hazard function computation."""
        hazard_result = example_gamma_pareto.hazard_fct(**sample_params)

        assert "x" in hazard_result
        assert "hazard" in hazard_result
        assert len(hazard_result["x"]) == 1000
        assert len(hazard_result["hazard"]) == 1000
        assert np.all(hazard_result["hazard"][hazard_result["x"] <= sample_params["theta"]] == 0)
        assert np.all(np.isfinite(hazard_result["hazard"]) | (hazard_result["hazard"] == 0))

    def test_hazard_non_negative(self, example_gamma_pareto, sample_params):
        """Test that hazard function is non-negative."""
        hazard_result = example_gamma_pareto.hazard_fct(**sample_params)
        assert np.all(hazard_result["hazard"] >= 0)

    def test_moments_valid(self, example_gamma_pareto, sample_params):
        """Test moments computation when condition is met."""
        # c = 0.3, so condition c < 1/r is met for r=1,2,3
        for r in [1, 2, 3]:
            moment_result = example_gamma_pareto.moments(r, **sample_params)

            assert moment_result["Order"] == r
            assert isinstance(moment_result["value"], float)
            assert moment_result["value"] > 0
            assert "condition is met" in moment_result["message"]

    def test_moments_infinite(self, example_gamma_pareto):
        """Test moments computation when condition is not met."""
        # Use c=0.6 so that for r=2, c < 1/2 condition is not met
        params = {"alpha": 2.5, "c": 0.6, "theta": 10.0}

        moment_result = example_gamma_pareto.moments(2, **params)

        assert moment_result["Order"] == 2
        assert np.isinf(moment_result["value"])
        assert "condition is not met" in moment_result["message"]

    def test_mean(self, example_gamma_pareto, sample_params):
        """Test mean computation."""
        mean_result = example_gamma_pareto.mean(**sample_params)

        assert "Mean" in mean_result
        assert isinstance(mean_result["Mean"], float)
        assert mean_result["Mean"] > sample_params["theta"]

        # Compare with manual calculation
        expected = sample_params["theta"] * (1 - sample_params["c"]) ** (-sample_params["alpha"])
        assert_allclose(mean_result["Mean"], expected, rtol=1e-10)

    def test_variance_finite(self, example_gamma_pareto):
        """Test variance when finite (c < 0.5)."""
        params = {"alpha": 2.5, "c": 0.3, "theta": 10.0}  # c < 0.5

        var_result = example_gamma_pareto.variance(**params)

        assert "Variance" in var_result
        assert isinstance(var_result["Variance"], float)
        assert var_result["Variance"] > 0
        assert "condition is met" in var_result["message"]

    def test_variance_infinite(self, example_gamma_pareto):
        """Test variance when infinite (c >= 0.5)."""
        params = {"alpha": 2.5, "c": 0.6, "theta": 10.0}  # c > 0.5

        var_result = example_gamma_pareto.variance(**params)

        assert np.isinf(var_result["Variance"])
        assert "condition is not met" in var_result["message"]

    def test_mode_alpha_le_1(self, example_gamma_pareto):
        """Test mode when alpha <= 1."""
        params = {"alpha": 0.8, "c": 0.3, "theta": 10.0}

        mode_result = example_gamma_pareto.mode(**params)

        assert mode_result["Mode"] == params["theta"]
        assert "alpha ≤ 1" in mode_result["message"]

    def test_mode_alpha_gt_1(self, example_gamma_pareto, sample_params):
        """Test mode when alpha > 1."""
        mode_result = example_gamma_pareto.mode(**sample_params)

        expected = sample_params["theta"] * np.exp(
            sample_params["c"] * (sample_params["alpha"] - 1) / (sample_params["c"] + 1)
        )

        assert_allclose(mode_result["Mode"], expected, rtol=1e-10)
        assert mode_result["Mode"] > sample_params["theta"]

    def test_neg_loglikelihood_valid(self, example_gamma_pareto, sample_params):
        """Test negative log-likelihood computation with valid parameters."""
        theta = sample_params["theta"]
        params = [sample_params["alpha"], sample_params["c"]]

        nll = example_gamma_pareto.neg_loglikelihood(params, theta)

        assert isinstance(nll, float)
        assert np.isfinite(nll)
        assert nll > 0  # Negative log-likelihood should be positive

    def test_neg_loglikelihood_alpha_zero(self, example_gamma_pareto, sample_params):
        """Test negative log-likelihood with alpha = 0."""
        theta = sample_params["theta"]
        result = example_gamma_pareto.neg_loglikelihood([0, sample_params["c"]], theta)

        assert isinstance(result, dict)
        assert result["Loglikelihood"] == -np.inf

    def test_neg_loglikelihood_alpha_negative(self, example_gamma_pareto, sample_params):
        """Test negative log-likelihood with negative alpha."""
        theta = sample_params["theta"]
        result = example_gamma_pareto.neg_loglikelihood([-1.0, sample_params["c"]], theta)

        assert isinstance(result, dict)
        assert result["Loglikelihood"] == -np.inf

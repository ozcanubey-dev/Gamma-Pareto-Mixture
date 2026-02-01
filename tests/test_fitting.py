"""
Test parameter estimation and fitting methods.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from GaPaDist.mixtureGPDist import GammaPareto, simulate_gamma_pareto_mixture, ks_test_gpm


class TestFittingMethods:
    """Test parameter estimation methods."""

    def test_fit_hybrid_valid(self, example_gamma_pareto):
        """Test hybrid fitting method."""
        result = example_gamma_pareto.fit(method="hybrid")

        assert "message" in result
        assert "loglik" in result
        assert "alpha" in result
        assert "c" in result
        assert "theta" in result
        assert "AIC" in result
        assert "alpha_se" in result
        assert "c_se" in result

        assert isinstance(result["alpha"], float)
        assert isinstance(result["c"], float)
        assert isinstance(result["theta"], float)
        assert isinstance(result["loglik"], float)
        assert isinstance(result["AIC"], float)

        assert result["alpha"] > 0
        assert result["c"] > 0
        assert result["theta"] > 0
        assert result["loglik"] > -np.inf
        assert np.isfinite(result["AIC"])

        # Standard errors should be positive
        assert result["alpha_se"] > 0
        assert result["c_se"] > 0

    def test_fit_optimize_valid(self, example_gamma_pareto):
        """Test optimization fitting method."""
        result = example_gamma_pareto.fit(method="optimize")

        assert "message" in result
        assert "loglik" in result
        assert "alpha" in result
        assert "c" in result
        assert "theta" in result
        assert "AIC" in result

        assert result["alpha"] > 0
        assert result["c"] > 0
        assert result["theta"] > 0
        assert result["loglik"] > -np.inf
        assert np.isfinite(result["AIC"])

    def test_fit_methods_consistency(self, example_gamma_pareto):
        """Test consistency between hybrid and optimize methods."""
        result_hybrid = example_gamma_pareto.fit(method="hybrid")
        result_optimize = example_gamma_pareto.fit(method="optimize")

        # Both methods should produce reasonable parameters
        assert result_hybrid["alpha"] > 0
        assert result_hybrid["c"] > 0
        assert result_optimize["alpha"] > 0
        assert result_optimize["c"] > 0

        # Theta should be the same (sample minimum)
        assert result_hybrid["theta"] == result_optimize["theta"]

        # Log-likelihoods should be similar (within reasonable tolerance)
        assert_allclose(
            result_hybrid["loglik"],
            result_optimize["loglik"],
            rtol=0.1
        )

    def test_fit_with_polars(self, gamma_pareto_with_polars):
        """Test fitting with polars DataFrame."""
        result = gamma_pareto_with_polars.fit(method="hybrid")

        assert "alpha" in result
        assert "c" in result
        assert "theta" in result
        assert result["alpha"] > 0
        assert result["c"] > 0
        assert result["theta"] > 0

    def test_fit_theta_is_minimum(self, example_gamma_pareto):
        """Test that theta is estimated as sample minimum."""
        result = example_gamma_pareto.fit(method="hybrid")
        data_min = np.min(example_gamma_pareto.get_data_vector())

        assert result["theta"] == data_min

    def test_fit_loglikelihood_increases(self, example_gamma_pareto, sample_params):
        """Test that fitted parameters improve log-likelihood."""
        result = example_gamma_pareto.fit(method="hybrid")

        # Compute log-likelihood with fitted parameters
        fitted_nll = example_gamma_pareto.neg_loglikelihood(
            [result["alpha"], result["c"]],
            result["theta"]
        )

        # Compute log-likelihood with sample parameters
        sample_nll = example_gamma_pareto.neg_loglikelihood(
            [sample_params["alpha"], sample_params["c"]],
            sample_params["theta"]
        )

        # Fitted parameters should have lower negative log-likelihood (higher likelihood)
        assert fitted_nll <= sample_nll

    def test_aic_calculation(self, example_gamma_pareto):
        """Test AIC calculation."""
        result = example_gamma_pareto.fit(method="hybrid")

        # AIC = 2k - 2*log(L) where k=3 parameters
        expected_aic = 2 * 3 - 2 * result["loglik"]

        assert_allclose(result["AIC"], expected_aic, rtol=1e-10)

    def test_standard_errors_positive(self, example_gamma_pareto):
        """Test that standard errors are positive."""
        result = example_gamma_pareto.fit(method="hybrid")

        assert result["alpha_se"] > 0
        assert result["c_se"] > 0

        # Standard errors should be reasonable relative to parameters
        assert result["alpha_se"] < result["alpha"]  # Usually true
        assert result["c_se"] < result["c"]  # Usually true
"""
Test data simulation and random generation functions.
"""
import numpy as np
import pytest
from scipy import stats

from GaPaDist.mixtureGPDist import GammaPareto, simulate_gamma_pareto_mixture, ks_test_gpm


class TestSimulationFunctions:
    """Test simulation and generation functions."""

    def test_simulate_gamma_pareto_mixture_valid(self):
        """Test valid simulation."""
        n_sim = 1000
        alpha = 2.5
        c = 0.3
        theta = 10.0

        samples = simulate_gamma_pareto_mixture(n_sim, alpha, c, theta, random_state=42)

        assert len(samples) == n_sim
        assert isinstance(samples, np.ndarray)
        assert np.all(samples > theta)  # All values should be > theta
        assert samples.dtype == np.float64

    def test_simulate_gamma_pareto_mixture_reproducibility(self):
        """Test that random_state ensures reproducibility."""
        params = (100, 2.0, 0.4, 5.0)

        samples1 = simulate_gamma_pareto_mixture(*params, random_state=42)
        samples2 = simulate_gamma_pareto_mixture(*params, random_state=42)
        samples3 = simulate_gamma_pareto_mixture(*params, random_state=43)

        np.testing.assert_array_equal(samples1, samples2)
        assert not np.array_equal(samples1, samples3)

    def test_simulate_gamma_pareto_mixture_invalid_n(self):
        """Test invalid n_sim parameter."""
        with pytest.raises(ValueError, match="n_sim must be positive"):
            simulate_gamma_pareto_mixture(0, 2.5, 0.3, 10.0)

        with pytest.raises(ValueError, match="n_sim must be positive"):
            simulate_gamma_pareto_mixture(-10, 2.5, 0.3, 10.0)

    def test_simulate_gamma_pareto_mixture_invalid_alpha(self):
        """Test invalid alpha parameter."""
        with pytest.raises(ValueError, match="alpha must be > 0"):
            simulate_gamma_pareto_mixture(100, 0, 0.3, 10.0)

        with pytest.raises(ValueError, match="alpha must be > 0"):
            simulate_gamma_pareto_mixture(100, -1.0, 0.3, 10.0)

    def test_simulate_gamma_pareto_mixture_invalid_c(self):
        """Test invalid c parameter."""
        with pytest.raises(ValueError, match="c must be > 0"):
            simulate_gamma_pareto_mixture(100, 2.5, 0, 10.0)

        with pytest.raises(ValueError, match="c must be > 0"):
            simulate_gamma_pareto_mixture(100, 2.5, -0.1, 10.0)

    def test_simulate_gamma_pareto_mixture_invalid_theta(self):
        """Test invalid theta parameter."""
        with pytest.raises(ValueError, match="theta must be > 0"):
            simulate_gamma_pareto_mixture(100, 2.5, 0.3, 0)

        with pytest.raises(ValueError, match="theta must be > 0"):
            simulate_gamma_pareto_mixture(100, 2.5, 0.3, -5.0)

    def test_simulate_gamma_pareto_mixture_distribution_properties(self, sample_params):
        """Test that simulated data has correct distribution properties."""
        n_sim = 5000
        alpha = sample_params["alpha"]
        c = sample_params["c"]
        theta = sample_params["theta"]

        samples = simulate_gamma_pareto_mixture(n_sim, alpha, c, theta, random_state=42)

        # Transform back to Gamma distribution
        y = np.log(samples / theta)

        # Check Gamma distribution properties
        assert np.abs(np.mean(y) - alpha * c) < 0.1  # Mean of Gamma
        assert np.abs(np.var(y) - alpha * c ** 2) < 0.1  # Variance of Gamma
        assert np.all(samples >= theta)  # All values ≥ theta

    def test_ks_test_gpm_valid_input(self):
        """Test KS test with valid input."""
        # Generate two similar datasets
        np.random.seed(42)
        data1 = np.random.gamma(2, 1, 100)
        data2 = np.random.gamma(2, 1, 100)

        result = ks_test_gpm(data1, data2)

        assert "statistic" in result
        assert "pvalue" in result
        assert "message" in result
        assert isinstance(result["statistic"], float)
        assert isinstance(result["pvalue"], float)
        assert isinstance(result["message"], str)
        assert 0 <= result["statistic"] <= 1
        assert 0 <= result["pvalue"] <= 1

    def test_ks_test_gpm_rejection_case(self):
        """Test KS test with clearly different distributions."""
        np.random.seed(42)
        data1 = np.random.normal(0, 1, 100)
        data2 = np.random.exponential(1, 100)

        result = ks_test_gpm(data1, data2)

        # Should reject null hypothesis
        assert "rejected" in result["message"]
        assert result["pvalue"] < 0.05

    def test_ks_test_gpm_empty_input(self):
        """Test KS test with empty arrays."""
        empty_array = np.array([])
        data = np.random.randn(10)

        with pytest.raises(Exception):
            ks_test_gpm(empty_array, data)

        with pytest.raises(Exception):
            ks_test_gpm(data, empty_array)

    def test_ks_test_gpm_single_value(self):
        """Test KS test with single value arrays."""
        result = ks_test_gpm([1.0], [1.0])
        assert np.isfinite(result["statistic"])
"""
Test edge cases and error handling.
"""
import numpy as np
import pandas as pd
import pytest

from GaPaDist.mixtureGPDist import GammaPareto, simulate_gamma_pareto_mixture, ks_test_gpm


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_value_dataset(self):
        """Test with single value dataset."""
        single_df = pd.DataFrame({"values": [5.0]})
        gp = GammaPareto(data=single_df, colname="values")

        # Should be able to initialize
        assert gp.get_datatype() == "Pandas"

        # But fitting should fail
        with pytest.raises(ValueError, match="Insufficient data points"):
            gp.fit(method="hybrid")

    def test_all_identical_values(self):
        """Test with all identical values."""
        identical_df = pd.DataFrame({"values": [10.0] * 10})
        gp = GammaPareto(data=identical_df, colname="values")

        # Fitting should fail
        with pytest.raises(ValueError, match="Insufficient data points"):
            gp.fit(method="hybrid")

    def test_negative_values_in_data(self):
        """Test with negative values in data (should fail for positive-only distribution)."""
        negative_df = pd.DataFrame({"values": [-1.0, 0.0, 1.0, 2.0]})
        gp = GammaPareto(data=negative_df, colname="values")

        # PDF calculation should handle this
        pdf_result = gp.pdf(alpha=2.0, c=0.5, theta=1.0)
        assert "x" in pdf_result
        assert "pdf" in pdf_result

    def test_very_large_dataset(self):
        """Test with very large dataset."""
        large_data = np.random.lognormal(2, 1, 10000)
        large_df = pd.DataFrame({"values": large_data})
        gp = GammaPareto(data=large_df, colname="values")

        # Should handle large datasets
        result = gp.fit(method="hybrid")
        assert result["alpha"] > 0
        assert result["c"] > 0
        assert result["theta"] > 0

    def test_nan_values_in_data(self):
        """Test with NaN values in data."""
        data_with_nan = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        nan_df = pd.DataFrame({"values": data_with_nan})

        # NaN handling depends on pandas/polars behavior
        gp = GammaPareto(data=nan_df, colname="values")

        # get_data_vector might propagate NaN
        data_vector = gp.get_data_vector()
        assert np.any(np.isnan(data_vector))

    def test_inf_values_in_data(self):
        """Test with infinite values in data."""
        data_with_inf = np.array([1.0, 2.0, np.inf, 4.0, 5.0])
        inf_df = pd.DataFrame({"values": data_with_inf})

        gp = GammaPareto(data=inf_df, colname="values")

        # This might cause issues in calculations
        with pytest.raises(Exception):
            gp.fit(method="hybrid")

    def test_extreme_parameter_values_pdf(self, example_gamma_pareto):
        """Test PDF with extreme parameter values."""
        # Very small parameters
        pdf_result = example_gamma_pareto.pdf(alpha=1e-6, c=1e-6, theta=1e-6)
        assert np.all(np.isfinite(pdf_result["pdf"]) | (pdf_result["pdf"] == 0))

        # Very large parameters
        pdf_result = example_gamma_pareto.pdf(alpha=1e6, c=1e6, theta=1e6)
        assert np.all(np.isfinite(pdf_result["pdf"]) | (pdf_result["pdf"] == 0))

    def test_extreme_parameter_values_cdf(self, example_gamma_pareto):
        """Test CDF with extreme parameter values."""
        # Very small parameters
        cdf_result = example_gamma_pareto.cdf(alpha=1e-6, c=1e-6, theta=1e-6)
        assert np.all((cdf_result["cdf"] >= 0) & (cdf_result["cdf"] <= 1))

        # Very large parameters
        cdf_result = example_gamma_pareto.cdf(alpha=1e6, c=1e6, theta=1e6)
        assert np.all((cdf_result["cdf"] >= 0) & (cdf_result["cdf"] <= 1))

    def test_moments_extreme_orders(self, example_gamma_pareto):
        """Test moments with extreme orders."""
        # Very high order moment (will likely be infinite)
        result = example_gamma_pareto.moments(
            r=100, alpha=2.5, c=0.3, theta=10.0
        )
        assert np.isinf(result["value"])
        assert "condition is not met" in result["message"]


    def test_column_names_with_special_characters(self):
        """Test with special characters in column names."""
        df = pd.DataFrame({"col-name.with.dots": [1.0, 2.0, 3.0]})

        gp = GammaPareto(data=df, colname="col-name.with.dots")
        assert gp.colname == "col-name.with.dots"
        assert len(gp.get_data_vector()) == 3

    def test_unicode_column_names(self):
        """Test with unicode characters in column names."""
        df = pd.DataFrame({"café_μσ": [1.0, 2.0, 3.0]})

        gp = GammaPareto(data=df, colname="café_μσ")
        assert gp.colname == "café_μσ"
        assert len(gp.get_data_vector()) == 3
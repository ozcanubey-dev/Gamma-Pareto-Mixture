"""
Test built-in example datasets.
"""
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from GaPaDist.mixtureGPDist import GammaPareto, simulate_gamma_pareto_mixture, ks_test_gpm


class TestExampleDatasets:
    """Test built-in example datasets."""

    def test_do_example_fatigue(self):
        """Test Fatigue dataset loading."""
        gp = GammaPareto.do_example("Fatigue")

        assert gp.colname == "Fatigue"
        assert gp.get_datatype() == "Pandas"

        data = gp.get_data_vector()
        assert len(data) == 101  # Fatigue dataset has 101 observations
        assert np.min(data) == 70
        assert np.max(data) == 212

    def test_do_example_flood(self):
        """Test Flood dataset loading."""
        gp = GammaPareto.do_example("Flood")

        assert gp.colname == "Flood"
        assert gp.get_datatype() == "Pandas"

        data = gp.get_data_vector()
        assert len(data) == 39  # Flood dataset has 39 observations
        assert np.min(data) == 318
        assert np.max(data) == 71500

    def test_do_example_tribolium(self):
        """Test Tribolium dataset loading."""
        gp = GammaPareto.do_example("Tribolium")

        assert gp.colname == "Tribolium"
        assert gp.get_datatype() == "Pandas"

        data = gp.get_data_vector()
        # Tribolium dataset has frequencies that sum to 370
        assert len(data) == 504
        assert np.min(data) == 55
        assert np.max(data) == 245

    def test_do_example_invalid_dataset(self):
        """Test invalid dataset name."""
        with pytest.raises(ValueError, match="must be"):
            GammaPareto.do_example("InvalidDataset")

    def test_example_fatigue_fitting(self):
        """Test fitting on Fatigue dataset."""
        gp = GammaPareto.do_example("Fatigue")
        result = gp.fit(method="hybrid")

        assert result["alpha"] > 0
        assert result["c"] > 0
        assert result["theta"] == 70  # Minimum of Fatigue dataset
        assert result["loglik"] > -np.inf

    def test_example_flood_fitting(self):
        """Test fitting on Flood dataset."""
        gp = GammaPareto.do_example("Flood")
        result = gp.fit(method="optimize")

        assert result["alpha"] > 0
        assert result["c"] > 0
        assert result["theta"] == 318  # Minimum of Flood dataset
        assert result["loglik"] > -np.inf

    def test_example_tribolium_fitting(self):
        """Test fitting on Tribolium dataset."""
        gp = GammaPareto.do_example("Tribolium")
        result = gp.fit(method="hybrid")

        assert result["alpha"] > 0
        assert result["c"] > 0
        assert result["theta"] == 55  # Minimum of Tribolium dataset
        assert result["loglik"] > -np.inf

    def test_example_datasets_consistency(self):
        """Test that example datasets are loaded consistently."""
        # Load datasets multiple times
        gp1 = GammaPareto.do_example("Fatigue")
        gp2 = GammaPareto.do_example("Fatigue")

        data1 = gp1.get_data_vector()
        data2 = gp2.get_data_vector()

        assert_array_equal(data1, data2)

    def test_example_dataset_statistics(self):
        """Test basic statistics of example datasets."""
        datasets = ["Fatigue", "Flood", "Tribolium"]

        for dataset_name in datasets:
            gp = GammaPareto.do_example(dataset_name)
            data = gp.get_data_vector()

            # Basic sanity checks
            assert len(data) > 0
            assert np.all(np.isfinite(data))
            assert np.min(data) > 0  # All values should be positive

            # Test PDF computation
            pdf_result = gp.pdf(alpha=2.0, c=0.5, theta=np.min(data))
            assert len(pdf_result["x"]) == 1000
            assert len(pdf_result["pdf"]) == 1000

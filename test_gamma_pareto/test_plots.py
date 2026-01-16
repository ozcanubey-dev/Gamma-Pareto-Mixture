"""
Test plotting and visualization functions.
"""
import matplotlib.pyplot as plt
import pytest
import os
import tempfile

from GaPaDist.mixtureGPDist import GammaPareto, simulate_gamma_pareto_mixture, ks_test_gpm


class TestPlottingFunctions:
    """Test plotting functionality."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        # Close any open figures before test
        plt.close("all")
        yield
        # Clean up after test
        plt.close("all")

    def test_plot_fit_creation(self, example_gamma_pareto, fitted_gamma_pareto):
        """Test plot creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            axes = example_gamma_pareto.plot_fit(
                alpha=fitted_gamma_pareto["alpha"],
                c=fitted_gamma_pareto["c"],
                theta=fitted_gamma_pareto["theta"],
                bins=30
            )

            # Check return value
            assert len(axes) == 3
            ax1, ax2, ax3 = axes

            # Check axes types
            assert isinstance(ax1, plt.Axes)
            assert isinstance(ax2, plt.Axes)
            assert isinstance(ax3, plt.Axes)


    def test_plot_fit_axes_labels(self, example_gamma_pareto, fitted_gamma_pareto):
        """Test plot axes labels."""
        axes = example_gamma_pareto.plot_fit(
            alpha=fitted_gamma_pareto["alpha"],
            c=fitted_gamma_pareto["c"],
            theta=fitted_gamma_pareto["theta"],            bins=20
        )

        ax1, ax2, ax3 = axes

        # Check axis labels
        assert ax1.get_xlabel() == "x"
        assert ax1.get_ylabel() == "Density"
        assert ax1.get_title() == "PDF Fit"

        assert ax2.get_xlabel() == "x"
        assert ax2.get_ylabel() == "Cumulative Probability"
        assert ax2.get_title() == "CDF Comparison"

        assert ax3.get_xlabel() == "Theoretical Quantiles"
        assert ax3.get_ylabel() == "Empirical Quantiles"
        assert ax3.get_title() == "Q-Q Plot"

    def test_plot_fit_legend(self, example_gamma_pareto, fitted_gamma_pareto):
        """Test plot legends."""
        axes = example_gamma_pareto.plot_fit(
            alpha=fitted_gamma_pareto["alpha"],
            c=fitted_gamma_pareto["c"],
            theta=fitted_gamma_pareto["theta"],
            bins=20
        )

        ax1, ax2, _ = axes

        # Check legends exist
        legend1 = ax1.get_legend()
        legend2 = ax2.get_legend()

        assert legend1 is not None
        assert legend2 is not None

        # Check legend labels
        legend_texts1 = [t.get_text() for t in legend1.get_texts()]
        legend_texts2 = [t.get_text() for t in legend2.get_texts()]

        assert "Data" in legend_texts1
        assert "Gamma-Pareto fit" in legend_texts1
        assert "Empirical CDF" in legend_texts2
        assert "Fitted CDF" in legend_texts2

    def test_plot_fit_grid(self, example_gamma_pareto, fitted_gamma_pareto):
        """Test plot grid settings."""
        axes = example_gamma_pareto.plot_fit(
            alpha=fitted_gamma_pareto["alpha"],
            c=fitted_gamma_pareto["c"],
            theta=fitted_gamma_pareto["theta"],
            bins=20
        )

        ax1, ax2, ax3 = axes

        # Check grid is enabled
        assert ax1.yaxis.get_gridlines()[0].get_visible()
        assert ax2.yaxis.get_gridlines()[0].get_visible()
        assert ax3.yaxis.get_gridlines()[0].get_visible()

    def test_plot_fit_different_bins(self, example_gamma_pareto, fitted_gamma_pareto):
        """Test plot with different bin counts."""
        for bins in [10, 20, 50]:
            axes = example_gamma_pareto.plot_fit(
                alpha=fitted_gamma_pareto["alpha"],
                c=fitted_gamma_pareto["c"],
                theta=fitted_gamma_pareto["theta"],
                bins=bins
            )

            ax1, _, _ = axes
            # Get histogram patches
            patches = ax1.patches
            assert len(patches) == bins

    def test_plot_fit_different_figsize(self, example_gamma_pareto, fitted_gamma_pareto):
        """Test plot with different figure sizes."""
        for figsize in [(6, 6), (8, 8), (10, 10)]:
            axes = example_gamma_pareto.plot_fit(
                alpha=fitted_gamma_pareto["alpha"],
                c=fitted_gamma_pareto["c"],
                theta=fitted_gamma_pareto["theta"],
                figsize=figsize
            )

            ax1, _, _ = axes
            fig = ax1.get_figure()

            # Check figure size (with some tolerance for DPI differences)
            actual_size = fig.get_size_inches()
            assert abs(actual_size[0] - figsize[0]) < 0.1
            assert abs(actual_size[1] - figsize[1]) < 0.1

    def test_plot_fit_qq_plot_line(self, example_gamma_pareto, fitted_gamma_pareto):
        """Test Q-Q plot reference line."""
        axes = example_gamma_pareto.plot_fit(
            alpha=fitted_gamma_pareto["alpha"],
            c=fitted_gamma_pareto["c"],
            theta=fitted_gamma_pareto["theta"],
            bins=20
        )

        _, _, ax3 = axes

        # Check for reference line (red dashed line)
        lines = ax3.get_lines()
        has_reference_line = any(
            line.get_linestyle() == "--" and
            line.get_color() in ["red", "#ff0000", "r"]
            for line in lines
        )
        assert has_reference_line
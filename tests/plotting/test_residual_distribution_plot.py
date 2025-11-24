"""Tests for ResidualDistributionPlot class."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from chemotools.plotting import ResidualDistributionPlot, is_displayable


class TestResidualDistributionPlotBasics:
    """Test basic functionality of ResidualDistributionPlot."""

    def test_implements_display_protocol(self):
        """Test that ResidualDistributionPlot implements Display protocol."""
        # Arrange
        residuals = np.random.randn(100)

        # Act
        plot = ResidualDistributionPlot(residuals)

        # Assert
        assert is_displayable(plot)

    def test_basic_plot_creation(self):
        """Test basic plot creation with residuals."""
        # Arrange
        residuals = np.random.randn(100)

        # Act
        plot = ResidualDistributionPlot(residuals)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_statistics_calculation(self):
        """Test that statistics are calculated correctly."""
        # Arrange
        residuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Act
        plot = ResidualDistributionPlot(residuals)

        # Assert
        assert hasattr(plot, "mean")
        assert hasattr(plot, "std")
        assert hasattr(plot, "skewness")
        assert hasattr(plot, "kurtosis")
        assert plot.mean == pytest.approx(3.0)

    def test_default_density_mode(self):
        """Test that default mode is density."""
        # Arrange
        residuals = np.random.randn(100)

        # Act
        plot = ResidualDistributionPlot(residuals)

        # Assert
        assert plot.density is True

    def test_count_mode(self):
        """Test count mode instead of density."""
        # Arrange
        residuals = np.random.randn(100)

        # Act
        plot = ResidualDistributionPlot(residuals, density=False)

        # Assert
        assert plot.density is False

    def test_normal_curve_added_by_default(self):
        """Test that normal curve is added by default."""
        # Arrange
        residuals = np.random.randn(50)

        # Act
        plot = ResidualDistributionPlot(residuals, add_normal_curve=True)

        # Assert
        assert plot.add_normal_curve is True

    def test_no_normal_curve_when_disabled(self):
        """Test that normal curve can be disabled."""
        # Arrange
        residuals = np.random.randn(50)

        # Act
        plot = ResidualDistributionPlot(residuals, add_normal_curve=False)

        # Assert
        assert plot.add_normal_curve is False

    def test_stats_box_added_by_default(self):
        """Test that stats box is added by default."""
        # Arrange
        residuals = np.random.randn(50)

        # Act
        plot = ResidualDistributionPlot(residuals, add_stats=True)

        # Assert
        assert plot.add_stats is True

    def test_no_stats_box_when_disabled(self):
        """Test that stats box can be disabled."""
        # Arrange
        residuals = np.random.randn(50)

        # Act
        plot = ResidualDistributionPlot(residuals, add_stats=False)

        # Assert
        assert plot.add_stats is False


class TestResidualDistributionPlotMultivariate:
    """Test multivariate regression support."""

    def test_multivariate_residuals_default_target(self):
        """Test with multivariate residuals, default target index."""
        # Arrange
        residuals = np.random.randn(100, 3)  # 3 targets

        # Act
        plot = ResidualDistributionPlot(residuals)

        # Assert
        assert plot.residuals.shape == (100, 3)
        assert plot.residuals_1d.shape == (100,)
        assert plot.target_index == 0
        np.testing.assert_array_equal(plot.residuals_1d, residuals[:, 0])

    def test_multivariate_residuals_custom_target(self):
        """Test with multivariate residuals, custom target index."""
        # Arrange
        residuals = np.random.randn(100, 3)

        # Act
        plot = ResidualDistributionPlot(residuals, target_index=2)

        # Assert
        assert plot.target_index == 2
        np.testing.assert_array_equal(plot.residuals_1d, residuals[:, 2])

    def test_multivariate_residuals_invalid_target_raises_error(self):
        """Test that invalid target index raises error."""
        # Arrange
        residuals = np.random.randn(100, 3)

        # Act & Assert
        with pytest.raises(ValueError, match="target_index 5 is out of bounds"):
            ResidualDistributionPlot(residuals, target_index=5)

    def test_univariate_residuals_ignores_target_index(self):
        """Test that target_index is ignored for 1D residuals."""
        # Arrange
        residuals = np.random.randn(100)

        # Act
        plot = ResidualDistributionPlot(residuals, target_index=5)  # Should be ignored

        # Assert
        assert plot.residuals_1d.shape == (100,)
        np.testing.assert_array_equal(plot.residuals_1d, residuals)


class TestResidualDistributionPlotValidation:
    """Test input validation."""

    def test_empty_residuals_raises_error(self):
        """Test that empty residuals array raises error."""
        # Arrange
        residuals = np.array([])

        # Act & Assert
        with pytest.raises(ValueError, match="Found array with 0 sample"):
            ResidualDistributionPlot(residuals)

    def test_too_few_residuals_raises_error(self):
        """Test that too few residuals raise error."""
        # Arrange
        residuals = np.array([1.0, 2.0])  # Only 2 points

        # Act & Assert
        with pytest.raises(ValueError, match="Need at least 3 residuals"):
            ResidualDistributionPlot(residuals)

    def test_3d_residuals_raises_error(self):
        """Test that 3D residuals raise error."""
        # Arrange
        residuals = np.random.randn(10, 5, 3)

        # Act & Assert
        with pytest.raises(ValueError, match="Found array with dim 3"):
            ResidualDistributionPlot(residuals)


class TestResidualDistributionPlotBinning:
    """Test histogram binning options."""

    def test_default_bins_auto(self):
        """Test that default bins is 'auto'."""
        # Arrange
        residuals = np.random.randn(100)

        # Act
        plot = ResidualDistributionPlot(residuals)

        # Assert
        assert plot.bins == "auto"

    def test_custom_number_of_bins(self):
        """Test with custom number of bins."""
        # Arrange
        residuals = np.random.randn(100)

        # Act
        plot = ResidualDistributionPlot(residuals, bins=20)
        fig = plot.show()

        # Assert
        assert plot.bins == 20
        plt.close(fig)

    def test_bins_strategy(self):
        """Test with different binning strategies."""
        # Arrange
        residuals = np.random.randn(100)

        # Act & Assert
        for strategy in ["auto", "fd", "sturges", "sqrt"]:
            plot = ResidualDistributionPlot(residuals, bins=strategy)
            fig = plot.show()
            assert plot.bins == strategy
            plt.close(fig)


class TestResidualDistributionPlotStyling:
    """Test styling options."""

    def test_custom_color(self):
        """Test with custom histogram color."""
        # Arrange
        residuals = np.random.randn(100)

        # Act
        plot = ResidualDistributionPlot(residuals, color="red")
        fig = plot.show()

        # Assert
        assert plot.color == "red"
        plt.close(fig)

    def test_custom_alpha(self):
        """Test with custom alpha value."""
        # Arrange
        residuals = np.random.randn(100)

        # Act
        plot = ResidualDistributionPlot(residuals, alpha=0.8)
        fig = plot.show()

        # Assert
        assert plot.alpha == 0.8
        plt.close(fig)


class TestResidualDistributionPlotRender:
    """Test render method."""

    def test_render_without_axes_creates_new(self):
        """Test that render without axes creates new figure."""
        # Arrange
        residuals = np.random.randn(100)
        plot = ResidualDistributionPlot(residuals)

        # Act
        fig, ax = plot.render()

        # Assert
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_render_with_existing_axes(self):
        """Test render on existing axes."""
        # Arrange
        residuals = np.random.randn(100)
        plot = ResidualDistributionPlot(residuals)
        fig, ax = plt.subplots()

        # Act
        result_fig, result_ax = plot.render(ax=ax)

        # Assert
        assert result_fig is fig
        assert result_ax is ax
        plt.close(fig)

    def test_render_with_limits(self):
        """Test render with custom axis limits."""
        # Arrange
        residuals = np.random.randn(100)
        plot = ResidualDistributionPlot(residuals)

        # Act
        fig, ax = plot.render(xlim=(-3, 3), ylim=(0, 0.5))

        # Assert
        assert ax.get_xlim() == (-3, 3)
        assert ax.get_ylim() == (0, 0.5)
        plt.close(fig)

    def test_render_sets_default_labels_density_mode(self):
        """Test that render sets default labels for density mode."""
        # Arrange
        residuals = np.random.randn(100)
        plot = ResidualDistributionPlot(residuals, density=True)
        fig, ax = plt.subplots()

        # Act
        plot.render(ax=ax)

        # Assert
        assert ax.get_xlabel() == "Residuals"
        assert ax.get_ylabel() == "Density"
        plt.close(fig)

    def test_render_sets_default_labels_count_mode(self):
        """Test that render sets default labels for count mode."""
        # Arrange
        residuals = np.random.randn(100)
        plot = ResidualDistributionPlot(residuals, density=False)
        fig, ax = plt.subplots()

        # Act
        plot.render(ax=ax)

        # Assert
        assert ax.get_xlabel() == "Residuals"
        assert ax.get_ylabel() == "Count"
        plt.close(fig)


class TestResidualDistributionPlotShow:
    """Test show method."""

    def test_show_with_default_title(self):
        """Test show with default title for univariate."""
        # Arrange
        residuals = np.random.randn(100)
        plot = ResidualDistributionPlot(residuals)

        # Act
        fig = plot.show()

        # Assert
        ax = fig.axes[0]
        assert "Residual Distribution" in ax.get_title()
        plt.close(fig)

    def test_show_with_multivariate_title(self):
        """Test show with default title for multivariate."""
        # Arrange
        residuals = np.random.randn(100, 3)
        plot = ResidualDistributionPlot(residuals, target_index=1)

        # Act
        fig = plot.show()

        # Assert
        ax = fig.axes[0]
        assert "Target 2" in ax.get_title()
        plt.close(fig)

    def test_show_with_custom_title(self):
        """Test show with custom title."""
        # Arrange
        residuals = np.random.randn(100)
        plot = ResidualDistributionPlot(residuals)

        # Act
        fig = plot.show(title="My Custom Title")

        # Assert
        ax = fig.axes[0]
        assert ax.get_title() == "My Custom Title"
        plt.close(fig)

    def test_show_with_custom_labels(self):
        """Test show with custom axis labels."""
        # Arrange
        residuals = np.random.randn(100)
        plot = ResidualDistributionPlot(residuals)

        # Act
        fig = plot.show(xlabel="Error", ylabel="Frequency")

        # Assert
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Error"
        assert ax.get_ylabel() == "Frequency"
        plt.close(fig)

    def test_show_with_figsize(self):
        """Test show with custom figure size."""
        # Arrange
        residuals = np.random.randn(100)
        plot = ResidualDistributionPlot(residuals)

        # Act
        fig = plot.show(figsize=(12, 8))

        # Assert
        assert fig.get_size_inches()[0] == 12
        assert fig.get_size_inches()[1] == 8
        plt.close(fig)


class TestResidualDistributionPlotStatistics:
    """Test statistical properties and display."""

    def test_statistics_for_normal_distribution(self):
        """Test statistics for normally distributed residuals."""
        # Arrange - generate perfectly normal residuals
        np.random.seed(42)
        residuals = np.random.randn(1000)

        # Act
        plot = ResidualDistributionPlot(residuals)

        # Assert - mean should be close to 0, skewness and kurtosis close to 0
        assert abs(plot.mean) < 0.1
        assert abs(plot.skewness) < 0.3
        assert abs(plot.kurtosis) < 0.5

    def test_statistics_for_skewed_distribution(self):
        """Test statistics for skewed residuals."""
        # Arrange - generate right-skewed residuals
        np.random.seed(42)
        residuals = np.random.exponential(scale=2.0, size=1000)

        # Act
        plot = ResidualDistributionPlot(residuals)

        # Assert - should have positive skewness
        assert plot.skewness > 0.5


class TestResidualDistributionPlotIntegration:
    """Integration tests with realistic scenarios."""

    def test_normal_residuals_workflow(self):
        """Test typical workflow with normal residuals."""
        # Arrange - simulate good regression with normal errors
        np.random.seed(42)
        residuals = np.random.randn(200) * 2.0

        # Act
        plot = ResidualDistributionPlot(
            residuals, bins=30, add_normal_curve=True, add_stats=True
        )
        fig = plot.show(title="Distribution: Normal Residuals")

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_multiple_targets_side_by_side(self):
        """Test comparing multiple targets side by side."""
        # Arrange
        residuals = np.random.randn(100, 3)

        # Act
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i in range(3):
            ResidualDistributionPlot(residuals, target_index=i, bins=20).render(axes[i])
            axes[i].set_title(f"Target {i + 1}")

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_count_histogram_without_normal_curve(self):
        """Test count histogram without normal overlay."""
        # Arrange
        residuals = np.random.randn(200)

        # Act
        plot = ResidualDistributionPlot(
            residuals, density=False, add_normal_curve=False, bins=25
        )
        fig = plot.show(title="Residual Counts", ylabel="Count")

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_minimal_histogram(self):
        """Test minimal histogram without extras."""
        # Arrange
        residuals = np.random.randn(100)

        # Act
        plot = ResidualDistributionPlot(
            residuals, add_normal_curve=False, add_stats=False
        )
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestResidualDistributionPlotEdgeCases:
    """Test edge cases."""

    def test_uniform_residuals(self):
        """Test with uniformly distributed residuals."""
        # Arrange
        np.random.seed(42)
        residuals = np.random.uniform(-2, 2, 500)

        # Act
        plot = ResidualDistributionPlot(residuals)
        fig = plot.show()

        # Assert - should still work, just won't look normal
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_constant_residuals(self):
        """Test with constant residuals (all same value)."""
        # Arrange
        residuals = np.ones(100) * 2.0

        # Act
        plot = ResidualDistributionPlot(residuals)

        # Assert
        assert plot.std == pytest.approx(0.0)
        assert plot.skewness == 0.0 or np.isnan(plot.skewness)
        assert plot.kurtosis == 0.0 or np.isnan(plot.kurtosis)

    def test_very_small_sample(self):
        """Test with very small sample (just above minimum)."""
        # Arrange
        residuals = np.array([1.0, 2.0, 3.0, 4.0])

        # Act
        plot = ResidualDistributionPlot(residuals, bins=2)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

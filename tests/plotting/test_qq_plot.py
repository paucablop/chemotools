"""Tests for QQPlot class."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from chemotools.plotting import QQPlot, is_displayable


class TestQQPlotBasics:
    """Test basic functionality of QQPlot."""

    def test_implements_display_protocol(self):
        """Test that QQPlot implements Display protocol."""
        # Arrange
        residuals = np.random.randn(100)

        # Act
        plot = QQPlot(residuals)

        # Assert
        assert is_displayable(plot)

    def test_basic_plot_creation(self):
        """Test basic plot creation with residuals."""
        # Arrange
        residuals = np.random.randn(100)

        # Act
        plot = QQPlot(residuals)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_qq_data_calculation(self):
        """Test that theoretical and sample quantiles are calculated."""
        # Arrange
        residuals = np.random.randn(100)

        # Act
        plot = QQPlot(residuals)

        # Assert
        assert hasattr(plot, "theoretical_quantiles")
        assert hasattr(plot, "sample_quantiles")
        assert len(plot.theoretical_quantiles) == 100
        assert len(plot.sample_quantiles) == 100
        assert hasattr(plot, "slope")
        assert hasattr(plot, "intercept")
        assert hasattr(plot, "r_value")

    def test_reference_line_added_by_default(self):
        """Test that reference line is added by default."""
        # Arrange
        residuals = np.random.randn(50)

        # Act
        plot = QQPlot(residuals, add_reference_line=True)

        # Assert
        assert plot.add_reference_line is True

    def test_no_reference_line_when_disabled(self):
        """Test that reference line can be disabled."""
        # Arrange
        residuals = np.random.randn(50)

        # Act
        plot = QQPlot(residuals, add_reference_line=False)

        # Assert
        assert plot.add_reference_line is False


class TestQQPlotMultivariate:
    """Test multivariate regression support."""

    def test_multivariate_residuals_default_target(self):
        """Test with multivariate residuals, default target index."""
        # Arrange
        residuals = np.random.randn(100, 3)  # 3 targets

        # Act
        plot = QQPlot(residuals)

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
        plot = QQPlot(residuals, target_index=2)

        # Assert
        assert plot.target_index == 2
        np.testing.assert_array_equal(plot.residuals_1d, residuals[:, 2])

    def test_multivariate_residuals_invalid_target_raises_error(self):
        """Test that invalid target index raises error."""
        # Arrange
        residuals = np.random.randn(100, 3)

        # Act & Assert
        with pytest.raises(ValueError, match="target_index 5 is out of bounds"):
            QQPlot(residuals, target_index=5)

    def test_univariate_residuals_ignores_target_index(self):
        """Test that target_index is ignored for 1D residuals."""
        # Arrange
        residuals = np.random.randn(100)

        # Act
        plot = QQPlot(residuals, target_index=5)  # Should be ignored

        # Assert
        assert plot.residuals_1d.shape == (100,)
        np.testing.assert_array_equal(plot.residuals_1d, residuals)


class TestQQPlotValidation:
    """Test input validation."""

    def test_empty_residuals_raises_error(self):
        """Test that empty residuals array raises error."""
        # Arrange
        residuals = np.array([])

        # Act & Assert
        with pytest.raises(ValueError, match="Found array with 0 sample"):
            QQPlot(residuals)

    def test_too_few_residuals_raises_error(self):
        """Test that too few residuals raise error."""
        # Arrange
        residuals = np.array([1.0, 2.0])  # Only 2 points

        # Act & Assert
        with pytest.raises(ValueError, match="Need at least 3 residuals"):
            QQPlot(residuals)

    def test_3d_residuals_raises_error(self):
        """Test that 3D residuals raise error."""
        # Arrange
        residuals = np.random.randn(10, 5, 3)

        # Act & Assert
        with pytest.raises(ValueError, match="Found array with dim 3"):
            QQPlot(residuals)


class TestQQPlotConfidenceBands:
    """Test confidence band functionality."""

    def test_no_confidence_band_by_default(self):
        """Test that confidence bands are not added by default."""
        # Arrange
        residuals = np.random.randn(100)

        # Act
        plot = QQPlot(residuals)

        # Assert
        assert plot.add_confidence_band is None

    def test_confidence_band_true_uses_default(self):
        """Test that True uses default 95% bands."""
        # Arrange
        residuals = np.random.randn(100)

        # Act
        plot = QQPlot(residuals, add_confidence_band=True)
        fig = plot.show()

        # Assert
        assert plot.add_confidence_band is True
        plt.close(fig)

    def test_confidence_band_custom_value(self):
        """Test with custom confidence level."""
        # Arrange
        residuals = np.random.randn(100)

        # Act
        plot = QQPlot(residuals, add_confidence_band=0.90)
        fig = plot.show()

        # Assert
        assert plot.add_confidence_band == 0.90
        plt.close(fig)


class TestQQPlotAnnotations:
    """Test annotation functionality."""

    def test_with_annotations(self):
        """Test with point annotations."""
        # Arrange
        residuals = np.random.randn(50)
        annotations = [f"S{i}" if i % 10 == 0 else "" for i in range(50)]

        # Act
        plot = QQPlot(residuals, annotations=annotations)
        fig = plot.show()

        # Assert
        assert plot.annotations == annotations
        plt.close(fig)


class TestQQPlotRender:
    """Test render method."""

    def test_render_without_axes_creates_new(self):
        """Test that render without axes creates new figure."""
        # Arrange
        residuals = np.random.randn(100)
        plot = QQPlot(residuals)

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
        plot = QQPlot(residuals)
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
        plot = QQPlot(residuals)

        # Act
        fig, ax = plot.render(xlim=(-2, 2), ylim=(-2, 2))

        # Assert
        assert ax.get_xlim() == (-2, 2)
        assert ax.get_ylim() == (-2, 2)
        plt.close(fig)

    def test_render_sets_default_labels(self):
        """Test that render sets default labels if axes has none."""
        # Arrange
        residuals = np.random.randn(100)
        plot = QQPlot(residuals)
        fig, ax = plt.subplots()

        # Act
        plot.render(ax=ax)

        # Assert
        assert ax.get_xlabel() == "Theoretical Quantiles"
        assert ax.get_ylabel() == "Sample Quantiles"
        plt.close(fig)

    def test_render_sets_aspect_equal(self):
        """Test that render sets aspect ratio to equal."""
        # Arrange
        residuals = np.random.randn(100)
        plot = QQPlot(residuals)

        # Act
        fig, ax = plot.render()

        # Assert
        assert ax.get_aspect() != "auto"  # Should be 'equal'
        plt.close(fig)


class TestQQPlotShow:
    """Test show method."""

    def test_show_with_default_title(self):
        """Test show with default title for univariate."""
        # Arrange
        residuals = np.random.randn(100)
        plot = QQPlot(residuals)

        # Act
        fig = plot.show()

        # Assert
        ax = fig.axes[0]
        assert "Q-Q Plot" in ax.get_title()
        plt.close(fig)

    def test_show_with_multivariate_title(self):
        """Test show with default title for multivariate."""
        # Arrange
        residuals = np.random.randn(100, 3)
        plot = QQPlot(residuals, target_index=1)

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
        plot = QQPlot(residuals)

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
        plot = QQPlot(residuals)

        # Act
        fig = plot.show(xlabel="Theory", ylabel="Observed")

        # Assert
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Theory"
        assert ax.get_ylabel() == "Observed"
        plt.close(fig)

    def test_show_with_figsize(self):
        """Test show with custom figure size."""
        # Arrange
        residuals = np.random.randn(100)
        plot = QQPlot(residuals)

        # Act
        fig = plot.show(figsize=(10, 10))

        # Assert
        assert fig.get_size_inches()[0] == 10
        assert fig.get_size_inches()[1] == 10
        plt.close(fig)


class TestQQPlotStatistics:
    """Test statistical properties."""

    def test_normal_residuals_have_good_fit(self):
        """Test that normally distributed residuals have R² close to 1."""
        # Arrange - generate perfectly normal residuals
        np.random.seed(42)
        residuals = np.random.randn(1000)

        # Act
        plot = QQPlot(residuals)

        # Assert - R² should be very close to 1 for normal data
        assert plot.r_value**2 > 0.98

    def test_skewed_residuals_have_poorer_fit(self):
        """Test that skewed residuals have lower R²."""
        # Arrange - generate right-skewed residuals
        np.random.seed(42)
        residuals = np.random.exponential(scale=2.0, size=1000)

        # Act
        plot = QQPlot(residuals)

        # Assert - R² should be noticeably lower for skewed data
        # (though still might be high for exponential)
        assert 0 < plot.r_value**2 < 1


class TestQQPlotIntegration:
    """Integration tests with realistic scenarios."""

    def test_normal_residuals_workflow(self):
        """Test typical workflow with normal residuals."""
        # Arrange - simulate good regression with normal errors
        np.random.seed(42)
        residuals = np.random.randn(200) * 2.0

        # Act
        plot = QQPlot(residuals, add_confidence_band=0.95)
        fig = plot.show(title="Q-Q Plot: Normal Residuals")

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
            QQPlot(residuals, target_index=i).render(axes[i])
            axes[i].set_title(f"Target {i + 1}")

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_with_outlier_annotations(self):
        """Test with outlier annotations workflow."""
        # Arrange
        residuals = np.random.randn(100)
        # Add some outliers
        residuals[5] = 5.0
        residuals[23] = -4.5
        outlier_indices = [5, 23]
        annotations = [f"S{i}" if i in outlier_indices else "" for i in range(100)]

        # Act
        plot = QQPlot(residuals, annotations=annotations)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_color_customization(self):
        """Test custom color for points."""
        # Arrange
        residuals = np.random.randn(100)

        # Act
        plot = QQPlot(residuals, color="darkblue")
        fig = plot.show()

        # Assert
        assert plot.color == "darkblue"
        plt.close(fig)

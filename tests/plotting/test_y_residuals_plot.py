"""Tests for YResidualsPlot class."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from chemotools.plotting import YResidualsPlot, is_displayable


class TestYResidualsPlotBasics:
    """Test basic functionality of YResidualsPlot."""

    def test_implements_display_protocol(self):
        """Test that YResidualsPlot implements Display protocol."""
        # Arrange
        residuals = np.random.randn(100)

        # Act
        plot = YResidualsPlot(residuals)

        # Assert
        assert is_displayable(plot)

    def test_basic_plot_creation(self):
        """Test basic plot creation with residuals."""
        # Arrange
        residuals = np.random.randn(100)

        # Act
        plot = YResidualsPlot(residuals)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_default_x_axis_is_sample_index(self):
        """Test that default x-axis is sample index."""
        # Arrange
        residuals = np.random.randn(50)

        # Act
        plot = YResidualsPlot(residuals)

        # Assert
        np.testing.assert_array_equal(plot.x_axis, np.arange(50))
        assert plot.x_label == "Sample Index"

    def test_custom_x_values(self):
        """Test with custom x-axis values."""
        # Arrange
        residuals = np.random.randn(50)
        x_values = np.linspace(0, 10, 50)

        # Act
        plot = YResidualsPlot(residuals, x_values=x_values)

        # Assert
        np.testing.assert_array_equal(plot.x_axis, x_values)
        assert plot.x_label == "X Values"

    def test_zero_line_added_by_default(self):
        """Test that zero reference line is added by default."""
        # Arrange
        residuals = np.random.randn(50)

        # Act
        plot = YResidualsPlot(residuals, add_zero_line=True)

        # Assert
        assert plot.add_zero_line is True

    def test_no_zero_line_when_disabled(self):
        """Test that zero line can be disabled."""
        # Arrange
        residuals = np.random.randn(50)

        # Act
        plot = YResidualsPlot(residuals, add_zero_line=False)

        # Assert
        assert plot.add_zero_line is False


class TestYResidualsPlotMultivariate:
    """Test multivariate regression support."""

    def test_multivariate_residuals_default_target(self):
        """Test with multivariate residuals, default target index."""
        # Arrange
        residuals = np.random.randn(100, 3)  # 3 targets

        # Act
        plot = YResidualsPlot(residuals)

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
        plot = YResidualsPlot(residuals, target_index=2)

        # Assert
        assert plot.target_index == 2
        np.testing.assert_array_equal(plot.residuals_1d, residuals[:, 2])

    def test_multivariate_residuals_invalid_target_raises_error(self):
        """Test that invalid target index raises error."""
        # Arrange
        residuals = np.random.randn(100, 3)

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid target_index 5"):
            YResidualsPlot(residuals, target_index=5)

    def test_univariate_residuals_ignores_target_index(self):
        """Test that target_index is ignored for 1D residuals."""
        # Arrange
        residuals = np.random.randn(100)

        # Act
        plot = YResidualsPlot(residuals, target_index=5)  # Should be ignored

        # Assert
        assert plot.residuals_1d.shape == (100,)
        np.testing.assert_array_equal(plot.residuals_1d, residuals)


class TestYResidualsPlotValidation:
    """Test input validation."""

    def test_empty_residuals_raises_error(self):
        """Test that empty residuals array raises error."""
        # Arrange
        residuals = np.array([])

        # Act & Assert
        with pytest.raises(ValueError, match="Found array with 0 sample"):
            YResidualsPlot(residuals)

    def test_3d_residuals_raises_error(self):
        """Test that 3D residuals raise error."""
        # Arrange
        residuals = np.random.randn(10, 5, 3)

        # Act & Assert
        with pytest.raises(ValueError, match="Found array with dim 3"):
            YResidualsPlot(residuals)

    def test_mismatched_x_values_length_raises_error(self):
        """Test that mismatched x_values length raises error."""
        # Arrange
        residuals = np.random.randn(100)
        x_values = np.linspace(0, 10, 50)  # Wrong length

        # Act & Assert
        with pytest.raises(ValueError, match="x_values length .* must match"):
            YResidualsPlot(residuals, x_values=x_values)


class TestYResidualsPlotConfidenceBands:
    """Test confidence band functionality."""

    def test_no_confidence_band_by_default(self):
        """Test that confidence bands are not added by default."""
        # Arrange
        residuals = np.random.randn(100)

        # Act
        plot = YResidualsPlot(residuals)

        # Assert
        assert plot.add_confidence_band is None

    def test_confidence_band_true_uses_default(self):
        """Test that True uses default ±2σ bands."""
        # Arrange
        residuals = np.random.randn(100)

        # Act
        plot = YResidualsPlot(residuals, add_confidence_band=True)
        fig = plot.show()

        # Assert
        assert plot.add_confidence_band is True
        plt.close(fig)

    def test_confidence_band_custom_value(self):
        """Test with custom confidence band value."""
        # Arrange
        residuals = np.random.randn(100)

        # Act
        plot = YResidualsPlot(residuals, add_confidence_band=3.0)
        fig = plot.show()

        # Assert
        assert plot.add_confidence_band == 3.0
        plt.close(fig)


class TestYResidualsPlotColoring:
    """Test coloring functionality."""

    def test_categorical_coloring(self):
        """Test with categorical color_by."""
        # Arrange
        residuals = np.random.randn(100)
        classes = np.array(["A"] * 40 + ["B"] * 30 + ["C"] * 30)

        # Act
        plot = YResidualsPlot(residuals, color_by=classes)

        # Assert
        assert plot.is_categorical is True
        assert plot.color_by is not None

    def test_continuous_coloring(self):
        """Test with continuous color_by."""
        # Arrange
        residuals = np.random.randn(100)
        color_by = np.linspace(0, 1, 100)

        # Act
        plot = YResidualsPlot(residuals, color_by=color_by)

        # Assert
        assert plot.is_categorical is False

    def test_single_color(self):
        """Test with single color specified."""
        # Arrange
        residuals = np.random.randn(100)

        # Act
        plot = YResidualsPlot(residuals, color="red")
        fig = plot.show()

        # Assert
        assert plot.color == "red"
        plt.close(fig)


class TestYResidualsPlotAnnotations:
    """Test annotation functionality."""

    def test_with_annotations(self):
        """Test with point annotations."""
        # Arrange
        residuals = np.random.randn(50)
        annotations = [f"S{i}" if i % 10 == 0 else "" for i in range(50)]

        # Act
        plot = YResidualsPlot(residuals, annotations=annotations)
        fig = plot.show()

        # Assert
        assert plot.annotations == annotations
        plt.close(fig)


class TestYResidualsPlotRender:
    """Test render method."""

    def test_render_without_axes_creates_new(self):
        """Test that render without axes creates new figure."""
        # Arrange
        residuals = np.random.randn(100)
        plot = YResidualsPlot(residuals)

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
        plot = YResidualsPlot(residuals)
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
        plot = YResidualsPlot(residuals)

        # Act
        fig, ax = plot.render(xlim=(0, 50), ylim=(-2, 2))

        # Assert
        assert ax.get_xlim() == (0, 50)
        assert ax.get_ylim() == (-2, 2)
        plt.close(fig)

    def test_render_sets_default_labels(self):
        """Test that render sets default labels if axes has none."""
        # Arrange
        residuals = np.random.randn(100)
        plot = YResidualsPlot(residuals)
        fig, ax = plt.subplots()

        # Act
        plot.render(ax=ax)

        # Assert
        assert ax.get_xlabel() == "Sample Index"
        assert ax.get_ylabel() == "Residuals"
        plt.close(fig)


class TestYResidualsPlotShow:
    """Test show method."""

    def test_show_with_default_title(self):
        """Test show with default title for univariate."""
        # Arrange
        residuals = np.random.randn(100)
        plot = YResidualsPlot(residuals)

        # Act
        fig = plot.show()

        # Assert
        ax = fig.axes[0]
        assert "Residuals Plot" in ax.get_title()
        plt.close(fig)

    def test_show_with_multivariate_title(self):
        """Test show with default title for multivariate."""
        # Arrange
        residuals = np.random.randn(100, 3)
        plot = YResidualsPlot(residuals, target_index=1)

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
        plot = YResidualsPlot(residuals)

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
        plot = YResidualsPlot(residuals)

        # Act
        fig = plot.show(xlabel="Predicted Y", ylabel="Error")

        # Assert
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Predicted Y"
        assert ax.get_ylabel() == "Error"
        plt.close(fig)

    def test_show_with_figsize(self):
        """Test show with custom figure size."""
        # Arrange
        residuals = np.random.randn(100)
        plot = YResidualsPlot(residuals)

        # Act
        fig = plot.show(figsize=(12, 8))

        # Assert
        assert fig.get_size_inches()[0] == 12
        assert fig.get_size_inches()[1] == 8
        plt.close(fig)


class TestYResidualsPlotIntegration:
    """Integration tests with realistic scenarios."""

    def test_residuals_vs_predicted_workflow(self):
        """Test typical workflow: residuals vs predicted values."""
        # Arrange - simulate regression
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.3
        residuals = y_true - y_pred

        # Act
        plot = YResidualsPlot(residuals, x_values=y_pred, add_confidence_band=2.0)
        fig = plot.show(title="Residuals vs Predicted", xlabel="Predicted Values")

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_multiple_datasets_composition(self):
        """Test composing multiple datasets on same axes."""
        # Arrange
        train_residuals = np.random.randn(100)
        test_residuals = np.random.randn(50)

        # Act
        fig, ax = plt.subplots()
        YResidualsPlot(train_residuals, label="Train", color="blue").render(ax)
        YResidualsPlot(test_residuals, label="Test", color="red").render(ax)
        ax.legend()

        # Assert
        assert isinstance(fig, Figure)
        legend = ax.get_legend()
        assert legend is not None
        plt.close(fig)

    def test_with_outlier_annotations(self):
        """Test with outlier annotations workflow."""
        # Arrange
        residuals = np.random.randn(100)
        outlier_indices = [5, 23, 47]
        annotations = [f"S{i}" if i in outlier_indices else "" for i in range(100)]

        # Act
        plot = YResidualsPlot(residuals, annotations=annotations)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

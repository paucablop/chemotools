"""Tests for ExplainedVariancePlot class."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from chemotools.plotting import ExplainedVariancePlot, is_displayable


class TestExplainedVariancePlotBasics:
    """Test basic functionality of ExplainedVariancePlot."""

    def test_implements_display_protocol(self):
        """Test that ExplainedVariancePlot implements Display protocol."""
        # Arrange
        variance_ratios = np.array([0.45, 0.25, 0.15, 0.10, 0.05])

        # Act
        plot = ExplainedVariancePlot(variance_ratios)

        # Assert
        assert is_displayable(plot)

    def test_basic_plot_creation(self):
        """Test basic plot creation with variance ratios."""
        # Arrange
        variance_ratios = np.array([0.45, 0.25, 0.15, 0.10, 0.05])

        # Act
        plot = ExplainedVariancePlot(variance_ratios)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_cumulative_variance_calculation(self):
        """Test that cumulative variance is calculated correctly."""
        # Arrange
        variance_ratios = np.array([0.45, 0.25, 0.15, 0.10, 0.05])

        # Act
        plot = ExplainedVariancePlot(variance_ratios)

        # Assert
        expected_cumulative = np.array([0.45, 0.70, 0.85, 0.95, 1.00])
        np.testing.assert_array_almost_equal(
            plot.cumulative_variance, expected_cumulative
        )

    def test_default_threshold(self):
        """Test default threshold is 0.95."""
        # Arrange
        variance_ratios = np.array([0.45, 0.25, 0.15, 0.10, 0.05])

        # Act
        plot = ExplainedVariancePlot(variance_ratios)

        # Assert
        assert plot.threshold == 0.95

    def test_custom_threshold(self):
        """Test custom threshold value."""
        # Arrange
        variance_ratios = np.array([0.45, 0.25, 0.15, 0.10, 0.05])

        # Act
        plot = ExplainedVariancePlot(variance_ratios, threshold=0.90)

        # Assert
        assert plot.threshold == 0.90

    def test_no_threshold(self):
        """Test with threshold=None."""
        # Arrange
        variance_ratios = np.array([0.45, 0.25, 0.15, 0.10, 0.05])

        # Act
        plot = ExplainedVariancePlot(variance_ratios, threshold=None)
        fig = plot.show()

        # Assert
        assert plot.threshold is None
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_default_axis_labels(self):
        """Test default axis labels."""
        # Arrange
        variance_ratios = np.array([0.45, 0.25, 0.15, 0.10, 0.05])

        # Act
        plot = ExplainedVariancePlot(variance_ratios)
        fig = plot.show()

        # Assert
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Component"
        assert ax.get_ylabel() == "Explained Variance Ratio"
        plt.close(fig)

    def test_custom_axis_labels(self):
        """Test custom axis labels."""
        # Arrange
        variance_ratios = np.array([0.45, 0.25, 0.15, 0.10, 0.05])

        # Act
        plot = ExplainedVariancePlot(variance_ratios)
        fig = plot.show(
            xlabel="Principal Component",
            ylabel="Variance Explained (%)",
        )

        # Assert
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Principal Component"
        assert ax.get_ylabel() == "Variance Explained (%)"
        plt.close(fig)

    def test_show_with_title(self):
        """Test show() with custom title."""
        # Arrange
        variance_ratios = np.array([0.45, 0.25, 0.15, 0.10, 0.05])
        plot = ExplainedVariancePlot(variance_ratios)

        # Act
        fig = plot.show(title="PCA Explained Variance")

        # Assert
        ax = fig.axes[0]
        assert ax.get_title() == "PCA Explained Variance"
        plt.close(fig)

    def test_show_with_figsize(self):
        """Test show() with custom figure size."""
        # Arrange
        variance_ratios = np.array([0.45, 0.25, 0.15, 0.10, 0.05])
        plot = ExplainedVariancePlot(variance_ratios)

        # Act
        fig = plot.show(figsize=(12, 6))

        # Assert
        assert fig.get_size_inches()[0] == 12
        assert fig.get_size_inches()[1] == 6
        plt.close(fig)

    def test_render_returns_axes(self):
        """Test that render() returns Figure and Axes object."""
        # Arrange
        variance_ratios = np.array([0.45, 0.25, 0.15, 0.10, 0.05])
        plot = ExplainedVariancePlot(variance_ratios)

        # Act
        fig, ax = plot.render()

        # Assert
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_render_on_existing_axes(self):
        """Test rendering on existing axes."""
        # Arrange
        variance_ratios = np.array([0.45, 0.25, 0.15, 0.10, 0.05])
        plot = ExplainedVariancePlot(variance_ratios)
        fig, ax = plt.subplots()

        # Act
        result_fig, result_ax = plot.render(ax=ax)

        # Assert
        assert result_ax is ax
        assert result_fig is fig
        plt.close(fig)


class TestExplainedVariancePlotValidation:
    """Test input validation for ExplainedVariancePlot."""

    def test_empty_array_raises_error(self):
        """Test that empty array raises ValueError."""
        # Arrange
        variance_ratios = np.array([])

        # Act & Assert
        with pytest.raises(ValueError, match="Found array with 0 sample"):
            ExplainedVariancePlot(variance_ratios)

    def test_2d_array_raises_error(self):
        """Test that 2D array raises ValueError."""
        # Arrange
        variance_ratios = np.array([[0.45, 0.25], [0.15, 0.10]])

        # Act & Assert
        with pytest.raises(ValueError, match="must be 1D"):
            ExplainedVariancePlot(variance_ratios)

    def test_threshold_too_low_raises_error(self):
        """Test that threshold <= 0 raises ValueError."""
        # Arrange
        variance_ratios = np.array([0.45, 0.25, 0.15, 0.10, 0.05])

        # Act & Assert
        with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
            ExplainedVariancePlot(variance_ratios, threshold=0.0)

    def test_threshold_too_high_raises_error(self):
        """Test that threshold > 1 raises ValueError."""
        # Arrange
        variance_ratios = np.array([0.45, 0.25, 0.15, 0.10, 0.05])

        # Act & Assert
        with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
            ExplainedVariancePlot(variance_ratios, threshold=1.5)

    def test_list_input_converted_to_array(self):
        """Test that list input is converted to numpy array."""
        # Arrange
        variance_ratios = [0.45, 0.25, 0.15, 0.10, 0.05]

        # Act
        plot = ExplainedVariancePlot(variance_ratios)

        # Assert
        assert isinstance(plot.explained_variance_ratio, np.ndarray)
        np.testing.assert_array_equal(plot.explained_variance_ratio, variance_ratios)


class TestExplainedVariancePlotRendering:
    """Test rendering details of ExplainedVariancePlot."""

    def test_plot_has_legend(self):
        """Test that plot has a legend."""
        # Arrange
        variance_ratios = np.array([0.45, 0.25, 0.15, 0.10, 0.05])
        plot = ExplainedVariancePlot(variance_ratios)

        # Act
        fig = plot.show()

        # Assert
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None
        plt.close(fig)

    def test_plot_has_grid(self):
        """Test that plot has grid lines."""
        # Arrange
        variance_ratios = np.array([0.45, 0.25, 0.15, 0.10, 0.05])
        plot = ExplainedVariancePlot(variance_ratios)

        # Act
        fig, ax = plot.render()

        # Assert
        assert ax.yaxis.get_gridlines()[0].get_visible()
        plt.close(fig)

    def test_threshold_line_present(self):
        """Test that threshold line is drawn."""
        # Arrange
        variance_ratios = np.array([0.45, 0.25, 0.15, 0.10, 0.05])
        plot = ExplainedVariancePlot(variance_ratios, threshold=0.95)

        # Act
        fig = plot.show()

        # Assert
        ax = fig.axes[0]
        # Check for horizontal line (threshold)
        lines = ax.get_lines()
        assert len(lines) >= 2  # At least cumulative line + threshold line
        plt.close(fig)

    def test_no_threshold_line_when_none(self):
        """Test that no threshold line is drawn when threshold=None."""
        # Arrange
        variance_ratios = np.array([0.45, 0.25, 0.15, 0.10, 0.05])
        plot = ExplainedVariancePlot(variance_ratios, threshold=None)

        # Act
        fig = plot.show()

        # Assert
        ax = fig.axes[0]
        lines = ax.get_lines()
        # Should only have cumulative variance line, no threshold
        assert len(lines) == 1
        plt.close(fig)

    def test_custom_bar_kwargs(self):
        """Test custom bar plot styling."""
        # Arrange
        variance_ratios = np.array([0.45, 0.25, 0.15, 0.10, 0.05])
        plot = ExplainedVariancePlot(variance_ratios)

        # Act
        fig = plot.show(bar_kwargs={"color": "orange", "alpha": 0.8})

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_line_kwargs(self):
        """Test custom line plot styling."""
        # Arrange
        variance_ratios = np.array([0.45, 0.25, 0.15, 0.10, 0.05])
        plot = ExplainedVariancePlot(variance_ratios)

        # Act
        fig = plot.show(line_kwargs={"color": "blue", "linewidth": 3})

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_xlim_applied(self):
        """Test that xlim is applied correctly."""
        # Arrange
        variance_ratios = np.array([0.45, 0.25, 0.15, 0.10, 0.05])
        plot = ExplainedVariancePlot(variance_ratios)

        # Act
        fig = plot.show(xlim=(0, 6))

        # Assert
        ax = fig.axes[0]
        assert ax.get_xlim() == (0, 6)
        plt.close(fig)

    def test_ylim_applied(self):
        """Test that ylim is applied correctly."""
        # Arrange
        variance_ratios = np.array([0.45, 0.25, 0.15, 0.10, 0.05])
        plot = ExplainedVariancePlot(variance_ratios)

        # Act
        fig = plot.show(ylim=(0, 1.2))

        # Assert
        ax = fig.axes[0]
        assert ax.get_ylim() == (0, 1.2)
        plt.close(fig)


class TestExplainedVariancePlotIntegration:
    """Test integration of ExplainedVariancePlot with decomposition models."""

    def test_pca_integration(self):
        """Test ExplainedVariancePlot with PCA model."""
        # Arrange
        from sklearn.decomposition import PCA

        np.random.seed(42)
        X = np.random.randn(50, 100)
        pca = PCA(n_components=10)
        pca.fit(X)

        # Act
        plot = ExplainedVariancePlot(pca.explained_variance_ratio_)
        fig = plot.show(title="PCA Explained Variance")

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_pls_x_space_integration(self):
        """Test ExplainedVariancePlot with PLS X-space variance."""
        # Arrange
        from chemotools.models import PLSRegression

        np.random.seed(42)
        X = np.random.randn(50, 100)
        y = np.random.randn(50)
        pls = PLSRegression(n_components=5)
        pls.fit(X, y)

        # Act
        plot = ExplainedVariancePlot(pls.explained_x_variance_ratio_)
        fig = plot.show(title="PLS Explained Variance in X")

        # Assert
        assert isinstance(fig, Figure)
        assert hasattr(pls, "explained_x_variance_ratio_")
        assert len(pls.explained_x_variance_ratio_) == 5
        plt.close(fig)

    def test_pls_y_space_integration(self):
        """Test ExplainedVariancePlot with PLS Y-space variance."""
        # Arrange
        from chemotools.models import PLSRegression

        np.random.seed(42)
        X = np.random.randn(50, 100)
        y = np.random.randn(50)
        pls = PLSRegression(n_components=5)
        pls.fit(X, y)

        # Act
        plot = ExplainedVariancePlot(pls.explained_y_variance_ratio_)
        fig = plot.show(title="PLS Explained Variance in Y")

        # Assert
        assert isinstance(fig, Figure)
        assert hasattr(pls, "explained_y_variance_ratio_")
        assert len(pls.explained_y_variance_ratio_) == 5
        plt.close(fig)

    def test_side_by_side_comparison(self):
        """Test creating side-by-side X and Y variance plots."""
        # Arrange
        from chemotools.models import PLSRegression

        np.random.seed(42)
        X = np.random.randn(50, 100)
        y = np.random.randn(50)
        pls = PLSRegression(n_components=5)
        pls.fit(X, y)

        # Act
        plot_x = ExplainedVariancePlot(pls.explained_x_variance_ratio_)
        plot_y = ExplainedVariancePlot(pls.explained_y_variance_ratio_)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plot_x.render(ax=axes[0])
        plot_y.render(ax=axes[1])
        axes[0].set_title("X-space")
        axes[1].set_title("Y-space")

        # Assert
        assert isinstance(fig, Figure)
        assert len(axes) == 2
        plt.close(fig)

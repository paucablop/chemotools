"""Tests for ScoresPlot class - simplified single-dataset API."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from chemotools.plotting import ScoresPlot


class TestScoresPlotBasics:
    """Test basic functionality of ScoresPlot."""

    def test_scores_plot_basic(self):
        """Test basic ScoresPlot initialization and rendering."""
        # Arrange
        scores = np.random.randn(50, 5)

        # Act
        plot = ScoresPlot(scores)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_multiple_datasets_composed(self):
        """Test plotting multiple datasets using composition."""
        # Arrange
        train_scores = np.random.randn(50, 5)
        test_scores = np.random.randn(30, 5)
        val_scores = np.random.randn(20, 5)

        # Act - compose multiple plots
        fig, ax = plt.subplots()
        ScoresPlot(train_scores, label="Train", components=(0, 1)).render(ax)
        ScoresPlot(test_scores, label="Test", components=(0, 1)).render(ax)
        ScoresPlot(val_scores, label="Val", components=(0, 1)).render(ax)
        ax.legend()  # Create legend after composing

        # Assert
        assert isinstance(fig, Figure)
        legend = ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) == 3  # Train, Test, Val
        plt.close(fig)

    def test_default_components(self):
        """Test that default components are (0, 1)."""
        # Arrange
        scores = np.random.randn(50, 5)

        # Act
        plot = ScoresPlot(scores)
        fig = plot.show()

        # Assert
        ax = fig.axes[0]
        assert "PC1" in ax.get_xlabel() or "Component 1" in ax.get_xlabel()
        assert "PC2" in ax.get_ylabel() or "Component 2" in ax.get_ylabel()
        plt.close(fig)

    def test_custom_components(self):
        """Test custom component selection."""
        # Arrange
        scores = np.random.randn(50, 10)

        # Act
        plot = ScoresPlot(scores, components=(2, 5))
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_labels(self):
        """Test custom axis labels."""
        # Arrange
        scores = np.random.randn(50, 5)
        xlabel = "Custom X"
        ylabel = "Custom Y"

        # Act
        plot = ScoresPlot(scores)
        fig = plot.show(xlabel=xlabel, ylabel=ylabel)

        # Assert
        ax = fig.axes[0]
        assert ax.get_xlabel() == xlabel
        assert ax.get_ylabel() == ylabel
        plt.close(fig)

    def test_with_label(self):
        """Test label parameter for legend."""
        # Arrange
        scores = np.random.randn(50, 5)

        # Act
        plot = ScoresPlot(scores, label="Dataset A")
        fig = plot.show()

        # Assert
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None
        plt.close(fig)

    def test_with_color(self):
        """Test custom color."""
        # Arrange
        scores = np.random.randn(50, 5)

        # Act
        plot = ScoresPlot(scores, color="red", label="Red Data")
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestScoresPlotComponentValidation:
    """Test component validation."""

    def test_component_out_of_bounds(self):
        """Test error when component indices exceed available components."""
        # Arrange
        scores = np.random.randn(50, 5)

        # Act & Assert
        with pytest.raises((ValueError, IndexError)):
            ScoresPlot(scores, components=(0, 10))

    def test_invalid_scores_shape(self):
        """Test error for 1D scores array."""
        # Arrange
        scores = np.random.randn(50)  # 1D array

        # Act & Assert
        with pytest.raises(ValueError, match="2D array"):
            ScoresPlot(scores)

    def test_invalid_components_length(self):
        """Test error when components has wrong length."""
        # Arrange
        scores = np.random.randn(50, 5)

        # Act & Assert
        with pytest.raises((ValueError, TypeError)):
            ScoresPlot(scores, components=(0, 1, 2))  # 3 components instead of 2


class TestScoresPlotCategoricalColoring:
    """Test categorical coloring."""

    def test_categorical_with_strings(self):
        """Test categorical coloring with string labels."""
        # Arrange
        scores = np.random.randn(30, 5)
        labels = np.array(["A"] * 10 + ["B"] * 10 + ["C"] * 10)

        # Act
        plot = ScoresPlot(scores, color_by=labels, colormap="tab10")
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_categorical_with_integers(self):
        """Test categorical detection with small number of unique integers."""
        # Arrange
        scores = np.random.randn(30, 5)
        labels = np.array([1, 2, 3] * 10)

        # Act
        plot = ScoresPlot(scores, color_by=labels, colormap="Set2")
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_categorical_with_few_unique_values(self):
        """Test automatic categorical detection with few unique values."""
        # Arrange
        scores = np.random.randn(50, 5)
        # 5 unique values - should be detected as categorical
        groups = np.array([0, 1, 2, 3, 4] * 10)

        # Act
        plot = ScoresPlot(scores, color_by=groups)
        fig = plot.show()

        # Assert - plot should work with categorical logic
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestScoresPlotContinuousColoring:
    """Test continuous coloring."""

    def test_continuous_with_floats(self):
        """Test continuous coloring with float values."""
        # Arrange
        scores = np.random.randn(50, 5)
        concentrations = np.linspace(0.1, 1.0, 50)

        # Act
        plot = ScoresPlot(scores, color_by=concentrations, colormap="viridis")
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        # Should have a colorbar
        assert len(fig.axes) > 1  # Main axis + colorbar axis
        plt.close(fig)

    def test_continuous_with_many_unique_values(self):
        """Test automatic continuous detection with many unique values."""
        # Arrange
        scores = np.random.randn(100, 5)
        # Many unique values - should be detected as continuous
        values = np.linspace(0, 100, 100)

        # Act
        plot = ScoresPlot(scores, color_by=values, colormap="plasma")
        fig = plot.show()

        # Assert - should have colorbar
        assert isinstance(fig, Figure)
        assert len(fig.axes) > 1
        plt.close(fig)


class TestScoresPlotAnnotations:
    """Test point annotations."""

    def test_annotations_list(self):
        """Test annotating points with a list of strings."""
        # Arrange
        scores = np.random.randn(10, 5)
        annotations = [f"Sample {i}" for i in range(10)]

        # Act
        plot = ScoresPlot(scores, annotations=annotations)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_annotations_array(self):
        """Test annotating points with numpy array."""
        # Arrange
        scores = np.random.randn(10, 5)
        annotations = np.array([f"S{i}" for i in range(10)])

        # Act
        plot = ScoresPlot(scores, annotations=annotations)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestScoresPlotCustomStyling:
    """Test custom styling options."""

    def test_custom_marker_size(self):
        """Test custom marker size via kwargs."""
        # Arrange
        scores = np.random.randn(50, 5)
        plot = ScoresPlot(scores)

        # Act
        fig, ax = plt.subplots()
        plot.render(ax, s=200)  # Large markers

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_alpha(self):
        """Test custom alpha (transparency) via kwargs."""
        # Arrange
        scores = np.random.randn(50, 5)
        plot = ScoresPlot(scores)

        # Act
        fig, ax = plt.subplots()
        plot.render(ax, alpha=0.3)

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_edge_colors(self):
        """Test custom edge colors and linewidths."""
        # Arrange
        scores = np.random.randn(50, 5)
        plot = ScoresPlot(scores)

        # Act
        fig, ax = plt.subplots()
        plot.render(ax, edgecolors="black", linewidths=1.5)

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestScoresPlotSubplots:
    """Test rendering on subplots."""

    def test_multiple_renders_on_subplots(self):
        """Test rendering multiple ScoresPlots on subplots."""
        # Arrange
        scores = np.random.randn(50, 5)
        plot1 = ScoresPlot(scores, components=(0, 1))
        plot2 = ScoresPlot(scores, components=(1, 2))

        # Act
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        plot1.render(ax1)
        plot2.render(ax2)

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_render_without_show(self):
        """Test that render can be used without calling show."""
        # Arrange
        scores = np.random.randn(50, 5)
        plot = ScoresPlot(scores)

        # Act
        fig, ax = plt.subplots()
        returned_fig, returned_ax = plot.render(ax)

        # Assert
        assert returned_fig is fig
        assert returned_ax is ax
        plt.close(fig)

    def test_comparison_subplots(self):
        """Test comparing different component pairs in subplots."""
        # Arrange
        scores = np.random.randn(50, 5)

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        # Act
        ScoresPlot(scores, components=(0, 1)).render(ax=axes[0, 0])
        ScoresPlot(scores, components=(1, 2)).render(ax=axes[0, 1])
        ScoresPlot(scores, components=(2, 3)).render(ax=axes[1, 0])
        ScoresPlot(scores, components=(3, 4)).render(ax=axes[1, 1])

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)


@pytest.mark.parametrize("components", [(0, 1), (1, 2), (0, 3), (2, 4)])
def test_different_component_pairs(components):
    """Test plotting different component pairs."""
    # Arrange
    scores = np.random.randn(50, 5)

    # Act
    plot = ScoresPlot(scores, components=components)
    fig = plot.show()

    # Assert
    assert isinstance(fig, Figure)
    plt.close(fig)


@pytest.mark.parametrize(
    "colormap,is_categorical",
    [
        ("tab10", True),
        ("Set2", True),
        ("Paired", True),
        ("viridis", False),
        ("plasma", False),
        ("cividis", False),
    ],
)
def test_colormap_with_appropriate_data(colormap, is_categorical):
    """Test different colormaps with appropriate data types."""
    # Arrange
    scores = np.random.randn(40, 5)

    if is_categorical:
        color_by = np.array(["A"] * 10 + ["B"] * 10 + ["C"] * 10 + ["D"] * 10)
    else:
        color_by = np.linspace(0.1, 1.0, 40)

    # Act
    plot = ScoresPlot(scores, color_by=color_by, colormap=colormap)
    fig = plot.show()

    # Assert
    assert isinstance(fig, Figure)
    plt.close(fig)


@pytest.mark.parametrize("n_samples", [10, 50, 100, 200])
def test_different_sample_sizes(n_samples):
    """Test plotting different numbers of samples."""
    # Arrange
    scores = np.random.randn(n_samples, 5)

    # Act
    plot = ScoresPlot(scores)
    fig = plot.show()

    # Assert
    assert isinstance(fig, Figure)
    plt.close(fig)


class TestScoresPlotRenderAxisLimits:
    """Test axis limit functionality."""

    def test_render_with_xlim(self):
        """Test render() with xlim parameter."""
        # Arrange
        scores = np.random.randn(50, 5)
        plot = ScoresPlot(scores)

        # Act
        fig, ax = plt.subplots()
        plot.render(ax, xlim=(-2, 2))

        # Assert
        assert ax.get_xlim() == (-2, 2)
        plt.close(fig)

    def test_render_with_ylim(self):
        """Test render() with ylim parameter."""
        # Arrange
        scores = np.random.randn(50, 5)
        plot = ScoresPlot(scores)

        # Act
        fig, ax = plt.subplots()
        plot.render(ax, ylim=(-3, 3))

        # Assert
        assert ax.get_ylim() == (-3, 3)
        plt.close(fig)

    def test_render_with_both_limits(self):
        """Test render() with both xlim and ylim."""
        # Arrange
        scores = np.random.randn(50, 5)
        plot = ScoresPlot(scores)

        # Act
        fig, ax = plt.subplots()
        plot.render(ax, xlim=(-2, 2), ylim=(-3, 3))

        # Assert
        assert ax.get_xlim() == (-2, 2)
        assert ax.get_ylim() == (-3, 3)
        plt.close(fig)


class TestScoresPlotConfidenceEllipse:
    """Test confidence ellipse functionality."""

    def test_confidence_ellipse_default(self):
        """Test confidence ellipse with default 95% confidence."""
        # Arrange
        scores = np.random.randn(50, 5)

        # Act
        plot = ScoresPlot(scores, confidence_ellipse=True)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        # Check that patches were added (ellipse is a patch)
        assert len(ax.patches) > 0
        plt.close(fig)

    def test_confidence_ellipse_custom_level(self):
        """Test confidence ellipse with custom confidence level."""
        # Arrange
        scores = np.random.randn(50, 5)

        # Act
        plot = ScoresPlot(scores, confidence_ellipse=0.99)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        assert len(ax.patches) > 0
        plt.close(fig)

    def test_confidence_ellipse_90_percent(self):
        """Test confidence ellipse with 90% confidence."""
        # Arrange
        scores = np.random.randn(50, 5)

        # Act
        plot = ScoresPlot(scores, confidence_ellipse=0.90)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_no_confidence_ellipse(self):
        """Test with no confidence ellipse (default behavior)."""
        # Arrange
        scores = np.random.randn(50, 5)

        # Act
        plot = ScoresPlot(scores, confidence_ellipse=False)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        # Should have no patches when ellipse is disabled
        assert len(ax.patches) == 0
        plt.close(fig)

    def test_confidence_ellipse_with_color(self):
        """Test confidence ellipse uses the same color as points."""
        # Arrange
        scores = np.random.randn(50, 5)

        # Act
        plot = ScoresPlot(scores, confidence_ellipse=True, color="red")
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        assert len(ax.patches) > 0
        plt.close(fig)

    def test_confidence_ellipse_composition(self):
        """Test confidence ellipses for multiple datasets."""
        # Arrange
        train_scores = np.random.randn(50, 5)
        test_scores = np.random.randn(30, 5)

        # Act - compose with ellipses
        fig, ax = plt.subplots()
        ScoresPlot(
            train_scores, label="Train", color="blue", confidence_ellipse=True
        ).render(ax)
        ScoresPlot(
            test_scores, label="Test", color="red", confidence_ellipse=0.99
        ).render(ax)
        ax.legend()

        # Assert
        assert isinstance(fig, Figure)
        # Should have 2 ellipses (patches)
        assert len(ax.patches) >= 2
        plt.close(fig)


class TestScoresPlotEdgeCases:
    """Test edge cases and error conditions for ScoresPlot."""

    def test_invalid_component_negative(self):
        """Test with negative component index."""
        scores = np.random.rand(50, 5)
        with pytest.raises(ValueError, match="Component index -1 is invalid"):
            ScoresPlot(scores, components=(-1, 1))

    def test_invalid_component_too_high(self):
        """Test with component index out of range."""
        scores = np.random.rand(50, 5)
        with pytest.raises(ValueError, match="Component index 10 is invalid"):
            ScoresPlot(scores, components=(0, 10))

    def test_same_components(self):
        """Test with same component for both axes."""
        scores = np.random.rand(50, 5)
        with pytest.raises(ValueError, match="Component indices must be different"):
            ScoresPlot(scores, components=(1, 1))

    def test_render_with_xlim_ylim(self):
        """Test render with custom xlim and ylim."""
        scores = np.random.rand(50, 5)
        plot = ScoresPlot(scores, components=(0, 1))
        fig, ax = plot.render(xlim=(-2, 2), ylim=(-3, 3))
        assert ax.get_xlim() == (-2, 2)
        assert ax.get_ylim() == (-3, 3)
        plt.close(fig)

    def test_render_with_continuous_colorby(self):
        """Test render with continuous color_by to trigger add_colorbar."""
        scores = np.random.rand(50, 5)
        color_by = np.random.rand(50)  # Continuous values
        plot = ScoresPlot(scores, components=(0, 1), color_by=color_by)
        fig, ax = plot.render()
        plt.close(fig)

    def test_render_with_existing_axes_no_labels(self):
        """Test render with existing axes that have no labels."""
        scores = np.random.rand(50, 5)
        plot = ScoresPlot(scores, components=(0, 1))
        fig, ax = plt.subplots()
        # Axes has no labels, so defaults should be set
        result_fig, result_ax = plot.render(ax=ax)
        assert result_ax.get_xlabel() == "PC1"
        assert result_ax.get_ylabel() == "PC2"
        plt.close(fig)

    def test_categorical_with_many_unique_values(self):
        """Test categorical detection with many unique values."""
        scores = np.random.rand(100, 5)
        # Integer with > 10 unique values should be continuous
        color_by = np.arange(100)
        plot = ScoresPlot(scores, components=(0, 1), color_by=color_by)
        # Should not be categorical
        assert not plot.is_categorical
        fig, ax = plot.render()
        plt.close(fig)

"""Tests for ScoresPlot class."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from chemotools.plotting import ScoresPlot, is_displayable


class TestScoresPlotBasics:
    """Test basic functionality of ScoresPlot."""

    def test_implements_display_protocol(self):
        """Test that ScoresPlot implements Display protocol."""
        # Arrange
        scores = {"train": np.random.randn(50, 5)}

        # Act
        plot = ScoresPlot(scores, components=(0, 1))

        # Assert
        assert is_displayable(plot)

    def test_single_dataset(self):
        """Test plotting a single dataset."""
        # Arrange
        scores = {"train": np.random.randn(50, 5)}

        # Act
        plot = ScoresPlot(scores, components=(0, 1))
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_multiple_datasets(self):
        """Test plotting multiple datasets (train/test/val)."""
        # Arrange
        scores = {
            "train": np.random.randn(50, 5),
            "test": np.random.randn(30, 5),
            "val": np.random.randn(20, 5),
        }

        # Act
        plot = ScoresPlot(scores, components=(0, 1))
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None
        plt.close(fig)

    def test_default_components(self):
        """Test default components (0, 1) - PC1 vs PC2."""
        # Arrange
        scores = {"train": np.random.randn(50, 5)}

        # Act
        plot = ScoresPlot(scores)  # No components specified
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        assert ax.get_xlabel() == "PC1"
        assert ax.get_ylabel() == "PC2"
        plt.close(fig)

    def test_custom_components(self):
        """Test custom component indices."""
        # Arrange
        scores = {"train": np.random.randn(50, 5)}

        # Act
        plot = ScoresPlot(scores, components=(1, 2))  # PC2 vs PC3
        fig = plot.show()

        # Assert
        ax = fig.axes[0]
        assert ax.get_xlabel() == "PC2"
        assert ax.get_ylabel() == "PC3"
        plt.close(fig)

    def test_custom_axis_labels(self):
        """Test custom x and y axis labels."""
        # Arrange
        scores = {"train": np.random.randn(50, 5)}

        # Act
        plot = ScoresPlot(
            scores,
            components=(0, 1),
            xlabel="First Latent Variable",
            ylabel="Second Latent Variable",
        )
        fig = plot.show()

        # Assert
        ax = fig.axes[0]
        assert ax.get_xlabel() == "First Latent Variable"
        assert ax.get_ylabel() == "Second Latent Variable"
        plt.close(fig)

    def test_show_with_title(self):
        """Test show() with custom title."""
        # Arrange
        scores = {"train": np.random.randn(50, 5)}
        plot = ScoresPlot(scores)

        # Act
        fig = plot.show(title="PCA Scores Plot")

        # Assert
        ax = fig.axes[0]
        assert ax.get_title() == "PCA Scores Plot"
        plt.close(fig)

    def test_show_with_figsize(self):
        """Test show() with custom figure size."""
        # Arrange
        scores = {"train": np.random.randn(50, 5)}
        plot = ScoresPlot(scores)

        # Act
        fig = plot.show(figsize=(10, 10))

        # Assert
        assert fig.get_size_inches()[0] == 10
        assert fig.get_size_inches()[1] == 10
        plt.close(fig)

    def test_render_returns_figure_and_axes(self):
        """Test that render() returns (Figure, Axes) tuple."""
        # Arrange
        scores = {"train": np.random.randn(50, 5)}
        plot = ScoresPlot(scores)

        # Act
        result = plot.render()

        # Assert
        assert isinstance(result, tuple)
        assert len(result) == 2
        fig, ax = result
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_render_with_existing_axes(self):
        """Test render() with existing axes."""
        # Arrange
        scores = {"train": np.random.randn(50, 5)}
        plot = ScoresPlot(scores)
        fig, ax = plt.subplots()

        # Act
        result_fig, result_ax = plot.render(ax=ax)

        # Assert
        assert result_fig is fig
        assert result_ax is ax
        plt.close(fig)

    def test_xlim_ylim(self):
        """Test xlim and ylim parameters."""
        # Arrange
        scores = {"train": np.random.randn(50, 5)}
        plot = ScoresPlot(scores)

        # Act
        fig = plot.show(xlim=(-5, 5), ylim=(-3, 3))

        # Assert
        ax = fig.axes[0]
        assert ax.get_xlim() == (-5, 5)
        assert ax.get_ylim() == (-3, 3)
        plt.close(fig)


class TestScoresPlotComponentValidation:
    """Test component validation."""

    def test_component_validation_at_init(self):
        """Test that invalid components raise error at initialization."""
        # Arrange
        scores = {"train": np.random.randn(50, 3)}  # Only 3 components

        # Act & Assert
        with pytest.raises(ValueError, match="Component index 5 is invalid"):
            ScoresPlot(scores, components=(0, 5))  # Component 5 doesn't exist

    def test_negative_component_index(self):
        """Test that negative component indices raise error."""
        # Arrange
        scores = {"train": np.random.randn(50, 5)}

        # Act & Assert
        with pytest.raises(ValueError, match="Component index -1 is invalid"):
            ScoresPlot(scores, components=(-1, 0))

    def test_same_component_twice(self):
        """Test that using same component for both axes raises error."""
        # Arrange
        scores = {"train": np.random.randn(50, 5)}

        # Act & Assert
        with pytest.raises(ValueError, match="Component indices must be different"):
            ScoresPlot(scores, components=(1, 1))

    def test_component_out_of_bounds_for_one_dataset(self):
        """Test error when components valid for one dataset but not another."""
        # Arrange
        scores = {
            "train": np.random.randn(50, 5),  # 5 components
            "test": np.random.randn(30, 3),  # Only 3 components
        }

        # Act & Assert
        with pytest.raises(ValueError, match="Component index 4 is invalid"):
            ScoresPlot(scores, components=(0, 4))  # Valid for train, not for test

    def test_empty_scores_dict(self):
        """Test that empty scores_dict raises error."""
        # Arrange
        scores = {}

        # Act & Assert
        with pytest.raises(ValueError, match="scores_dict cannot be empty"):
            ScoresPlot(scores)

    def test_1d_scores_array_raises_error(self):
        """Test that 1D scores array raises error."""
        # Arrange
        scores = {"train": np.random.randn(50)}  # 1D array

        # Act & Assert
        with pytest.raises(ValueError, match="must be 2D array"):
            ScoresPlot(scores)


class TestScoresPlotCategoricalColoring:
    """Test categorical coloring functionality."""

    def test_categorical_with_strings(self):
        """Test categorical coloring with string class labels."""
        # Arrange
        scores = {"train": np.random.randn(30, 5)}
        classes = np.array(["A"] * 10 + ["B"] * 10 + ["C"] * 10)
        color_by_dict = {"train": classes}

        # Act
        plot = ScoresPlot(scores, color_by_dict=color_by_dict, colormap="tab10")
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None
        plt.close(fig)

    def test_categorical_with_integers(self):
        """Test categorical coloring with integer labels (≤10 unique)."""
        # Arrange
        scores = {"train": np.random.randn(40, 5)}
        classes = np.array([1] * 10 + [2] * 10 + [3] * 10 + [4] * 10)
        color_by_dict = {"train": classes}

        # Act
        plot = ScoresPlot(scores, color_by_dict=color_by_dict)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_categorical_with_repeating_floats(self):
        """Test categorical detection with repeating float values."""
        # Arrange
        scores = {"train": np.random.randn(30, 5)}
        # Repeating values should be detected as categorical
        groups = np.array([1.0] * 10 + [2.0] * 10 + [3.0] * 10)
        color_by_dict = {"train": groups}

        # Act
        plot = ScoresPlot(scores, color_by_dict=color_by_dict)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_force_categorical_with_parameter(self):
        """Test forcing categorical treatment with categorical=True."""
        # Arrange
        scores = {"train": np.random.randn(20, 5)}
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 4)
        color_by_dict = {"train": values}

        # Act
        plot = ScoresPlot(
            scores, color_by_dict=color_by_dict, categorical=True, colormap="Set2"
        )
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestScoresPlotContinuousColoring:
    """Test continuous coloring functionality."""

    def test_continuous_with_floats(self):
        """Test continuous coloring with float values."""
        # Arrange
        scores = {"train": np.random.randn(50, 5)}
        concentrations = np.linspace(0.1, 1.0, 50)
        color_by_dict = {"train": concentrations}

        # Act
        plot = ScoresPlot(scores, color_by_dict=color_by_dict, colormap="viridis")
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        # Should have colorbar for continuous
        assert len(fig.axes) == 2  # Main axis + colorbar axis
        plt.close(fig)

    def test_continuous_custom_colorbar_label(self):
        """Test continuous coloring with custom colorbar label."""
        # Arrange
        scores = {"train": np.random.randn(40, 5)}
        concentrations = np.linspace(0.5, 5.0, 40)
        color_by_dict = {"train": concentrations}

        # Act
        plot = ScoresPlot(
            scores,
            color_by_dict=color_by_dict,
            colormap="plasma",
            colorbar_label="Concentration (g/L)",
        )
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2
        # Check colorbar has the custom label
        cbar_ax = fig.axes[1]
        assert cbar_ax.get_ylabel() == "Concentration (g/L)"
        plt.close(fig)

    def test_force_continuous_with_parameter(self):
        """Test forcing continuous treatment with categorical=False."""
        # Arrange
        scores = {"train": np.random.randn(25, 5)}
        # 5 unique integers might be detected as categorical
        levels = np.array([1, 2, 3, 4, 5] * 5)
        color_by_dict = {"train": levels}

        # Act
        plot = ScoresPlot(
            scores,
            color_by_dict=color_by_dict,
            categorical=False,
            colormap="viridis",
        )
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        # Should have colorbar for continuous
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_mixed_coloring_datasets(self):
        """Test some datasets with continuous coloring, others without."""
        # Arrange
        scores = {
            "train": np.random.randn(30, 5),
            "test": np.random.randn(20, 5),
            "val": np.random.randn(15, 5),
        }
        # Only color train set
        concentrations = np.linspace(0.1, 1.0, 30)
        color_by_dict = {"train": concentrations}

        # Act
        plot = ScoresPlot(scores, color_by_dict=color_by_dict, colormap="viridis")
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        # Should have colorbar
        assert len(fig.axes) == 2
        plt.close(fig)


class TestScoresPlotConfidenceEllipses:
    """Test confidence ellipse functionality."""

    def test_confidence_ellipse_default_95(self):
        """Test confidence ellipse with default 95% confidence."""
        # Arrange
        scores = {"train": np.random.randn(50, 5)}

        # Act
        plot = ScoresPlot(scores, confidence_ellipse=True)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_confidence_ellipse_custom_level(self):
        """Test confidence ellipse with custom confidence level."""
        # Arrange
        scores = {"train": np.random.randn(50, 5)}

        # Act
        plot = ScoresPlot(scores, confidence_ellipse=0.90)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_confidence_ellipse_99_percent(self):
        """Test confidence ellipse with 99% confidence."""
        # Arrange
        scores = {"train": np.random.randn(50, 5)}

        # Act
        plot = ScoresPlot(scores, confidence_ellipse=0.99)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_confidence_ellipse_no_ellipse(self):
        """Test with confidence_ellipse=False (no ellipse)."""
        # Arrange
        scores = {"train": np.random.randn(50, 5)}

        # Act
        plot = ScoresPlot(scores, confidence_ellipse=False)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_confidence_ellipse_multiple_datasets_default(self):
        """Test that only train gets ellipse by default."""
        # Arrange
        scores = {
            "train": np.random.randn(50, 5),
            "test": np.random.randn(30, 5),
            "val": np.random.randn(20, 5),
        }

        # Act
        plot = ScoresPlot(scores, confidence_ellipse=True)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_confidence_ellipse_specific_datasets(self):
        """Test confidence ellipse for specific datasets."""
        # Arrange
        scores = {
            "train": np.random.randn(50, 5),
            "test": np.random.randn(30, 5),
            "val": np.random.randn(20, 5),
        }

        # Act
        plot = ScoresPlot(scores, confidence_ellipse=["train", "test"])
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_confidence_ellipse_with_continuous_coloring(self):
        """Test confidence ellipse with continuous coloring."""
        # Arrange
        scores = {"train": np.random.randn(50, 5)}
        concentrations = np.linspace(0.1, 1.0, 50)
        color_by_dict = {"train": concentrations}

        # Act
        plot = ScoresPlot(
            scores,
            color_by_dict=color_by_dict,
            confidence_ellipse=True,
            colormap="viridis",
        )
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestScoresPlotAnnotations:
    """Test point annotation functionality."""

    def test_annotations_single_dataset(self):
        """Test annotating points in a single dataset."""
        # Arrange
        scores = {"train": np.random.randn(10, 5)}
        annotations = {"train": [f"Sample {i}" for i in range(10)]}

        # Act
        plot = ScoresPlot(scores, annotations_dict=annotations)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_annotations_multiple_datasets(self):
        """Test annotating points in multiple datasets."""
        # Arrange
        scores = {
            "train": np.random.randn(10, 5),
            "test": np.random.randn(5, 5),
        }
        annotations = {
            "train": [f"Train {i}" for i in range(10)],
            "test": [f"Test {i}" for i in range(5)],
        }

        # Act
        plot = ScoresPlot(scores, annotations_dict=annotations)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_annotations_subset_of_datasets(self):
        """Test annotating only some datasets."""
        # Arrange
        scores = {
            "train": np.random.randn(10, 5),
            "test": np.random.randn(5, 5),
            "val": np.random.randn(5, 5),
        }
        # Only annotate train
        annotations = {"train": [f"S{i}" for i in range(10)]}

        # Act
        plot = ScoresPlot(scores, annotations_dict=annotations)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestScoresPlotBackwardsCompatibility:
    """Test backwards compatibility with deprecated parameters."""

    def test_labels_dict_deprecated_parameter(self):
        """Test that labels_dict parameter still works (backwards compatibility)."""
        # Arrange
        scores = {"train": np.random.randn(30, 5)}
        labels = np.array(["A"] * 10 + ["B"] * 10 + ["C"] * 10)
        labels_dict = {"train": labels}

        # Act
        plot = ScoresPlot(scores, labels_dict=labels_dict)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_color_by_dict_overrides_labels_dict(self):
        """Test that color_by_dict takes precedence over labels_dict."""
        # Arrange
        scores = {"train": np.random.randn(30, 5)}
        labels = np.array(["A"] * 10 + ["B"] * 10 + ["C"] * 10)
        colors = np.linspace(0, 1, 30)

        # Act
        plot = ScoresPlot(
            scores,
            labels_dict={"train": labels},
            color_by_dict={"train": colors},
        )
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        # Should have colorbar (continuous from color_by_dict)
        assert len(fig.axes) == 2
        plt.close(fig)


class TestScoresPlotCustomStyling:
    """Test custom styling and kwargs."""

    def test_custom_marker_size(self):
        """Test custom marker size via kwargs."""
        # Arrange
        scores = {"train": np.random.randn(50, 5)}
        plot = ScoresPlot(scores)

        # Act
        fig = plot.show(s=100)  # Large markers

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_alpha(self):
        """Test custom alpha (transparency) via kwargs."""
        # Arrange
        scores = {"train": np.random.randn(50, 5)}
        plot = ScoresPlot(scores)

        # Act
        fig = plot.show(alpha=0.5)

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_edge_colors(self):
        """Test custom edge colors and linewidths."""
        # Arrange
        scores = {"train": np.random.randn(50, 5)}
        plot = ScoresPlot(scores)

        # Act
        fig = plot.show(edgecolors="black", linewidths=0.5)

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_dataset_colors_parameter(self):
        """Test custom dataset colors."""
        # Arrange
        scores = {
            "train": np.random.randn(30, 5),
            "test": np.random.randn(20, 5),
        }
        custom_colors = {"train": "red", "test": "blue"}

        # Act
        plot = ScoresPlot(scores, dataset_colors=custom_colors)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestScoresPlotSubplots:
    """Test ScoresPlot with subplots."""

    def test_multiple_renders_on_subplots(self):
        """Test rendering multiple ScoresPlots on subplots."""
        # Arrange
        scores = {"train": np.random.randn(50, 5)}
        plot1 = ScoresPlot(scores, components=(0, 1))
        plot2 = ScoresPlot(scores, components=(1, 2))

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Act
        fig1, ax1 = plot1.render(ax=axes[0])
        fig2, ax2 = plot2.render(ax=axes[1])

        # Assert
        assert fig1 is fig
        assert fig2 is fig
        assert ax1 is axes[0]
        assert ax2 is axes[1]
        plt.close(fig)

    def test_render_without_show(self):
        """Test that render can be used without calling show."""
        # Arrange
        scores = {"train": np.random.randn(50, 5)}
        plot = ScoresPlot(scores)

        # Act
        fig, ax = plot.render()

        # Assert
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_comparison_subplots(self):
        """Test comparing different component pairs in subplots."""
        # Arrange
        scores = {"train": np.random.randn(50, 5)}

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        # Act
        ScoresPlot(scores, components=(0, 1)).render(ax=axes[0, 0])
        ScoresPlot(scores, components=(0, 2)).render(ax=axes[0, 1])
        ScoresPlot(scores, components=(1, 2)).render(ax=axes[1, 0])
        ScoresPlot(scores, components=(2, 3)).render(ax=axes[1, 1])

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)


@pytest.mark.parametrize("components", [(0, 1), (1, 2), (0, 3), (2, 4)])
def test_different_component_pairs(components):
    """Test plotting different component pairs."""
    # Arrange
    scores = {"train": np.random.randn(50, 5)}

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
    scores = {"train": np.random.randn(40, 5)}

    if is_categorical:
        color_by = np.array(["A"] * 10 + ["B"] * 10 + ["C"] * 10 + ["D"] * 10)
    else:
        color_by = np.linspace(0.1, 1.0, 40)

    color_by_dict = {"train": color_by}

    # Act
    plot = ScoresPlot(scores, color_by_dict=color_by_dict, colormap=colormap)
    fig = plot.show()

    # Assert
    assert isinstance(fig, Figure)
    plt.close(fig)


@pytest.mark.parametrize("n_samples", [10, 50, 100, 200])
def test_different_sample_sizes(n_samples):
    """Test plotting different numbers of samples."""
    # Arrange
    scores = {"train": np.random.randn(n_samples, 5)}

    # Act
    plot = ScoresPlot(scores)
    fig = plot.show()

    # Assert
    assert isinstance(fig, Figure)
    plt.close(fig)


class TestScoresPlotRenderAxisLimits:
    """Test render() method with axis limits."""

    def test_render_with_xlim(self):
        """Test render() with xlim parameter."""
        # Arrange
        scores = {"train": np.random.randn(50, 5)}
        plot = ScoresPlot(scores)

        # Act
        fig, ax = plot.render(xlim=(-2, 2))

        # Assert
        assert ax.get_xlim() == (-2, 2)
        plt.close(fig)

    def test_render_with_ylim(self):
        """Test render() with ylim parameter."""
        # Arrange
        scores = {"train": np.random.randn(50, 5)}
        plot = ScoresPlot(scores)

        # Act
        fig, ax = plot.render(ylim=(-3, 3))

        # Assert
        assert ax.get_ylim() == (-3, 3)
        plt.close(fig)

    def test_render_with_both_limits(self):
        """Test render() with both xlim and ylim."""
        # Arrange
        scores = {"train": np.random.randn(50, 5)}
        plot = ScoresPlot(scores)

        # Act
        fig, ax = plot.render(xlim=(-2, 2), ylim=(-3, 3))

        # Assert
        assert ax.get_xlim() == (-2, 2)
        assert ax.get_ylim() == (-3, 3)
        plt.close(fig)

    def test_render_axes_without_figure_raises_error(self):
        """Test that render() raises error if axes has no figure."""
        # Arrange
        scores = {"train": np.random.randn(50, 5)}
        plot = ScoresPlot(scores)

        # Create a mock axes object with get_figure() returning None
        from unittest.mock import Mock

        ax = Mock(spec=Axes)
        ax.get_figure.return_value = None

        # Act & Assert
        with pytest.raises(ValueError, match="Axes object has no associated figure"):
            plot.render(ax=ax)


class TestScoresPlotConfidenceEllipseEdgeCases:
    """Test confidence ellipse with edge cases."""

    def test_ellipse_with_continuous_coloring(self):
        """Test confidence ellipse color with continuous coloring."""
        # Arrange
        scores = {"train": np.random.randn(50, 5)}
        color_by = np.linspace(0, 1, 50)

        plot = ScoresPlot(
            scores,
            color_by_dict={"train": color_by},
            colormap="viridis",
            confidence_ellipse=True,
        )

        # Act
        fig = plot.show()

        # Assert - should use gray color for ellipse with continuous coloring
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_ellipse_with_categorical_coloring(self):
        """Test confidence ellipse color with categorical coloring."""
        # Arrange
        scores = {"train": np.random.randn(50, 5)}
        classes = np.array(["A"] * 25 + ["B"] * 25)

        plot = ScoresPlot(
            scores, color_by_dict={"train": classes}, confidence_ellipse=True
        )

        # Act
        fig = plot.show()

        # Assert - should use dataset color for ellipse with categorical
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_ellipse_without_color_by(self):
        """Test confidence ellipse without color_by_dict."""
        # Arrange
        scores = {"train": np.random.randn(50, 5)}

        plot = ScoresPlot(scores, confidence_ellipse=True)

        # Act
        fig = plot.show()

        # Assert - should use dataset color
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_multiple_datasets_with_selective_ellipses(self):
        """Test ellipses for specific datasets only."""
        # Arrange
        scores = {
            "train": np.random.randn(50, 5),
            "test": np.random.randn(30, 5),
            "val": np.random.randn(20, 5),
        }

        # Only train and test get ellipses
        plot = ScoresPlot(
            scores,
            confidence_ellipse=["train", "test"],  # val should not get ellipse
        )

        # Act
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

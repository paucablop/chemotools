"""Tests for SpectraPlot class."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from chemotools.plotting import SpectraPlot, is_displayable


class TestSpectraPlotBasics:
    """Test basic functionality of SpectraPlot."""

    def test_implements_display_protocol(self):
        """Test that SpectraPlot implements Display protocol."""
        # Arrange
        x = np.linspace(400, 2500, 100)
        y = np.random.randn(5, 100)

        # Act
        plot = SpectraPlot(x, y)

        # Assert
        assert is_displayable(plot)

    def test_single_spectrum_1d(self):
        """Test plotting a single spectrum (1D y data)."""
        # Arrange
        x = np.linspace(400, 2500, 100)
        y = np.random.randn(100)

        # Act
        plot = SpectraPlot(x, y)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_multiple_spectra_2d(self):
        """Test plotting multiple spectra (2D y data)."""
        # Arrange
        x = np.linspace(400, 2500, 100)
        y = np.random.randn(5, 100)

        # Act
        plot = SpectraPlot(x, y)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_axis_labels(self):
        """Test custom x and y axis labels."""
        # Arrange
        x = np.linspace(4000, 400, 100)
        y = np.random.randn(3, 100)

        # Act
        plot = SpectraPlot(x, y)
        fig = plot.show(xlabel="Wavenumber (cm⁻¹)", ylabel="Intensity (a.u.)")

        # Assert
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Wavenumber (cm⁻¹)"
        assert ax.get_ylabel() == "Intensity (a.u.)"
        plt.close(fig)

    def test_default_axis_labels(self):
        """Test default axis labels."""
        # Arrange
        x = np.linspace(400, 2500, 100)
        y = np.random.randn(3, 100)

        # Act
        plot = SpectraPlot(x, y)
        fig = plot.show()

        # Assert
        ax = fig.axes[0]
        assert ax.get_xlabel() == "X-axis"
        assert ax.get_ylabel() == "Y-axis"
        plt.close(fig)

    def test_show_with_title(self):
        """Test show() with custom title."""
        # Arrange
        x = np.linspace(400, 2500, 100)
        y = np.random.randn(3, 100)
        plot = SpectraPlot(x, y)

        # Act
        fig = plot.show(title="Test Spectra")

        # Assert
        ax = fig.axes[0]
        assert ax.get_title() == "Test Spectra"
        plt.close(fig)

    def test_show_with_figsize(self):
        """Test show() with custom figure size."""
        # Arrange
        x = np.linspace(400, 2500, 100)
        y = np.random.randn(3, 100)
        plot = SpectraPlot(x, y)

        # Act
        fig = plot.show(figsize=(12, 6))

        # Assert
        assert fig.get_size_inches()[0] == 12
        assert fig.get_size_inches()[1] == 6
        plt.close(fig)

    def test_render_returns_figure_and_axes(self):
        """Test that render() returns (Figure, Axes) tuple."""
        # Arrange
        x = np.linspace(400, 2500, 100)
        y = np.random.randn(3, 100)
        plot = SpectraPlot(x, y)

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
        x = np.linspace(400, 2500, 100)
        y = np.random.randn(3, 100)
        plot = SpectraPlot(x, y)
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
        x = np.linspace(400, 2500, 100)
        y = np.random.randn(3, 100)
        plot = SpectraPlot(x, y)

        # Act
        fig = plot.show(xlim=(1000, 2000), ylim=(-1, 1))

        # Assert
        ax = fig.axes[0]
        assert ax.get_xlim() == (1000, 2000)
        assert ax.get_ylim() == (-1, 1)
        plt.close(fig)


class TestSpectraPlotCategoricalColoring:
    """Test categorical coloring functionality."""

    def test_categorical_with_strings(self):
        """Test categorical coloring with string labels."""
        # Arrange
        x = np.linspace(400, 2500, 100)
        y = np.random.randn(6, 100)
        classes = np.array(["A", "A", "B", "B", "C", "C"])

        # Act
        plot = SpectraPlot(x, y, color_by=classes)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        assert ax.get_legend() is not None
        plt.close(fig)

    def test_categorical_with_integers(self):
        """Test categorical coloring with integer labels (≤10 unique)."""
        # Arrange
        x = np.linspace(400, 2500, 100)
        y = np.random.randn(8, 100)
        classes = np.array([1, 1, 2, 2, 3, 3, 4, 4])

        # Act
        plot = SpectraPlot(x, y, color_by=classes)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        assert ax.get_legend() is not None
        plt.close(fig)

    def test_categorical_with_repeating_floats(self):
        """Test categorical detection with repeating float values."""
        # Arrange
        x = np.linspace(400, 2500, 100)
        y = np.random.randn(6, 100)
        # Repeating values should be detected as categorical
        groups = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])

        # Act
        plot = SpectraPlot(x, y, color_by=groups)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        assert ax.get_legend() is not None
        plt.close(fig)

    def test_categorical_custom_colormap(self):
        """Test categorical coloring with custom colormap."""
        # Arrange
        x = np.linspace(400, 2500, 100)
        y = np.random.randn(6, 100)
        classes = np.array(
            ["Class A", "Class A", "Class B", "Class B", "Class C", "Class C"]
        )

        # Act
        plot = SpectraPlot(x, y, color_by=classes, colormap="Set2")
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_force_categorical_with_parameter(self):
        """Test forcing categorical treatment with color_mode='categorical'."""
        # Arrange
        x = np.linspace(400, 2500, 100)
        y = np.random.randn(4, 100)
        # 4 unique non-repeating values might not be detected as categorical
        values = np.array([1.0, 2.0, 3.0, 4.0])

        # Act
        plot = SpectraPlot(x, y, color_by=values, color_mode="categorical")
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        # Should have legend for categorical
        assert ax.get_legend() is not None
        plt.close(fig)


class TestSpectraPlotContinuousColoring:
    """Test continuous coloring functionality."""

    def test_continuous_with_floats(self):
        """Test continuous coloring with float values."""
        # Arrange
        x = np.linspace(400, 2500, 100)
        y = np.random.randn(10, 100)
        concentrations = np.linspace(0.1, 1.0, 10)

        # Act
        plot = SpectraPlot(x, y, color_by=concentrations, colormap="viridis")
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        # Should have colorbar for continuous
        assert len(fig.axes) == 2  # Main axis + colorbar axis
        plt.close(fig)

    def test_continuous_custom_colorbar_label(self):
        """Test continuous coloring with custom colorbar label."""
        # Arrange
        x = np.linspace(400, 2500, 100)
        y = np.random.randn(8, 100)
        concentrations = np.linspace(0.5, 5.0, 8)

        # Act
        plot = SpectraPlot(
            x,
            y,
            color_by=concentrations,
            colormap="plasma",
            colorbar_label="Concentration (mg/L)",
        )
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2
        # Check colorbar has the custom label
        cbar_ax = fig.axes[1]
        assert cbar_ax.get_ylabel() == "Concentration (mg/L)"
        plt.close(fig)

    def test_continuous_with_different_colormaps(self):
        """Test continuous coloring with different colormaps."""
        # Arrange
        x = np.linspace(400, 2500, 100)
        y = np.random.randn(10, 100)
        values = np.linspace(0, 1, 10)

        for cmap in ["viridis", "plasma", "cividis", "coolwarm"]:
            # Act
            plot = SpectraPlot(x, y, color_by=values, colormap=cmap)
            fig = plot.show()

            # Assert
            assert isinstance(fig, Figure)
            plt.close(fig)

    def test_force_continuous_with_parameter(self):
        """Test forcing continuous treatment with color_mode='continuous'."""
        # Arrange
        x = np.linspace(400, 2500, 100)
        y = np.random.randn(5, 100)
        # 5 repeating integers might be detected as categorical
        levels = np.array([1, 2, 3, 4, 5])

        # Act
        plot = SpectraPlot(
            x, y, color_by=levels, color_mode="continuous", colormap="viridis"
        )
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        # Should have colorbar for continuous
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_four_unique_floats_continuous(self):
        """Test edge case: 4 unique non-repeating floats should be continuous."""
        # Arrange
        x = np.linspace(400, 2500, 100)
        y = np.random.randn(4, 100)
        # 4 unique floats without repeats should be continuous
        levels = np.array([1.0, 2.0, 3.0, 4.0])

        # Act
        plot = SpectraPlot(x, y, color_by=levels, colormap="viridis")
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        # Should have colorbar for continuous
        assert len(fig.axes) == 2
        plt.close(fig)


class TestSpectraPlotNoColoring:
    """Test default behavior without color_by."""

    def test_no_color_by_parameter(self):
        """Test plotting without color_by uses default colors."""
        # Arrange
        x = np.linspace(400, 2500, 100)
        y = np.random.randn(5, 100)

        # Act
        plot = SpectraPlot(x, y)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        # No colorbar should be present
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_with_labels_parameter(self):
        """Test with labels parameter for legend."""
        # Arrange
        x = np.linspace(400, 2500, 100)
        y = np.random.randn(3, 100)
        labels = ["Sample 1", "Sample 2", "Sample 3"]

        # Act
        plot = SpectraPlot(x, y, labels=labels)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None
        plt.close(fig)


class TestSpectraPlotEdgeCases:
    """Test edge cases and error handling."""

    def test_mismatched_x_y_dimensions(self):
        """Test that mismatched x and y dimensions work correctly."""
        # Arrange
        x = np.linspace(400, 2500, 100)
        y = np.random.randn(5, 100)  # 5 spectra, 100 points each

        # Act
        plot = SpectraPlot(x, y)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_single_spectrum_with_color_by(self):
        """Test single spectrum with color_by parameter."""
        # Arrange
        x = np.linspace(400, 2500, 100)
        y = np.random.randn(100)  # Single spectrum
        # color_by should be scalar for single spectrum
        color_value = np.array([0.5])

        # Act
        plot = SpectraPlot(x, y, color_by=color_value)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_kwargs_passed_to_plot(self):
        """Test that additional kwargs are passed to plot function."""
        # Arrange
        x = np.linspace(400, 2500, 100)
        y = np.random.randn(3, 100)

        # Act
        plot = SpectraPlot(x, y)
        fig = plot.show(alpha=0.5, linewidth=2)

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestSpectraPlotSubplots:
    """Test SpectraPlot with subplots."""

    def test_multiple_renders_on_subplots(self):
        """Test rendering multiple SpectraPlots on subplots."""
        # Arrange
        x1 = np.linspace(400, 2500, 100)
        y1 = np.random.randn(3, 100)
        plot1 = SpectraPlot(x1, y1)

        x2 = np.linspace(1000, 2000, 100)
        y2 = np.random.randn(3, 100)
        plot2 = SpectraPlot(x2, y2)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

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
        x = np.linspace(400, 2500, 100)
        y = np.random.randn(5, 100)
        plot = SpectraPlot(x, y)

        # Act
        fig, ax = plot.render()

        # Assert
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
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
    x = np.linspace(400, 2500, 100)
    y = np.random.randn(8, 100)

    if is_categorical:
        color_by = np.array(["A", "A", "B", "B", "C", "C", "D", "D"])
    else:
        color_by = np.linspace(0.1, 1.0, 8)

    # Act
    plot = SpectraPlot(x, y, color_by=color_by, colormap=colormap)
    fig = plot.show()

    # Assert
    assert isinstance(fig, Figure)
    plt.close(fig)


@pytest.mark.parametrize("n_spectra", [1, 3, 10, 50])
def test_different_numbers_of_spectra(n_spectra):
    """Test plotting different numbers of spectra."""
    # Arrange
    x = np.linspace(400, 2500, 100)
    if n_spectra == 1:
        y = np.random.randn(100)
    else:
        y = np.random.randn(n_spectra, 100)

    # Act
    plot = SpectraPlot(x, y)
    fig = plot.show()

    # Assert
    assert isinstance(fig, Figure)
    plt.close(fig)


class TestSpectraPlotAxisLimits:
    """Test axis limits and auto-scaling functionality."""

    def test_xlim_with_auto_yscaling(self):
        """Test xlim parameter with automatic y-axis scaling."""
        # Arrange
        x = np.linspace(400, 2500, 210)
        y = np.random.randn(3, 210)
        plot = SpectraPlot(x, y)

        # Act - xlim without ylim should auto-scale y-axis
        fig = plot.show(xlim=(1000, 1500))

        # Assert
        ax = fig.axes[0]
        assert ax.get_xlim() == (1000, 1500)
        # Y-axis should be auto-scaled to the data in x-range
        ylim = ax.get_ylim()
        assert ylim[0] < ylim[1]  # Valid range
        plt.close(fig)

    def test_xlim_with_explicit_ylim(self):
        """Test xlim with explicit ylim (no auto-scaling)."""
        # Arrange
        x = np.linspace(400, 2500, 100)
        y = np.random.randn(3, 100)
        plot = SpectraPlot(x, y)

        # Act - both xlim and ylim specified
        fig = plot.show(xlim=(1000, 1500), ylim=(-2, 2))

        # Assert
        ax = fig.axes[0]
        assert ax.get_xlim() == (1000, 1500)
        assert ax.get_ylim() == (-2, 2)
        plt.close(fig)

    def test_render_xlim_with_auto_yscaling(self):
        """Test render() with xlim parameter and automatic y-axis scaling."""
        # Arrange
        x = np.linspace(400, 2500, 210)
        y = np.random.randn(5, 210)
        plot = SpectraPlot(x, y)

        # Act
        fig, ax = plot.render(xlim=(800, 1200))

        # Assert
        assert ax.get_xlim() == (800, 1200)
        # Y-axis should be auto-scaled
        ylim = ax.get_ylim()
        assert ylim[0] < ylim[1]
        plt.close(fig)

    def test_render_axes_without_figure_raises_error(self):
        """Test that render() raises error if axes has no figure."""
        # Arrange
        x = np.linspace(400, 2500, 100)
        y = np.random.randn(3, 100)
        plot = SpectraPlot(x, y)

        # Create a mock axes object with get_figure() returning None
        from unittest.mock import Mock

        ax = Mock(spec=Axes)
        ax.get_figure.return_value = None

        # Act & Assert
        with pytest.raises(ValueError, match="Axes object has no associated figure"):
            plot.render(ax=ax)

    def test_calculate_ylim_no_data_in_range(self):
        """Test _calculate_ylim_for_xlim when no data in x-range."""
        # Arrange
        x = np.linspace(400, 2500, 100)
        y = np.random.randn(3, 100)
        plot = SpectraPlot(x, y)

        # Act - xlim outside data range
        fig = plot.show(xlim=(3000, 4000))

        # Assert
        ax = fig.axes[0]
        # Should return default limits (0, 1) when no data in range
        ylim = ax.get_ylim()
        assert ylim == (0, 1)
        plt.close(fig)

    def test_calculate_ylim_all_same_values(self):
        """Test _calculate_ylim_for_xlim when all y-values are the same."""
        # Arrange
        x = np.linspace(400, 2500, 100)
        # Create spectra with constant values
        y = np.ones((3, 100)) * 5.0
        plot = SpectraPlot(x, y)

        # Act
        fig = plot.show(xlim=(1000, 1500))

        # Assert
        ax = fig.axes[0]
        ylim = ax.get_ylim()
        # Should add small margin around the constant value
        assert ylim[0] < 5.0  # Should be below 5
        assert ylim[1] > 5.0  # Should be above 5
        assert abs(ylim[1] - ylim[0]) > 0  # Non-zero range
        plt.close(fig)

    def test_calculate_ylim_with_custom_margin(self):
        """Test _calculate_ylim_for_xlim includes margin."""
        # Arrange
        x = np.linspace(400, 2500, 100)
        y = np.random.randn(3, 100)
        plot = SpectraPlot(x, y)

        # Act
        fig = plot.show(xlim=(1000, 1500))

        # Assert
        ax = fig.axes[0]
        ylim = ax.get_ylim()

        # Get data in the x-range to verify margin was added
        mask = (x >= 1000) & (x <= 1500)
        y_in_range = y[:, mask]
        data_min = np.min(y_in_range)
        data_max = np.max(y_in_range)

        # Y-limits should be wider than the data range (due to margin)
        assert ylim[0] < data_min
        assert ylim[1] > data_max
        plt.close(fig)

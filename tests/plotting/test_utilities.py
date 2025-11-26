"""Tests for plotting utilities."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from chemotools.plotting._utils import (
    setup_figure,
    get_colors_from_labels,
    add_confidence_ellipse,
    annotate_points,
    detect_categorical,
    get_default_colormap,
    add_colorbar,
    scatter_with_colormap,
)


class TestSetupFigure:
    """Test setup_figure utility function."""

    def test_basic_figure_creation(self):
        """Test basic figure creation."""
        # Act
        fig, ax = setup_figure()

        # Assert
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_with_title(self):
        """Test figure with title."""
        # Act
        fig, ax = setup_figure(title="Test Title")

        # Assert
        assert ax.get_title() == "Test Title"
        plt.close(fig)

    def test_with_labels(self):
        """Test figure with axis labels."""
        # Act
        fig, ax = setup_figure(xlabel="X Label", ylabel="Y Label")

        # Assert
        assert ax.get_xlabel() == "X Label"
        assert ax.get_ylabel() == "Y Label"
        plt.close(fig)

    def test_custom_figsize(self):
        """Test custom figure size."""
        # Act
        fig, ax = setup_figure(figsize=(12, 8))

        # Assert
        assert fig.get_size_inches()[0] == 12
        assert fig.get_size_inches()[1] == 8
        plt.close(fig)


class TestGetColorsFromLabels:
    """Test get_colors_from_labels utility function."""

    def test_string_labels(self):
        """Test with string labels."""
        # Arrange
        labels = np.array(["A", "B", "A", "C", "B"])

        # Act
        colors = get_colors_from_labels(labels)

        # Assert
        assert colors.shape == (5, 4)  # RGBA colors
        # Same labels should have same colors
        assert np.array_equal(colors[0], colors[2])  # Both "A"
        assert np.array_equal(colors[1], colors[4])  # Both "B"

    def test_integer_labels(self):
        """Test with integer labels."""
        # Arrange
        labels = np.array([1, 2, 1, 3, 2])

        # Act
        colors = get_colors_from_labels(labels)

        # Assert
        assert colors.shape == (5, 4)
        assert np.array_equal(colors[0], colors[2])  # Both 1
        assert np.array_equal(colors[1], colors[4])  # Both 2

    def test_custom_colormap(self):
        """Test with custom colormap."""
        # Arrange
        labels = np.array(["A", "B", "C"])

        # Act
        colors = get_colors_from_labels(labels, colormap="Set2")

        # Assert
        assert colors.shape == (3, 4)


class TestAddConfidenceEllipse:
    """Test add_confidence_ellipse utility function."""

    def test_basic_ellipse(self):
        """Test adding basic confidence ellipse."""
        # Arrange
        fig, ax = plt.subplots()
        x = np.random.randn(100)
        y = np.random.randn(100)

        # Act - should not raise any errors
        add_confidence_ellipse(ax, x, y)

        # Assert
        plt.close(fig)

    def test_custom_confidence_level(self):
        """Test with custom confidence level."""
        # Arrange
        fig, ax = plt.subplots()
        x = np.random.randn(100)
        y = np.random.randn(100)

        # Act
        add_confidence_ellipse(ax, x, y, confidence=0.90)

        # Assert
        plt.close(fig)

    def test_n_std_parameter(self):
        """Test with n_std parameter instead of confidence."""
        # Arrange
        fig, ax = plt.subplots()
        x = np.random.randn(100)
        y = np.random.randn(100)

        # Act - use n_std parameter (should override confidence)
        add_confidence_ellipse(ax, x, y, n_std=2)

        # Assert
        plt.close(fig)

    def test_custom_styling(self):
        """Test with custom styling parameters."""
        # Arrange
        fig, ax = plt.subplots()
        x = np.random.randn(100)
        y = np.random.randn(100)

        # Act
        add_confidence_ellipse(
            ax, x, y, edgecolor="red", linewidth=2, linestyle="--", alpha=0.5
        )

        # Assert
        plt.close(fig)

    def test_mismatched_lengths_raises_error(self):
        """Test that mismatched x and y lengths raise error."""
        # Arrange
        fig, ax = plt.subplots()
        x = np.random.randn(100)
        y = np.random.randn(50)  # Different length

        # Act & Assert
        with pytest.raises(ValueError, match="x and y must have the same length"):
            add_confidence_ellipse(ax, x, y)
        plt.close(fig)

    def test_too_few_points_raises_error(self):
        """Test that too few points raise error."""
        # Arrange
        fig, ax = plt.subplots()
        x = np.array([1, 2])  # Only 2 points
        y = np.array([1, 2])

        # Act & Assert
        with pytest.raises(ValueError, match="Need at least 3 points"):
            add_confidence_ellipse(ax, x, y)
        plt.close(fig)


class TestAnnotatePoints:
    """Test annotate_points utility function."""

    def test_basic_annotation(self):
        """Test basic point annotation."""
        # Arrange
        fig, ax = plt.subplots()
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3])
        labels = ["A", "B", "C"]

        # Act
        annotate_points(ax, x, y, labels)

        # Assert
        plt.close(fig)

    def test_with_custom_kwargs(self):
        """Test annotation with custom kwargs."""
        # Arrange
        fig, ax = plt.subplots()
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3])
        labels = ["A", "B", "C"]

        # Act
        annotate_points(
            ax,
            x,
            y,
            labels,
            fontsize=10,
            color="red",
            xytext=(5, 5),
            textcoords="offset points",
        )

        # Assert
        plt.close(fig)


class TestDetectCategorical:
    """Test detect_categorical utility function."""

    def test_string_is_categorical(self):
        """Test that string arrays are detected as categorical."""
        # Arrange
        color_by = np.array(["A", "B", "C"])

        # Act
        result = detect_categorical(color_by)

        # Assert
        assert result is True

    def test_boolean_is_categorical(self):
        """Test that boolean arrays are detected as categorical."""
        # Arrange
        color_by = np.array([True, False, True, False])

        # Act
        result = detect_categorical(color_by)

        # Assert
        assert result is True

    def test_small_integer_is_categorical(self):
        """Test that integers with ≤10 unique values are categorical."""
        # Arrange
        color_by = np.array([1, 2, 3, 1, 2, 3])

        # Act
        result = detect_categorical(color_by)

        # Assert
        assert result is True

    def test_many_integer_is_continuous(self):
        """Test that integers with >10 unique values are continuous."""
        # Arrange
        color_by = np.arange(15)

        # Act
        result = detect_categorical(color_by)

        # Assert
        assert result is False

    def test_repeating_floats_is_categorical(self):
        """Test that repeating float values are categorical."""
        # Arrange
        color_by = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])

        # Act
        result = detect_categorical(color_by)

        # Assert
        assert result is True

    def test_unique_floats_is_continuous(self):
        """Test that unique float values are continuous."""
        # Arrange
        color_by = np.array([1.0, 2.0, 3.0, 4.0])

        # Act
        result = detect_categorical(color_by)

        # Assert
        assert result is False

    def test_many_unique_floats_is_continuous(self):
        """Test that many unique floats are continuous."""
        # Arrange
        color_by = np.linspace(0, 1, 20)

        # Act
        result = detect_categorical(color_by)

        # Assert
        assert result is False


class TestGetDefaultColormap:
    """Test get_default_colormap utility function."""

    def test_categorical_default(self):
        """Test default colormap for categorical data."""
        # Act
        cmap = get_default_colormap(is_categorical=True)

        # Assert
        assert cmap == "tab10"

    def test_continuous_default(self):
        """Test default colormap for continuous data."""
        # Act
        cmap = get_default_colormap(is_categorical=False)

        # Assert
        assert cmap == "shap"

    def test_custom_colormap_override(self):
        """Test that custom colormap overrides defaults."""
        # Act
        cmap = get_default_colormap(is_categorical=True, colormap="Set2")

        # Assert
        assert cmap == "Set2"

        # Act
        cmap = get_default_colormap(is_categorical=False, colormap="plasma")

        # Assert
        assert cmap == "plasma"


class TestAddColorbar:
    """Test add_colorbar utility function."""

    def test_basic_colorbar(self):
        """Test adding basic colorbar."""
        # Arrange
        fig, ax = plt.subplots()
        color_by = np.linspace(0, 1, 10)

        # Act
        add_colorbar(ax, color_by, "viridis")

        # Assert - check that colorbar was added (figure should have 2 axes now)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_custom_label(self):
        """Test colorbar with custom label."""
        # Arrange
        fig, ax = plt.subplots()
        color_by = np.linspace(0, 1, 10)

        # Act
        add_colorbar(ax, color_by, "plasma", label="Concentration (mg/L)")

        # Assert - check colorbar label
        cbar_ax = fig.axes[1]
        assert cbar_ax.get_ylabel() == "Concentration (mg/L)"
        plt.close(fig)

    def test_different_colormaps(self):
        """Test with different colormaps."""
        # Arrange
        color_by = np.linspace(0, 1, 10)

        # Act & Assert
        for cmap in ["viridis", "plasma", "cividis", "coolwarm"]:
            fig, ax = plt.subplots()
            add_colorbar(ax, color_by, cmap)
            assert len(fig.axes) == 2
            plt.close(fig)


class TestCalculateYlimForXlim:
    """Tests for calculate_ylim_for_xlim utility function."""

    def test_with_2d_data_x_matches_axis_0(self):
        """Test with 2D data where x matches axis 0 (rows)."""
        from chemotools.plotting._utils import calculate_ylim_for_xlim

        x = np.linspace(0, 10, 100)
        y = np.random.rand(100, 5)  # x matches rows
        ylim = calculate_ylim_for_xlim(x, y, xlim=(3, 7))
        assert isinstance(ylim, tuple)
        assert len(ylim) == 2
        assert ylim[0] < ylim[1]

    def test_raises_on_mismatched_dimensions(self):
        """Test that it raises error when x doesn't match y dimensions."""
        from chemotools.plotting._utils import calculate_ylim_for_xlim

        x = np.linspace(0, 10, 100)
        y = np.random.rand(50, 60)  # Neither dimension matches x
        with pytest.raises(ValueError, match="x length .* must match either dimension"):
            calculate_ylim_for_xlim(x, y, xlim=(3, 7))


class TestScatterWithColormap:
    """Test scatter_with_colormap utility function."""

    def test_simple_scatter(self):
        """Test simple scatter plot without coloring."""
        fig, ax = plt.subplots()
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3])

        scatter_with_colormap(ax, x, y, color="blue", label="Test")

        # Check if scatter was called
        assert len(ax.collections) == 1
        # Check label
        assert ax.collections[0].get_label() == "Test"
        plt.close(fig)

    def test_categorical_coloring(self):
        """Test scatter plot with categorical coloring."""
        fig, ax = plt.subplots()
        x = np.array([1, 2, 3, 4])
        y = np.array([1, 2, 3, 4])
        color_by = np.array(["A", "B", "A", "B"])

        scatter_with_colormap(
            ax,
            x,
            y,
            color_by=color_by,
            is_categorical=True,
            colormap="tab10",
            label="Data",
        )

        # Should have 2 collections (one for A, one for B)
        assert len(ax.collections) == 2
        # Check labels
        labels = [c.get_label() for c in ax.collections]
        assert "Data - A" in labels
        assert "Data - B" in labels
        plt.close(fig)

    def test_continuous_coloring(self):
        """Test scatter plot with continuous coloring."""
        fig, ax = plt.subplots()
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3])
        color_by = np.array([0.1, 0.5, 0.9])

        scatter_with_colormap(
            ax,
            x,
            y,
            color_by=color_by,
            is_categorical=False,
            colormap="viridis",
            label="Data",
        )

        # Should have 1 collection
        assert len(ax.collections) == 1
        # Check label
        assert ax.collections[0].get_label() == "Data"
        # Check if array is set for colormapping
        assert ax.collections[0].get_array() is not None
        plt.close(fig)

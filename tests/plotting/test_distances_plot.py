"""Tests for DistancesPlot class - simplified single-dataset API."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from chemotools.plotting import DistancesPlot


class TestDistancesPlotBasics:
    """Test basic functionality of DistancesPlot."""

    def test_distances_plot_basic_1d(self):
        """Test basic initialization with 1D distances."""
        # Arrange
        distances = np.random.rand(100)

        # Act
        plot = DistancesPlot(distances)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_distances_plot_basic_2d(self):
        """Test basic initialization with 2D distances."""
        # Arrange
        distances = np.random.rand(100, 2)

        # Act
        plot = DistancesPlot(distances)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_multiple_datasets_composed(self):
        """Test plotting multiple datasets using composition."""
        # Arrange
        train_distances = np.random.rand(100, 2)
        test_distances = np.random.rand(50, 2)

        # Act - compose multiple plots
        fig, ax = plt.subplots()
        DistancesPlot(
            train_distances, label="Train", distances_selection=(0, 1)
        ).render(ax)
        DistancesPlot(test_distances, label="Test", distances_selection=(0, 1)).render(
            ax
        )
        ax.legend()

        # Assert
        assert isinstance(fig, Figure)
        legend = ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) == 2  # Train, Test
        plt.close(fig)

    def test_custom_labels(self):
        """Test custom axis labels."""
        # Arrange
        distances = np.random.rand(100)

        # Act
        plot = DistancesPlot(distances)
        fig = plot.show(xlabel="Sample Number", ylabel="Q Residuals")

        # Assert
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Sample Number"
        assert ax.get_ylabel() == "Q Residuals"
        plt.close(fig)

    def test_with_color(self):
        """Test with custom color."""
        # Arrange
        distances = np.random.rand(100)

        # Act
        plot = DistancesPlot(distances, color="red", label="Red Data")
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestDistancesPlotAxesSelection:
    """Test distances axis selection."""

    def test_default_selection_1d(self):
        """Test default is (None, 0) for 1D - sample index vs distance."""
        # Arrange
        distances = np.random.rand(100)

        # Act
        plot = DistancesPlot(distances)

        # Assert
        assert plot.distances_selection == (None, 0)

    def test_default_selection_2d(self):
        """Test default is (0, 1) for 2D - distance 0 vs distance 1."""
        # Arrange
        distances = np.random.rand(100, 3)

        # Act
        plot = DistancesPlot(distances)

        # Assert
        assert plot.distances_selection == (0, 1)

    def test_custom_selection_tuple(self):
        """Test custom distances tuple selection."""
        # Arrange
        distances = np.random.rand(100, 3)

        # Act
        plot = DistancesPlot(distances, distances_selection=(0, 2))

        # Assert
        assert plot.distances_selection == (0, 2)

    def test_selection_as_single_int(self):
        """Test distances parameter as single int (plots against sample index)."""
        # Arrange
        distances = np.random.rand(100, 2)

        # Act
        plot = DistancesPlot(distances, distances_selection=1)

        # Assert
        assert plot.distances_selection == (None, 1)

    def test_sample_index_selection(self):
        """Test explicit sample index selection with None."""
        # Arrange
        distances = np.random.rand(100, 2)

        # Act
        plot = DistancesPlot(distances, distances_selection=(None, 1))

        # Assert
        assert plot.distances_selection == (None, 1)


class TestDistancesPlotValidation:
    """Test input validation."""

    def test_invalid_dimensions_raises_error(self):
        """Test that 3D distances raise ValueError."""
        # Arrange
        distances = np.random.rand(100, 3, 4)

        # Act & Assert
        with pytest.raises(ValueError, match="must be 1D or 2D array"):
            DistancesPlot(distances)

    def test_invalid_x_axis_raises_error(self):
        """Test that invalid x_axis index raises ValueError."""
        # Arrange
        distances = np.random.rand(100, 2)

        # Act & Assert
        with pytest.raises(ValueError, match="x_axis index.*invalid"):
            DistancesPlot(distances, distances_selection=(5, 1))

    def test_invalid_y_axis_raises_error(self):
        """Test that invalid y_axis index raises ValueError."""
        # Arrange
        distances = np.random.rand(100, 2)

        # Act & Assert
        with pytest.raises(ValueError, match="y_axis index.*invalid"):
            DistancesPlot(distances, distances_selection=(0, 5))

    def test_same_axis_raises_error(self):
        """Test that same indices raises ValueError."""
        # Arrange
        distances = np.random.rand(100, 3)

        # Act & Assert
        with pytest.raises(ValueError, match="cannot be the same"):
            DistancesPlot(distances, distances_selection=(1, 1))


class TestDistancesPlotConfidenceLines:
    """Test confidence/threshold lines."""

    def test_y_threshold_only(self):
        """Test horizontal threshold line."""
        # Arrange
        distances = np.random.rand(100)

        # Act
        plot = DistancesPlot(distances, confidence_lines=(None, 0.8))
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        # Check that a horizontal line was drawn
        assert len(ax.get_lines()) > 0
        plt.close(fig)

    def test_both_thresholds(self):
        """Test both horizontal and vertical threshold lines."""
        # Arrange
        distances = np.random.rand(100, 2)

        # Act
        plot = DistancesPlot(
            distances, distances_selection=(0, 1), confidence_lines=(0.5, 0.8)
        )
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        # Should have both vertical and horizontal lines
        assert len(ax.get_lines()) >= 2
        plt.close(fig)

    def test_x_threshold_only(self):
        """Test vertical threshold line only."""
        # Arrange
        distances = np.random.rand(100, 2)

        # Act
        plot = DistancesPlot(
            distances, distances_selection=(0, 1), confidence_lines=(0.5, None)
        )
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestDistancesPlotColoring:
    """Test coloring functionality."""

    def test_categorical_coloring(self):
        """Test categorical coloring by class."""
        # Arrange
        distances = np.random.rand(100)
        classes = np.array(["A", "B", "C"] * 33 + ["A"])

        # Act
        plot = DistancesPlot(distances, color_by=classes, colormap="tab10")
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_continuous_coloring(self):
        """Test continuous coloring by numeric values."""
        # Arrange
        distances = np.random.rand(100)
        values = np.random.rand(100)

        # Act
        plot = DistancesPlot(distances, color_by=values, colormap="viridis")
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        # Should have a colorbar
        assert len(fig.axes) > 1
        plt.close(fig)


class TestDistancesPlotAnnotations:
    """Test annotations."""

    def test_with_annotations(self):
        """Test point annotations."""
        # Arrange
        distances = np.random.rand(20)
        annotations = [f"S{i}" for i in range(20)]

        # Act
        plot = DistancesPlot(distances, annotations=annotations)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_partial_annotations(self):
        """Test with selective annotations (some empty strings)."""
        # Arrange
        distances = np.random.rand(20)
        annotations = ["" if i % 5 != 0 else f"S{i}" for i in range(20)]

        # Act
        plot = DistancesPlot(distances, annotations=annotations)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestDistancesPlotRender:
    """Test render functionality."""

    def test_render_without_axes_creates_new(self):
        """Test that render creates new figure when ax is None."""
        # Arrange
        distances = np.random.rand(100)
        plot = DistancesPlot(distances)

        # Act
        fig, ax = plot.render()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_render_with_existing_axes(self):
        """Test render on existing axes."""
        # Arrange
        distances = np.random.rand(100)
        plot = DistancesPlot(distances)

        # Act
        fig, ax = plt.subplots()
        returned_fig, returned_ax = plot.render(ax)

        # Assert
        assert returned_fig is fig
        assert returned_ax is ax
        plt.close(fig)

    def test_render_with_limits(self):
        """Test render with axis limits."""
        # Arrange
        distances = np.random.rand(100)
        plot = DistancesPlot(distances)

        # Act
        fig, ax = plt.subplots()
        plot.render(ax, xlim=(0, 50), ylim=(0, 1))

        # Assert
        assert ax.get_xlim() == (0, 50)
        assert ax.get_ylim() == (0, 1)
        plt.close(fig)


class TestDistancesPlotRealWorld:
    """Test real-world scenarios."""

    def test_q_residuals_with_threshold(self):
        """Test Q residuals plot with control limit."""
        # Arrange - simulate Q residuals
        np.random.seed(42)
        q_train = np.random.chisquare(df=5, size=100)

        # Act
        plot = DistancesPlot(q_train, confidence_lines=(None, 12.0))
        fig = plot.show(ylabel="Q Residuals (SPE)")

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_t2_vs_q_plot(self):
        """Test T² vs Q plot for outlier detection."""
        # Arrange - simulate T² and Q values
        np.random.seed(42)
        t2_train = np.random.chisquare(df=3, size=100)
        q_train = np.random.chisquare(df=5, size=100)
        distances = np.column_stack([t2_train, q_train])

        # Act
        plot = DistancesPlot(
            distances,
            distances_selection=(0, 1),
            confidence_lines=(9.35, 12.0),
        )
        fig = plot.show(xlabel="Hotelling's T²", ylabel="Q Residuals")

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_outlier_identification(self):
        """Test identifying and annotating outliers."""
        # Arrange
        np.random.seed(42)
        q_values = np.random.chisquare(df=5, size=100)
        threshold = 12.0

        # Identify outliers
        outlier_mask = q_values > threshold
        outlier_indices = np.where(outlier_mask)[0]

        # Create annotations only for outliers
        annotations = [
            f"Outlier {i}" if i in outlier_indices else "" for i in range(100)
        ]

        # Act
        plot = DistancesPlot(
            q_values,
            confidence_lines=(None, threshold),
            annotations=annotations,
        )
        fig = plot.show(ylabel="Q Residuals")

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_multiclass_distances(self):
        """Test distances colored by class."""
        # Arrange
        np.random.seed(42)
        distances = np.random.rand(150)
        classes = np.array(["Class A"] * 50 + ["Class B"] * 50 + ["Class C"] * 50)

        # Act
        plot = DistancesPlot(
            distances,
            color_by=classes,
        )
        fig = plot.show(ylabel="Mahalanobis Distance")

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_composed_datasets_with_different_thresholds(self):
        """Test multiple datasets with their own thresholds."""
        # Arrange
        train_dist = np.random.rand(100, 2)
        test_dist = np.random.rand(50, 2)

        # Act
        fig, ax = plt.subplots()
        DistancesPlot(
            train_dist,
            distances_selection=(0, 1),
            label="Train",
            color="blue",
            confidence_lines=(0.7, 0.8),
        ).render(ax)
        DistancesPlot(
            test_dist,
            distances_selection=(0, 1),
            label="Test",
            color="red",
            confidence_lines=(0.9, 0.95),
        ).render(ax)
        ax.legend()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestDistancesPlotEdgeCases:
    """Test edge cases for DistancesPlot."""

    def test_single_sample(self):
        """Test with single sample."""
        # Arrange
        distances = np.array([0.5])

        # Act
        plot = DistancesPlot(distances)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_zero_distances(self):
        """Test with all zero distances."""
        # Arrange
        distances = np.zeros(100)

        # Act
        plot = DistancesPlot(distances)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_negative_distances(self):
        """Test with negative distance values."""
        # Arrange
        distances = np.random.randn(100)  # Can be negative

        # Act
        plot = DistancesPlot(distances)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_confidence_lines_true(self):
        """Test with confidence_lines=True."""
        distances = np.random.rand(50, 2)
        plot = DistancesPlot(distances, confidence_lines=True)
        # Should set thresholds to None (not implemented yet)
        assert plot.x_threshold is None
        assert plot.y_threshold is None

    def test_confidence_lines_false(self):
        """Test with confidence_lines=False."""
        distances = np.random.rand(50, 2)
        plot = DistancesPlot(distances, confidence_lines=False)
        assert plot.x_threshold is None
        assert plot.y_threshold is None

    def test_1d_distances_edge_case(self):
        """Test with 1D distances array (edge case in _render_plot)."""
        # This tests line 406 in _distances.py
        distances = np.random.rand(50)
        plot = DistancesPlot(distances, distances_selection=0)
        fig, ax = plot.render()
        plt.close(fig)

    def test_render_with_existing_axes_no_labels(self):
        """Test render with existing axes that have no labels."""
        distances = np.random.rand(50, 2)
        plot = DistancesPlot(distances, distances_selection=(0, 1))
        fig, ax = plt.subplots()
        # Axes has no labels, so defaults should be set
        result_fig, result_ax = plot.render(ax=ax)
        assert "Distance" in result_ax.get_xlabel()
        assert "Distance" in result_ax.get_ylabel()
        plt.close(fig)

    def test_render_with_none_x_axis(self):
        """Test render when x_axis is None (sample index case)."""
        distances = np.random.rand(50, 2)
        plot = DistancesPlot(distances, distances_selection=(None, 1))
        fig, ax = plot.render()
        # Should use "Sample Index" as xlabel
        plt.close(fig)

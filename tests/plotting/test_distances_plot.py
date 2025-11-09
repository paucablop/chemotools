"""Tests for the refactored DistancesPlot class."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from chemotools.plotting import DistancesPlot


class TestDistancesPlotBasics:
    """Smoke tests covering the primary plotting entry points."""

    def test_show_with_auto_index(self) -> None:
        y = np.random.rand(50)

        plot = DistancesPlot(y)
        fig = plot.show(title="Auto Index")

        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Sample Index"
        assert ax.get_ylabel() == "Distance"
        plt.close(fig)

    def test_show_with_explicit_x(self) -> None:
        x = np.linspace(0, 5, 25)
        y = np.random.rand(25)

        plot = DistancesPlot(y, x=x, label="Explicit")
        fig = plot.show()

        ax = fig.axes[0]
        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Distance"
        plt.close(fig)

    def test_custom_labels(self) -> None:
        x = np.linspace(0, 1, 40)
        y = np.random.rand(40)

        plot = DistancesPlot(y, x=x)
        fig = plot.show(xlabel="Hotelling's T²", ylabel="Q Residuals")

        ax = fig.axes[0]
        assert ax.get_xlabel() == "Hotelling's T²"
        assert ax.get_ylabel() == "Q Residuals"
        plt.close(fig)

    def test_multiple_datasets_composed(self) -> None:
        x = np.linspace(0, 1, 30)
        y_train = np.random.rand(30)
        y_test = np.random.rand(30)

        fig, ax = plt.subplots()
        DistancesPlot(y_train, x=x, label="Train", color="blue").render(ax)
        DistancesPlot(y_test, x=x, label="Test", color="red").render(ax)
        ax.legend()

        legend = ax.get_legend()
        assert legend is not None
        assert {text.get_text() for text in legend.get_texts()} == {"Train", "Test"}
        plt.close(fig)


class TestValidation:
    """Input validation covering x/y and accessory arrays."""

    def test_y_must_be_1d(self) -> None:
        y = np.random.rand(10, 2)
        with pytest.raises(ValueError, match="must be a 1D array"):
            DistancesPlot(y)

    def test_length_mismatch(self) -> None:
        y = np.random.rand(10)
        x = np.arange(5)
        with pytest.raises(ValueError, match="same length"):
            DistancesPlot(y, x=x)

    def test_color_length_mismatch(self) -> None:
        y = np.random.rand(10)
        color_by = np.random.rand(5)
        with pytest.raises(ValueError, match="same length"):
            DistancesPlot(y, color_by=color_by)

    def test_annotation_length_mismatch(self) -> None:
        y = np.random.rand(10)
        annotations = [f"S{i}" for i in range(5)]
        with pytest.raises(ValueError, match="same length"):
            DistancesPlot(y, annotations=annotations)


class TestConfidenceLines:
    def test_horizontal_and_vertical_lines(self) -> None:
        x = np.linspace(0, 5, 40)
        y = np.random.rand(40)

        plot = DistancesPlot(y, x=x, confidence_lines=(1.5, 0.8))
        fig = plot.show()

        ax = fig.axes[0]
        assert len(ax.lines) >= 2
        plt.close(fig)


class TestColoring:
    def test_categorical_coloring(self) -> None:
        y = np.random.rand(90)
        classes = np.array(["A", "B", "C"] * 30)

        plot = DistancesPlot(y, color_by=classes, colormap="tab10", label="Samples")
        fig = plot.show()

        legend = fig.axes[0].get_legend()
        assert legend is not None
        assert {text.get_text() for text in legend.get_texts()} == {
            "Samples - A",
            "Samples - B",
            "Samples - C",
        }
        plt.close(fig)

    def test_continuous_coloring(self) -> None:
        y = np.random.rand(60)
        weights = np.linspace(0, 1, 60)

        plot = DistancesPlot(y, color_by=weights, colormap="viridis")
        fig = plot.show()

        assert len(fig.axes) > 1  # colorbar axes
        plt.close(fig)


class TestAnnotations:
    def test_annotations_are_added(self) -> None:
        y = np.random.rand(12)
        annotations = [f"S{i}" for i in range(12)]

        plot = DistancesPlot(y, annotations=annotations)
        fig = plot.show()

        texts = [text.get_text() for text in fig.axes[0].texts]
        assert sum(text.startswith("S") for text in texts) >= 1
        plt.close(fig)


class TestRender:
    def test_render_creates_new_figure(self) -> None:
        y = np.random.rand(20)
        plot = DistancesPlot(y)

        fig, ax = plot.render()
        assert isinstance(fig, Figure)
        assert ax in fig.axes
        plt.close(fig)

    def test_render_on_existing_axes(self) -> None:
        y = np.random.rand(20)
        plot = DistancesPlot(y)

        base_fig, base_ax = plt.subplots()
        fig, ax = plot.render(base_ax, xlim=(0, 10), ylim=(0, 1))

        assert fig is base_fig
        assert ax is base_ax
        assert ax.get_xlim() == (0, 10)
        assert ax.get_ylim() == (0, 1)
        plt.close(fig)


class TestRealWorldUsage:
    def test_t2_vs_q_plot(self) -> None:
        rng = np.random.default_rng(42)
        t2 = rng.chisquare(df=3, size=120)
        q = rng.chisquare(df=5, size=120)

        plot = DistancesPlot(
            y=q,
            x=t2,
            confidence_lines=(9.35, 12.0),
        )
        fig = plot.show(
            title="T² vs Q",
            xlabel="Hotelling's T²",
            ylabel="Q Residuals",
        )

        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_outlier_annotations(self) -> None:
        rng = np.random.default_rng(7)
        q = rng.chisquare(df=5, size=80)
        threshold = 11.0
        annotations = ["" for _ in range(80)]
        for idx in np.where(q > threshold)[0]:
            annotations[idx] = f"Outlier {idx}"

        plot = DistancesPlot(
            q, annotations=annotations, confidence_lines=(None, threshold)
        )
        fig = plot.show()

        texts = [text.get_text() for text in fig.axes[0].texts]
        assert any(text.startswith("Outlier") for text in texts)
        plt.close(fig)


def test_show_respects_limits() -> None:
    y = np.random.rand(30)
    plot = DistancesPlot(y)

    fig = plot.show(xlim=(-1, 1), ylim=(0, 0.5))

    ax = fig.axes[0]
    assert ax.get_xlim() == (-1, 1)
    assert ax.get_ylim() == (0, 0.5)
    plt.close(fig)

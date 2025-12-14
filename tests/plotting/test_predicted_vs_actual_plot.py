import numpy as np
import matplotlib.pyplot as plt
import pytest

from chemotools.plotting._predicted_vs_actual import PredictedVsActualPlot


def test_show_multitarget_sets_defaults_and_limits():
    """Test that show method sets default labels and applies custom limits for multi-target data."""
    # Arrange
    base = np.linspace(0.0, 1.0, 20)
    y_true = np.column_stack([base, base + 1.0])
    y_pred = y_true + 0.05
    plot = PredictedVsActualPlot(y_true, y_pred, target_index=1)

    # Act
    fig = plot.show(xlim=(0.0, 2.0), ylim=(1.0, 2.5))
    ax = fig.axes[0]

    # Assert
    assert ax.get_title() == "Predicted vs Actual (Target 2)"
    assert ax.get_xlim() == pytest.approx((0.0, 2.0))
    assert ax.get_ylim() == pytest.approx((1.0, 2.5))
    assert any(line.get_label() == "Ideal" for line in ax.lines)

    # Cleanup
    plt.close(fig)


def test_render_with_existing_axes_and_label_omits_ideal_line():
    """Test that render uses existing axes, displays label legend, and omits ideal line when requested."""
    # Arrange
    y_true = np.linspace(0.0, 1.0, 8)
    y_pred = y_true + 0.1
    plot = PredictedVsActualPlot(
        y_true,
        y_pred,
        label="Calibration",
        color="crimson",
        add_ideal_line=False,
    )
    fig, ax = plt.subplots()

    # Act
    returned_fig, returned_ax = plot.render(ax=ax)

    # Assert
    assert returned_fig is fig and returned_ax is ax
    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["Calibration"]
    assert not ax.lines  # ideal line skipped

    # Cleanup
    plt.close(fig)


def test_render_continuous_color_by_adds_colorbar():
    """Test that continuous color_by values create a colorbar."""
    # Arrange
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=30)
    y_pred = y_true + rng.normal(scale=0.05, size=30)
    color_by = np.linspace(0.0, 1.0, 30)
    plot = PredictedVsActualPlot(y_true, y_pred, color_by=color_by)

    # Act
    fig, ax = plot.render(figsize=(4.0, 4.0))

    # Assert
    assert len(fig.axes) == 2  # scatter axis + colorbar axis
    assert fig.axes[1].get_ylabel() == "Value"

    # Cleanup
    plt.close(fig)


def test_render_categorical_color_by_builds_legend():
    """Test that categorical color_by values create a legend with category labels."""
    # Arrange
    categories = np.array(["Batch A", "Batch B", "Batch A", "Batch C"])
    y_true = np.array([1.0, 2.0, 1.5, 2.5])
    y_pred = y_true + np.array([0.1, -0.2, 0.0, 0.3])
    plot = PredictedVsActualPlot(y_true, y_pred, color_by=categories)
    fig, ax = plt.subplots()

    # Act
    plot.render(ax=ax)

    # Assert
    legend = ax.get_legend()
    assert legend is not None
    # Note: "Ideal" line is also added to legend
    expected_labels = ["Batch A", "Batch B", "Batch C", "Ideal"]
    actual_labels = sorted([text.get_text() for text in legend.get_texts()])
    assert actual_labels == sorted(expected_labels)

    # Cleanup
    plt.close(fig)


def test_predicted_vs_actual_validation_errors():
    """Test that validation errors are raised for invalid inputs."""
    # Arrange
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0])

    # Act & Assert - mismatched shapes
    with pytest.raises(ValueError, match="must have same shape"):
        PredictedVsActualPlot(y_true, y_pred)

    # Act & Assert - empty arrays
    with pytest.raises(ValueError, match="Found array with 0 sample"):
        PredictedVsActualPlot(np.array([]), np.array([]))

    # Act & Assert - target_index out of bounds
    y_multi = np.ones((5, 2))
    with pytest.raises(ValueError, match="target_index 2 is out of bounds"):
        PredictedVsActualPlot(y_multi, y_multi, target_index=2)

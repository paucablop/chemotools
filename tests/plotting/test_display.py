"""Tests for Display protocol and implementations."""

import pytest
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from chemotools.plotting import Display, is_displayable


class MinimalDisplay:
    """Minimal implementation of Display protocol for testing."""

    def show(self, figsize=None, title=None, **kwargs):
        fig, ax = plt.subplots(figsize=figsize or (8, 6))
        ax.plot([1, 2, 3])
        if title:
            ax.set_title(title)
        return fig

    def render(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        ax.plot([1, 2, 3])
        return fig, ax


class NotADisplay:
    """Class that does not implement Display protocol."""

    def plot(self):
        pass


def test_minimal_display_satisfies_protocol():
    """Test that minimal implementation satisfies Display protocol."""
    # Arrange
    plotter = MinimalDisplay()

    # Act
    is_displayable_result = is_displayable(plotter)
    is_instance_result = isinstance(plotter, Display)

    # Assert
    assert is_displayable_result
    assert is_instance_result


def test_not_display_fails_protocol():
    """Test that non-implementing class fails protocol check."""
    # Arrange
    not_plotter = NotADisplay()

    # Act
    is_displayable_result = is_displayable(not_plotter)
    is_instance_result = isinstance(not_plotter, Display)

    # Assert
    assert not is_displayable_result
    assert not is_instance_result


def test_show_returns_figure():
    """Test that show() returns a Figure object."""
    # Arrange
    plotter = MinimalDisplay()

    # Act
    fig = plotter.show()

    # Assert
    assert isinstance(fig, Figure)


def test_show_with_parameters():
    """Test that show() accepts and uses parameters."""
    # Arrange
    plotter = MinimalDisplay()

    # Act
    fig = plotter.show(figsize=(10, 8), title="Test Title")

    # Assert
    assert isinstance(fig, Figure)
    assert fig.get_size_inches()[0] == 10
    assert fig.get_size_inches()[1] == 8


def test_render_returns_tuple():
    """Test that render() returns (Figure, Axes) tuple."""
    # Arrange
    plotter = MinimalDisplay()

    # Act
    result = plotter.render()

    # Assert
    assert isinstance(result, tuple)
    assert len(result) == 2
    fig, ax = result
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_render_with_existing_axes():
    """Test that render() can use existing axes."""
    # Arrange
    plotter = MinimalDisplay()
    fig, ax = plt.subplots()

    # Act
    result_fig, result_ax = plotter.render(ax=ax)

    # Assert
    assert result_fig is fig
    assert result_ax is ax
    plt.close(fig)


def test_render_creates_new_axes_when_none():
    """Test that render() creates new axes when ax=None."""
    # Arrange
    plotter = MinimalDisplay()

    # Act
    fig, ax = plotter.render(ax=None)

    # Assert
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_polymorphic_usage():
    """Test that Display protocol enables polymorphic functions."""

    # Arrange
    def save_plot(plotter: Display, **kwargs) -> Figure:
        """Function that works with any Display implementation."""
        return plotter.show(**kwargs)

    plotter = MinimalDisplay()

    # Act
    fig = save_plot(plotter, title="Polymorphic Test")

    # Assert
    assert isinstance(fig, Figure)


def test_multiple_renders_on_subplot():
    """Test creating subplots with multiple Display objects."""
    # Arrange
    plotter1 = MinimalDisplay()
    plotter2 = MinimalDisplay()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Act
    fig1, ax1 = plotter1.render(ax=axes[0])
    fig2, ax2 = plotter2.render(ax=axes[1])

    # Assert
    assert fig1 is fig
    assert fig2 is fig
    assert ax1 is axes[0]
    assert ax2 is axes[1]
    plt.close(fig)


def test_is_displayable_function():
    """Test the is_displayable helper function."""
    # Arrange
    minimal_display = MinimalDisplay()
    not_display = NotADisplay()

    # Act & Assert
    assert is_displayable(minimal_display)
    assert not is_displayable(not_display)
    assert not is_displayable("string")
    assert not is_displayable(42)
    assert not is_displayable(None)


@pytest.mark.parametrize("figsize", [(8, 6), (10, 8), (12, 10)])
def test_show_with_different_sizes(figsize):
    """Test show() with different figure sizes."""
    # Arrange
    plotter = MinimalDisplay()

    # Act
    fig = plotter.show(figsize=figsize)

    # Assert
    assert fig.get_size_inches()[0] == figsize[0]
    assert fig.get_size_inches()[1] == figsize[1]
    plt.close(fig)

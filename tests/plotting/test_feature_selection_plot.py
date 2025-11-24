"""Tests for FeatureSelectionPlot class."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

from chemotools.plotting import FeatureSelectionPlot


class TestFeatureSelectionPlot:
    """Test functionality of FeatureSelectionPlot."""

    @pytest.fixture
    def data(self):
        """Create dummy data for tests."""
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + np.random.normal(0, 0.1, (5, 100))
        support = np.zeros(100, dtype=bool)
        # Select two regions: indices 10-20 and 50-60
        support[10:20] = True
        support[50:60] = True
        return x, y, support

    def test_initialization_valid(self, data):
        """Test valid initialization."""
        # Arrange
        x, y, support = data

        # Act
        plot = FeatureSelectionPlot(x, y, support)

        # Assert
        np.testing.assert_array_equal(plot.support, support)
        assert plot.selection_color == "red"
        assert plot.selection_alpha == 0.2

    def test_initialization_invalid_support_length(self, data):
        """Test initialization with mismatched support length."""
        # Arrange
        x, y, _ = data
        invalid_support = np.zeros(len(x) + 1, dtype=bool)

        # Act & Assert
        with pytest.raises(ValueError, match="Support mask length"):
            FeatureSelectionPlot(x, y, invalid_support)

    def test_render_adds_spans(self, data):
        """Test that rendering adds vertical spans for selected regions."""
        # Arrange
        x, y, support = data
        plot = FeatureSelectionPlot(x, y, support)

        # Act
        fig, ax = plot.render()

        # Assert
        # Check for polygons/rectangles (axvspan creates Polygon or Rectangle)
        # We expect regions where support is False to be highlighted
        # Note: matplotlib might add other patches, but we look for ours
        spans = [
            child
            for child in ax.get_children()
            if isinstance(child, (Polygon, Rectangle))
            and child.get_label() == "Excluded Features"
        ]

        # Depending on matplotlib version/implementation, axvspan might be a Polygon
        # Let's check patches if children check fails or is ambiguous
        if not spans:
            spans = [
                p
                for p in ax.patches
                if isinstance(p, (Polygon, Rectangle))
                and p.get_label() == "Excluded Features"
            ]

        # We expect at least one span to have the label "Excluded Features"
        # (The implementation only labels the first one to avoid duplicate legend entries)
        assert len(spans) > 0

        # Check color and alpha of the first span found
        span = spans[0]
        assert span.get_facecolor() == (1.0, 0.0, 0.0, 0.2)  # Red with 0.2 alpha (RGBA)

        plt.close(fig)

    def test_custom_styling(self, data):
        """Test custom color and alpha."""
        # Arrange
        x, y, support = data
        color = "blue"
        alpha = 0.5

        # Act
        plot = FeatureSelectionPlot(
            x, y, support, selection_color=color, selection_alpha=alpha
        )
        fig, ax = plot.render()

        # Assert
        # Find the span
        spans = [
            p
            for p in ax.patches
            if isinstance(p, (Polygon, Rectangle))
            and p.get_label() == "Excluded Features"
        ]
        assert len(spans) > 0

        # Check color (blue is (0, 0, 1))
        expected_rgba = (0.0, 0.0, 1.0, 0.5)
        assert spans[0].get_facecolor() == expected_rgba

        plt.close(fig)

    def test_continuous_regions_logic(self):
        """Test the logic for identifying continuous regions."""
        # Arrange
        x = np.arange(10)
        y = np.zeros((1, 10))

        # Case 1: Single block in middle
        support1 = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
        plot1 = FeatureSelectionPlot(x, y, support1)

        # Case 2: Two blocks
        support2 = np.array([1, 1, 0, 0, 1, 1, 1, 0, 0, 0], dtype=bool)
        plot2 = FeatureSelectionPlot(x, y, support2)

        # Case 3: Single point
        support3 = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
        plot3 = FeatureSelectionPlot(x, y, support3)

        # Case 4: All selected
        support4 = np.ones(10, dtype=bool)
        plot4 = FeatureSelectionPlot(x, y, support4)

        # Act
        regions1 = plot1._get_continuous_regions(support1)
        regions2 = plot2._get_continuous_regions(support2)
        regions3 = plot3._get_continuous_regions(support3)
        regions4 = plot4._get_continuous_regions(support4)

        # Assert
        assert regions1 == [(2, 4)]
        assert regions2 == [(0, 1), (4, 6)]
        assert regions3 == [(3, 3)]
        assert regions4 == [(0, 9)]

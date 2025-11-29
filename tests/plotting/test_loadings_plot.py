"""Tests for LoadingsPlot class."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from chemotools.plotting._loadings import LoadingsPlot
from chemotools.plotting import is_displayable


class TestLoadingsPlotBasics:
    """Test basic functionality of LoadingsPlot."""

    def test_implements_display_protocol(self):
        """Test that LoadingsPlot implements Display protocol."""
        # Arrange
        loadings = np.random.randn(100, 5)

        # Act
        plot = LoadingsPlot(loadings, components=0)

        # Assert
        assert is_displayable(plot)

    def test_single_component(self):
        """Test plotting a single component."""
        # Arrange
        loadings = np.random.randn(100, 5)

        # Act
        plot = LoadingsPlot(loadings, components=0)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_multiple_components_overlaid(self):
        """Test plotting multiple components overlaid."""
        # Arrange
        loadings = np.random.randn(100, 5)

        # Act
        plot = LoadingsPlot(loadings, components=[0, 1, 2])
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None  # Should have legend for multiple components
        plt.close(fig)

    def test_single_component_no_legend(self):
        """Test that single component doesn't show legend."""
        # Arrange
        loadings = np.random.randn(100, 5)

        # Act
        plot = LoadingsPlot(loadings, components=0)
        fig = plot.show()

        # Assert
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is None  # No legend for single component
        plt.close(fig)

    def test_with_feature_names(self):
        """Test plotting with custom feature names."""
        # Arrange
        loadings = np.random.randn(100, 5)
        wavenumbers = np.linspace(400, 2500, 100)

        # Act
        plot = LoadingsPlot(loadings, feature_names=wavenumbers, components=0)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_without_feature_names(self):
        """Test plotting without feature names (uses indices)."""
        # Arrange
        loadings = np.random.randn(100, 5)

        # Act
        plot = LoadingsPlot(loadings, components=0)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_axis_labels(self):
        """Test custom x and y axis labels."""
        # Arrange
        loadings = np.random.randn(100, 5)

        # Act
        plot = LoadingsPlot(loadings, components=0)
        fig = plot.show(
            xlabel="Wavenumber (cm⁻¹)",
            ylabel="Loading Coefficient",
        )

        # Assert
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Wavenumber (cm⁻¹)"
        assert ax.get_ylabel() == "Loading Coefficient"
        plt.close(fig)

    def test_default_axis_labels(self):
        """Test default axis labels."""
        # Arrange
        loadings = np.random.randn(100, 5)

        # Act
        plot = LoadingsPlot(loadings, components=0)
        fig = plot.show()

        # Assert
        ax = fig.axes[0]
        assert ax.get_xlabel() == "X-axis"
        assert ax.get_ylabel() == "Y-axis"
        plt.close(fig)

    def test_show_with_title(self):
        """Test show() with custom title."""
        # Arrange
        loadings = np.random.randn(100, 5)
        plot = LoadingsPlot(loadings, components=0)

        # Act
        fig = plot.show(title="PC1 Loadings")

        # Assert
        ax = fig.axes[0]
        assert ax.get_title() == "PC1 Loadings"
        plt.close(fig)

    def test_show_with_figsize(self):
        """Test show() with custom figure size."""
        # Arrange
        loadings = np.random.randn(100, 5)
        plot = LoadingsPlot(loadings, components=0)

        # Act
        fig = plot.show(figsize=(14, 6))

        # Assert
        assert fig.get_size_inches()[0] == 14
        assert fig.get_size_inches()[1] == 6
        plt.close(fig)

    def test_render_returns_figure_and_axes(self):
        """Test that render() returns (Figure, Axes) tuple."""
        # Arrange
        loadings = np.random.randn(100, 5)
        plot = LoadingsPlot(loadings, components=0)

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
        loadings = np.random.randn(100, 5)
        plot = LoadingsPlot(loadings, components=0)
        fig, ax = plt.subplots()

        # Act
        result_fig, result_ax = plot.render(ax=ax)

        # Assert
        assert result_fig is fig
        assert result_ax is ax
        plt.close(fig)

    def test_zero_reference_line(self):
        """Test that a zero reference line is drawn."""
        # Arrange
        loadings = np.random.randn(100, 5)
        plot = LoadingsPlot(loadings, components=0)

        # Act
        fig = plot.show()

        # Assert
        ax = fig.axes[0]
        # Check that there's a horizontal line at y=0
        lines = ax.get_lines()
        # Should have at least 2 lines: the data plot and the zero reference
        assert len(lines) >= 2
        plt.close(fig)


class TestLoadingsPlotComponentValidation:
    """Test component validation."""

    def test_component_validation_at_init(self):
        """Test that invalid components raise error at initialization."""
        # Arrange
        loadings = np.random.randn(100, 3)  # Only 3 components

        # Act & Assert
        with pytest.raises(ValueError, match="Component index 5 is out of bounds"):
            LoadingsPlot(loadings, components=5)

    def test_negative_component_index(self):
        """Test that negative component indices raise error."""
        # Arrange
        loadings = np.random.randn(100, 5)

        # Act & Assert
        with pytest.raises(ValueError, match="Component index -1 is out of bounds"):
            LoadingsPlot(loadings, components=-1)

    def test_multiple_invalid_components(self):
        """Test validation with multiple components, some invalid."""
        # Arrange
        loadings = np.random.randn(100, 3)

        # Act & Assert
        with pytest.raises(ValueError, match="Component index 10 is out of bounds"):
            LoadingsPlot(loadings, components=[0, 1, 10])

    def test_default_component_zero(self):
        """Test that default component is 0."""
        # Arrange
        loadings = np.random.randn(100, 5)

        # Act
        plot = LoadingsPlot(loadings)  # No components specified
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_edge_case_single_component_loadings(self):
        """Test with loadings that have only one component."""
        # Arrange
        loadings = np.random.randn(100, 1)

        # Act
        plot = LoadingsPlot(loadings, components=0)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_edge_case_last_component(self):
        """Test plotting the last available component."""
        # Arrange
        n_components = 5
        loadings = np.random.randn(100, n_components)

        # Act
        plot = LoadingsPlot(loadings, components=n_components - 1)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestLoadingsPlotInputValidation:
    """Test input validation."""

    def test_loadings_must_be_2d(self):
        """Test that loadings must be 2D array."""
        # Arrange
        loadings_1d = np.random.randn(100)

        # Act & Assert
        with pytest.raises(ValueError, match="Expected 2D array, got 1D array instead"):
            LoadingsPlot(loadings_1d, components=0)

    def test_feature_names_length_mismatch(self):
        """Test that feature_names length must match n_features."""
        # Arrange
        loadings = np.random.randn(100, 5)
        feature_names = np.linspace(400, 2500, 50)  # Wrong length!

        # Act & Assert
        with pytest.raises(ValueError, match="feature_names length .* must match"):
            LoadingsPlot(loadings, feature_names=feature_names, components=0)

    def test_feature_names_as_list(self):
        """Test that feature_names can be provided as a list."""
        # Arrange
        loadings = np.random.randn(100, 5)
        feature_names = list(range(100))

        # Act
        plot = LoadingsPlot(loadings, feature_names=feature_names, components=0)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestLoadingsPlotAxisLimits:
    """Test axis limits (xlim/ylim)."""

    def test_xlim(self):
        """Test xlim parameter."""
        # Arrange
        loadings = np.random.randn(1000, 5)
        wavenumbers = np.linspace(400, 4000, 1000)
        plot = LoadingsPlot(loadings, feature_names=wavenumbers, components=0)

        # Act
        fig = plot.show(xlim=(1000, 2000))

        # Assert
        ax = fig.axes[0]
        assert ax.get_xlim() == (1000, 2000)
        plt.close(fig)

    def test_ylim(self):
        """Test ylim parameter."""
        # Arrange
        loadings = np.random.randn(100, 5)
        plot = LoadingsPlot(loadings, components=0)

        # Act
        fig = plot.show(ylim=(-1, 1))

        # Assert
        ax = fig.axes[0]
        assert ax.get_ylim() == (-1, 1)
        plt.close(fig)

    def test_xlim_and_ylim_together(self):
        """Test xlim and ylim together."""
        # Arrange
        loadings = np.random.randn(1000, 5)
        wavenumbers = np.linspace(400, 4000, 1000)
        plot = LoadingsPlot(loadings, feature_names=wavenumbers, components=0)

        # Act
        fig = plot.show(xlim=(1000, 2000), ylim=(-0.5, 0.5))

        # Assert
        ax = fig.axes[0]
        assert ax.get_xlim() == (1000, 2000)
        assert ax.get_ylim() == (-0.5, 0.5)
        plt.close(fig)

    def test_auto_ylim_with_xlim_single_component(self):
        """Test that ylim auto-scales when xlim is set (single component)."""
        # Arrange
        loadings = np.random.randn(1000, 5)
        wavenumbers = np.linspace(400, 4000, 1000)
        plot = LoadingsPlot(loadings, feature_names=wavenumbers, components=0)

        # Act
        fig = plot.show(xlim=(1000, 2000))  # Only xlim, no ylim

        # Assert
        ax = fig.axes[0]
        # ylim should be auto-scaled based on data in xlim range
        ylim = ax.get_ylim()
        assert ylim[0] < ylim[1]  # Valid range
        plt.close(fig)

    def test_auto_ylim_with_xlim_multiple_components(self):
        """Test that ylim auto-scales with multiple components."""
        # Arrange
        loadings = np.random.randn(1000, 5)
        wavenumbers = np.linspace(400, 4000, 1000)
        plot = LoadingsPlot(loadings, feature_names=wavenumbers, components=[0, 1, 2])

        # Act
        fig = plot.show(xlim=(1000, 2000))  # Only xlim, no ylim

        # Assert
        ax = fig.axes[0]
        # ylim should be auto-scaled based on all components in xlim range
        ylim = ax.get_ylim()
        assert ylim[0] < ylim[1]  # Valid range
        plt.close(fig)

    def test_render_with_xlim_ylim(self):
        """Test that render() also respects xlim and ylim."""
        # Arrange
        loadings = np.random.randn(1000, 5)
        wavenumbers = np.linspace(400, 4000, 1000)
        plot = LoadingsPlot(loadings, feature_names=wavenumbers, components=0)

        # Act
        fig, ax = plot.render(xlim=(1500, 2500), ylim=(-0.3, 0.3))

        # Assert
        assert ax.get_xlim() == (1500, 2500)
        assert ax.get_ylim() == (-0.3, 0.3)
        plt.close(fig)


class TestLoadingsPlotCustomStyling:
    """Test custom styling with kwargs."""

    def test_custom_linewidth(self):
        """Test custom linewidth."""
        # Arrange
        loadings = np.random.randn(100, 5)
        plot = LoadingsPlot(loadings, components=0)

        # Act
        fig = plot.show(linewidth=3)

        # Assert
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        lines = [
            line for line in ax.get_lines() if line.get_ydata()[0] != 0
        ]  # Exclude zero line
        if lines:
            assert lines[0].get_linewidth() == 3
        plt.close(fig)

    def test_custom_alpha(self):
        """Test custom alpha transparency."""
        # Arrange
        loadings = np.random.randn(100, 5)
        plot = LoadingsPlot(loadings, components=0)

        # Act
        fig = plot.show(alpha=0.5)

        # Assert
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        lines = [line for line in ax.get_lines() if line.get_ydata()[0] != 0]
        if lines:
            assert lines[0].get_alpha() == 0.5
        plt.close(fig)

    def test_custom_color(self):
        """Test custom line color."""
        # Arrange
        loadings = np.random.randn(100, 5)
        plot = LoadingsPlot(loadings, components=0)

        # Act
        fig = plot.show(color="red")

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_multiple_styling_kwargs(self):
        """Test multiple styling kwargs together."""
        # Arrange
        loadings = np.random.randn(100, 5)
        plot = LoadingsPlot(loadings, components=0)

        # Act
        fig = plot.show(linewidth=2, alpha=0.8, color="darkblue", linestyle="--")

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestLoadingsPlotSubplots:
    """Test creating subplots with render()."""

    def test_render_in_subplots(self):
        """Test rendering multiple loadings in subplots."""
        # Arrange
        loadings = np.random.randn(100, 5)
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Act
        plot1 = LoadingsPlot(loadings, components=0)
        plot1.render(ax=axes[0])

        plot2 = LoadingsPlot(loadings, components=1)
        plot2.render(ax=axes[1])

        # Assert
        assert axes[0].has_data()
        assert axes[1].has_data()
        plt.close(fig)

    def test_render_overlaid_components_in_subplots(self):
        """Test rendering overlaid components in multiple subplots."""
        # Arrange
        loadings = np.random.randn(100, 5)
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Act
        plot1 = LoadingsPlot(loadings, components=[0, 1])
        plot1.render(ax=axes[0])

        plot2 = LoadingsPlot(loadings, components=[2, 3])
        plot2.render(ax=axes[1])

        # Assert
        assert axes[0].has_data()
        assert axes[1].has_data()
        # Both should have legends
        assert axes[0].get_legend() is not None
        assert axes[1].get_legend() is not None
        plt.close(fig)

    def test_complex_grid_layout(self):
        """Test rendering in complex grid layout."""
        # Arrange
        loadings = np.random.randn(100, 5)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Act
        for i, ax in enumerate(axes.flat):
            if i < 4:  # Only use first 4 components
                plot = LoadingsPlot(loadings, components=i)
                plot.render(ax=ax)

        # Assert
        for ax in axes.flat[:4]:
            assert ax.has_data()
        plt.close(fig)


class TestLoadingsPlotEdgeCases:
    """Test edge cases and special scenarios."""

    def test_all_components_overlaid(self):
        """Test plotting all components overlaid."""
        # Arrange
        n_components = 5
        loadings = np.random.randn(100, n_components)

        # Act
        plot = LoadingsPlot(loadings, components=list(range(n_components)))
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        assert ax.get_legend() is not None
        plt.close(fig)

    def test_very_small_loadings(self):
        """Test with very small loading values (near zero)."""
        # Arrange
        loadings = np.random.randn(100, 5) * 1e-10

        # Act
        plot = LoadingsPlot(loadings, components=0)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_very_large_loadings(self):
        """Test with very large loading values."""
        # Arrange
        loadings = np.random.randn(100, 5) * 1e10

        # Act
        plot = LoadingsPlot(loadings, components=0)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_constant_loadings(self):
        """Test with constant loadings (all same value)."""
        # Arrange
        loadings = np.ones((100, 5)) * 0.5

        # Act
        plot = LoadingsPlot(loadings, components=0)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_sparse_loadings(self):
        """Test with sparse loadings (mostly zeros)."""
        # Arrange
        loadings = np.zeros((100, 5))
        loadings[10:15, 0] = 1.0  # Only a few non-zero values

        # Act
        plot = LoadingsPlot(loadings, components=0)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_many_features(self):
        """Test with many features (e.g., high-resolution spectra)."""
        # Arrange
        loadings = np.random.randn(10000, 5)

        # Act
        plot = LoadingsPlot(loadings, components=0)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_few_features(self):
        """Test with very few features."""
        # Arrange
        loadings = np.random.randn(10, 5)

        # Act
        plot = LoadingsPlot(loadings, components=0)
        fig = plot.show()

        # Assert
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_render_with_xlim_triggers_ylim_calculation(self):
        """Test that xlim triggers automatic ylim calculation."""
        # Create loadings with distinct values to ensure ylim calculation is noticeable
        loadings = np.random.rand(100, 5) * 10 - 5  # Range from -5 to 5
        features = np.linspace(4000, 400, 100)
        plot = LoadingsPlot(loadings, feature_names=features, components=[0, 1])
        fig, ax = plot.render(xlim=(3000, 2000))
        # Verify ylim was set (should not be None or default autoscale)
        ylim = ax.get_ylim()
        assert ylim is not None
        assert isinstance(ylim, tuple)
        assert len(ylim) == 2
        # The ylim should encompass the data within xlim range
        # Just verify it's been set to some reasonable values
        assert ylim[0] < ylim[1]
        plt.close(fig)


class TestLoadingsPlotIntegration:
    """Test integration with real-world scenarios."""

    def test_pca_loadings_workflow(self):
        """Test typical PCA loadings workflow."""
        # Arrange - simulate PCA components
        n_features = 1000
        n_components = 5
        loadings = np.random.randn(n_features, n_components)
        # Normalize like PCA would
        for i in range(n_components):
            loadings[:, i] /= np.linalg.norm(loadings[:, i])
        wavenumbers = np.linspace(400, 4000, n_features)

        # Act
        plot = LoadingsPlot(
            loadings,
            feature_names=wavenumbers,
            components=0,
        )
        fig = plot.show(
            title="PC1 Loadings",
            xlabel="Wavenumber (cm⁻¹)",
            ylabel="Loading Coefficient",
        )

        # Assert
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Wavenumber (cm⁻¹)"
        assert ax.get_ylabel() == "Loading Coefficient"
        assert ax.get_title() == "PC1 Loadings"
        plt.close(fig)

    def test_pls_loadings_workflow(self):
        """Test typical PLS loadings workflow."""
        # Arrange - simulate PLS x-loadings
        n_features = 500
        n_components = 3
        loadings = np.random.randn(n_features, n_components)
        wavelengths = np.linspace(200, 2500, n_features)

        # Act - plot all LVs overlaid
        plot = LoadingsPlot(
            loadings,
            feature_names=wavelengths,
            components=[0, 1, 2],
        )
        fig = plot.show(
            title="PLS X-Loadings",
            xlabel="Wavelength (nm)",
            ylabel="PLS Loading",
            linewidth=2,
            alpha=0.7,
        )

        # Assert
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        assert ax.get_legend() is not None
        plt.close(fig)

    def test_dashboard_creation(self):
        """Test creating an integrated dashboard with multiple loadings plots."""
        # Arrange
        loadings = np.random.randn(1000, 5)
        wavenumbers = np.linspace(400, 4000, 1000)
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        # Act - create dashboard with first 3 PCs
        for i, ax in enumerate(axes):
            plot = LoadingsPlot(
                loadings,
                feature_names=wavenumbers,
                components=i,
            )
            plot.render(
                ax=ax,
                xlabel="Wavenumber (cm⁻¹)",
                ylabel="Loading",
            )
            ax.set_title(f"PC{i + 1} Loadings")
            ax.grid(alpha=0.3)

        # Assert
        for ax in axes:
            assert ax.has_data()
            assert ax.get_title() in ["PC1 Loadings", "PC2 Loadings", "PC3 Loadings"]
        plt.close(fig)

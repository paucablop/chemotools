"""Tests for spectra plot creation functions."""

import numpy as np
import pytest
import matplotlib.pyplot as plt

from chemotools.inspector.helpers._spectra import (
    create_spectra_plots_single_dataset,
    create_spectra_plots_multi_dataset,
)


@pytest.fixture
def sample_spectra_data():
    """Create sample spectra data for testing."""
    np.random.seed(42)
    n_samples = 50
    n_features = 100
    return {
        "X_raw": np.random.rand(n_samples, n_features),
        "X_preprocessed": np.random.rand(n_samples, n_features) * 0.8,
        "wavenumbers": np.linspace(4000, 400, n_features),
        "y": np.random.randint(0, 3, n_samples),
    }


class TestCreateSpectraPlotsSingleDataset:
    """Tests for create_spectra_plots_single_dataset function."""

    def test_basic_spectra_plots(self, sample_spectra_data):
        """Test basic spectra plots creation."""
        # Arrange
        X_raw = sample_spectra_data["X_raw"]
        X_preprocessed = sample_spectra_data["X_preprocessed"]
        y = sample_spectra_data["y"]
        wavenumbers = sample_spectra_data["wavenumbers"]
        dataset_name = "train"
        color_by = "y"
        xlabel = "Wavenumber (cm⁻¹)"
        xlim = None
        figsize = (12, 5)

        # Act
        figs = create_spectra_plots_single_dataset(
            X_raw=X_raw,
            X_preprocessed=X_preprocessed,
            y=y,
            x_axis=wavenumbers,
            preprocessed_x_axis=wavenumbers,
            dataset_name=dataset_name,
            color_by=color_by,
            xlabel=xlabel,
            xlim=xlim,
            figsize=figsize,
        )

        # Assert
        assert isinstance(figs, dict)
        assert "raw_spectra" in figs
        assert "preprocessed_spectra" in figs

        # Cleanup
        for fig in figs.values():
            plt.close(fig)

    def test_without_y(self, sample_spectra_data):
        """Test spectra plots without y data."""
        # Arrange
        X_raw = sample_spectra_data["X_raw"]
        X_preprocessed = sample_spectra_data["X_preprocessed"]
        y = None
        wavenumbers = sample_spectra_data["wavenumbers"]
        dataset_name = "train"
        color_by = None
        xlabel = "Wavenumber (cm⁻¹)"
        xlim = None
        figsize = (12, 5)

        # Act
        figs = create_spectra_plots_single_dataset(
            X_raw=X_raw,
            X_preprocessed=X_preprocessed,
            y=y,
            x_axis=wavenumbers,
            preprocessed_x_axis=wavenumbers,
            dataset_name=dataset_name,
            color_by=color_by,
            xlabel=xlabel,
            xlim=xlim,
            figsize=figsize,
        )

        # Assert
        assert isinstance(figs, dict)
        assert len(figs) == 2

        # Cleanup
        for fig in figs.values():
            plt.close(fig)

    def test_with_xlim(self, sample_spectra_data):
        """Test spectra plots with xlim."""
        # Arrange
        X_raw = sample_spectra_data["X_raw"]
        X_preprocessed = sample_spectra_data["X_preprocessed"]
        y = sample_spectra_data["y"]
        wavenumbers = sample_spectra_data["wavenumbers"]
        dataset_name = "test"
        color_by = None
        xlabel = "Wavenumber (cm⁻¹)"
        xlim = (3000, 2800)
        figsize = (12, 5)

        # Act
        figs = create_spectra_plots_single_dataset(
            X_raw=X_raw,
            X_preprocessed=X_preprocessed,
            y=y,
            x_axis=wavenumbers,
            preprocessed_x_axis=wavenumbers,
            dataset_name=dataset_name,
            color_by=color_by,
            xlabel=xlabel,
            xlim=xlim,
            figsize=figsize,
        )

        # Assert
        assert isinstance(figs, dict)

        # Cleanup
        for fig in figs.values():
            plt.close(fig)

    def test_with_feature_indices(self, sample_spectra_data):
        """Test spectra plots with feature indices instead of wavenumbers."""
        # Arrange
        X_raw = sample_spectra_data["X_raw"]
        X_preprocessed = sample_spectra_data["X_preprocessed"]
        y = None
        n_features = X_raw.shape[1]
        wavenumbers = np.arange(n_features)
        dataset_name = "val"
        color_by = None
        xlabel = "Feature Index"
        xlim = None
        figsize = (12, 5)

        # Act
        figs = create_spectra_plots_single_dataset(
            X_raw=X_raw,
            X_preprocessed=X_preprocessed,
            y=y,
            x_axis=wavenumbers,
            preprocessed_x_axis=wavenumbers,
            dataset_name=dataset_name,
            color_by=color_by,
            xlabel=xlabel,
            xlim=xlim,
            figsize=figsize,
        )

        # Assert
        assert isinstance(figs, dict)

        # Cleanup
        for fig in figs.values():
            plt.close(fig)


class TestCreateSpectraPlotsMultiDataset:
    """Tests for create_spectra_plots_multi_dataset function."""

    def test_multi_dataset_spectra(self, sample_spectra_data):
        """Test multi-dataset spectra plots."""
        # Arrange
        raw_data = {
            "train": sample_spectra_data["X_raw"],
            "test": np.random.rand(30, 100),
        }
        preprocessed_data = {
            "train": sample_spectra_data["X_preprocessed"],
            "test": np.random.rand(30, 100) * 0.8,
        }
        wavenumbers = sample_spectra_data["wavenumbers"]
        xlabel = "Wavenumber (cm⁻¹)"
        xlim = None
        figsize = (12, 5)

        # Act
        figs = create_spectra_plots_multi_dataset(
            raw_data=raw_data,
            preprocessed_data=preprocessed_data,
            x_axis=wavenumbers,
            preprocessed_x_axis=wavenumbers,
            xlabel=xlabel,
            xlim=xlim,
            figsize=figsize,
        )

        # Assert
        assert isinstance(figs, dict)
        assert "raw_spectra" in figs
        assert "preprocessed_spectra" in figs

        # Cleanup
        for fig in figs.values():
            plt.close(fig)

    def test_single_dataset_in_multi(self, sample_spectra_data):
        """Test multi-dataset function with single dataset."""
        # Arrange
        raw_data = {"train": sample_spectra_data["X_raw"]}
        preprocessed_data = {"train": sample_spectra_data["X_preprocessed"]}
        wavenumbers = sample_spectra_data["wavenumbers"]
        xlabel = "Wavenumber (cm⁻¹)"
        xlim = None
        figsize = (12, 5)

        # Act
        figs = create_spectra_plots_multi_dataset(
            raw_data=raw_data,
            preprocessed_data=preprocessed_data,
            x_axis=wavenumbers,
            preprocessed_x_axis=wavenumbers,
            xlabel=xlabel,
            xlim=xlim,
            figsize=figsize,
        )

        # Assert
        assert isinstance(figs, dict)
        assert len(figs) == 2

        # Cleanup
        for fig in figs.values():
            plt.close(fig)

    def test_with_xlim_multi(self, sample_spectra_data):
        """Test multi-dataset spectra plots with xlim."""
        # Arrange
        raw_data = {
            "train": sample_spectra_data["X_raw"],
            "test": np.random.rand(30, 100),
        }
        preprocessed_data = {
            "train": sample_spectra_data["X_preprocessed"],
            "test": np.random.rand(30, 100) * 0.8,
        }
        wavenumbers = sample_spectra_data["wavenumbers"]
        xlabel = "Wavenumber (cm⁻¹)"
        xlim = (3500, 2500)
        figsize = (12, 5)

        # Act
        figs = create_spectra_plots_multi_dataset(
            raw_data=raw_data,
            preprocessed_data=preprocessed_data,
            x_axis=wavenumbers,
            preprocessed_x_axis=wavenumbers,
            xlabel=xlabel,
            xlim=xlim,
            figsize=figsize,
        )

        # Assert
        assert isinstance(figs, dict)

        # Cleanup
        for fig in figs.values():
            plt.close(fig)

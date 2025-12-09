"""Tests for SpectraMixin."""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from chemotools.inspector.core.spectra import SpectraMixin


class _DummyTransformer(Pipeline):
    """Dummy transformer for testing."""

    def __init__(self):
        super().__init__([("scaler", StandardScaler())])
        # Fit the pipeline with dummy data
        self.fit(np.random.rand(10, 5))


class _DummyInspectorBase:
    """Base class providing required methods for SpectraMixin."""

    def __init__(
        self,
        *,
        raw_data,
        preprocessed_data,
        x_axis,
        preprocessed_feature_names,
        feature_names=None,
        transformer=None,
        **kwargs,
    ):
        self._raw_store = raw_data
        self._preprocessed_store = preprocessed_data
        self._x_axis = x_axis
        self._preprocessed_feature_names = preprocessed_feature_names
        self._feature_names = feature_names
        self._transformer = transformer
        super().__init__(**kwargs)

    @property
    def transformer(self):
        return self._transformer

    @property
    def x_axis(self):
        return self._x_axis

    @property
    def feature_names(self):
        return self._feature_names

    def _get_raw_data(self, dataset: str):
        return self._raw_store[dataset]

    def _get_preprocessed_data(self, dataset: str):
        return self._preprocessed_store[dataset]

    def _get_preprocessed_feature_names(self):
        return self._preprocessed_feature_names

    def _get_preprocessed_x_axis(self):
        """Get x_axis after feature selection (delegates to _get_preprocessed_feature_names)."""
        return self._get_preprocessed_feature_names()


class DummySpectraInspector(SpectraMixin, _DummyInspectorBase):
    """Dummy inspector for testing SpectraMixin."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@pytest.fixture
def sample_spectra_data():
    """Create sample spectra data for testing."""
    np.random.seed(42)
    n_samples_train = 50
    n_samples_test = 20
    n_features = 100

    X_train_raw = np.random.rand(n_samples_train, n_features)
    X_train_preprocessed = np.random.rand(n_samples_train, n_features) * 0.8
    y_train = np.random.randint(0, 3, n_samples_train)

    X_test_raw = np.random.rand(n_samples_test, n_features)
    X_test_preprocessed = np.random.rand(n_samples_test, n_features) * 0.8
    y_test = np.random.randint(0, 3, n_samples_test)

    wavenumbers = np.linspace(4000, 400, n_features)

    return {
        "raw_data": {
            "train": (X_train_raw, y_train),
            "test": (X_test_raw, y_test),
        },
        "preprocessed_data": {
            "train": X_train_preprocessed,
            "test": X_test_preprocessed,
        },
        "x_axis": wavenumbers,
        "preprocessed_feature_names": wavenumbers,
    }


class TestSpectraMixinInspectSpectra:
    """Tests for inspect_spectra method."""

    def test_inspect_spectra_returns_two_figures(self, sample_spectra_data):
        """Test that inspect_spectra returns raw and preprocessed figures."""
        # Arrange
        inspector = DummySpectraInspector(
            raw_data=sample_spectra_data["raw_data"],
            preprocessed_data=sample_spectra_data["preprocessed_data"],
            x_axis=sample_spectra_data["x_axis"],
            preprocessed_feature_names=sample_spectra_data[
                "preprocessed_feature_names"
            ],
            feature_names=sample_spectra_data["x_axis"],
            transformer=_DummyTransformer(),
        )

        # Act
        figures = inspector.inspect_spectra()

        # Assert
        assert isinstance(figures, dict)
        assert "raw_spectra" in figures
        assert "preprocessed_spectra" in figures
        assert all(isinstance(fig, plt.Figure) for fig in figures.values())

        # Cleanup
        for fig in figures.values():
            plt.close(fig)

    def test_inspect_spectra_without_transformer_raises_error(
        self, sample_spectra_data
    ):
        """Test that inspect_spectra raises ValueError when no transformer exists."""
        # Arrange
        inspector = DummySpectraInspector(
            raw_data=sample_spectra_data["raw_data"],
            preprocessed_data=sample_spectra_data["preprocessed_data"],
            x_axis=sample_spectra_data["x_axis"],
            preprocessed_feature_names=sample_spectra_data[
                "preprocessed_feature_names"
            ],
            transformer=None,  # No transformer
        )

        # Act & Assert
        with pytest.raises(
            ValueError, match="Spectra inspection requires a preprocessing pipeline"
        ):
            inspector.inspect_spectra()

    def test_inspect_spectra_single_dataset(self, sample_spectra_data):
        """Test inspect_spectra with a single dataset."""
        # Arrange
        inspector = DummySpectraInspector(
            raw_data=sample_spectra_data["raw_data"],
            preprocessed_data=sample_spectra_data["preprocessed_data"],
            x_axis=sample_spectra_data["x_axis"],
            preprocessed_feature_names=sample_spectra_data[
                "preprocessed_feature_names"
            ],
            feature_names=sample_spectra_data["x_axis"],
            transformer=_DummyTransformer(),
        )

        # Act
        figures = inspector.inspect_spectra(dataset="train")

        # Assert
        assert len(figures) == 2
        assert "raw_spectra" in figures
        assert "preprocessed_spectra" in figures

        # Cleanup
        for fig in figures.values():
            plt.close(fig)

    def test_inspect_spectra_multiple_datasets(self, sample_spectra_data):
        """Test inspect_spectra with multiple datasets."""
        # Arrange
        inspector = DummySpectraInspector(
            raw_data=sample_spectra_data["raw_data"],
            preprocessed_data=sample_spectra_data["preprocessed_data"],
            x_axis=sample_spectra_data["x_axis"],
            preprocessed_feature_names=sample_spectra_data[
                "preprocessed_feature_names"
            ],
            feature_names=sample_spectra_data["x_axis"],
            transformer=_DummyTransformer(),
        )

        # Act
        figures = inspector.inspect_spectra(dataset=["train", "test"])

        # Assert
        assert len(figures) == 2
        assert "raw_spectra" in figures
        assert "preprocessed_spectra" in figures

        # Cleanup
        for fig in figures.values():
            plt.close(fig)

    def test_inspect_spectra_with_color_by_y(self, sample_spectra_data):
        """Test inspect_spectra with color_by='y' enabled."""
        # Arrange
        inspector = DummySpectraInspector(
            raw_data=sample_spectra_data["raw_data"],
            preprocessed_data=sample_spectra_data["preprocessed_data"],
            x_axis=sample_spectra_data["x_axis"],
            preprocessed_feature_names=sample_spectra_data[
                "preprocessed_feature_names"
            ],
            feature_names=sample_spectra_data["x_axis"],
            transformer=_DummyTransformer(),
        )

        # Act
        figures = inspector.inspect_spectra(color_by="y")

        # Assert
        assert len(figures) == 2

        # Cleanup
        for fig in figures.values():
            plt.close(fig)

    def test_inspect_spectra_without_color_by_y(self, sample_spectra_data):
        """Test inspect_spectra with color_by='y' disabled."""
        # Arrange
        inspector = DummySpectraInspector(
            raw_data=sample_spectra_data["raw_data"],
            preprocessed_data=sample_spectra_data["preprocessed_data"],
            x_axis=sample_spectra_data["x_axis"],
            preprocessed_feature_names=sample_spectra_data[
                "preprocessed_feature_names"
            ],
            feature_names=sample_spectra_data["x_axis"],
            transformer=_DummyTransformer(),
        )

        # Act
        figures = inspector.inspect_spectra(color_by=None)

        # Assert
        assert len(figures) == 2

        # Cleanup
        for fig in figures.values():
            plt.close(fig)

    def test_inspect_spectra_with_xlim(self, sample_spectra_data):
        """Test inspect_spectra with custom xlim."""
        # Arrange
        inspector = DummySpectraInspector(
            raw_data=sample_spectra_data["raw_data"],
            preprocessed_data=sample_spectra_data["preprocessed_data"],
            x_axis=sample_spectra_data["x_axis"],
            preprocessed_feature_names=sample_spectra_data[
                "preprocessed_feature_names"
            ],
            feature_names=sample_spectra_data["x_axis"],
            transformer=_DummyTransformer(),
        )

        # Act
        figures = inspector.inspect_spectra(xlim=(1000, 3000))

        # Assert
        assert len(figures) == 2

        # Cleanup
        for fig in figures.values():
            plt.close(fig)

    def test_inspect_spectra_custom_figsize(self, sample_spectra_data):
        """Test inspect_spectra with custom figsize."""
        # Arrange
        inspector = DummySpectraInspector(
            raw_data=sample_spectra_data["raw_data"],
            preprocessed_data=sample_spectra_data["preprocessed_data"],
            x_axis=sample_spectra_data["x_axis"],
            preprocessed_feature_names=sample_spectra_data[
                "preprocessed_feature_names"
            ],
            feature_names=sample_spectra_data["x_axis"],
            transformer=_DummyTransformer(),
        )

        # Act
        figures = inspector.inspect_spectra(figsize=(10, 4))

        # Assert
        raw_fig = figures["raw_spectra"]
        assert raw_fig.get_size_inches()[0] == pytest.approx(10, rel=0.1)

        # Cleanup
        for fig in figures.values():
            plt.close(fig)


class TestSpectraMixinGetPreprocessedXAxis:
    """Tests for _get_preprocessed_x_axis method."""

    def test_get_preprocessed_x_axis_returns_feature_names(self, sample_spectra_data):
        """Test that _get_preprocessed_x_axis returns preprocessed feature names."""
        # Arrange
        preprocessed_names = np.array([100, 200, 300])
        inspector = DummySpectraInspector(
            raw_data=sample_spectra_data["raw_data"],
            preprocessed_data=sample_spectra_data["preprocessed_data"],
            x_axis=sample_spectra_data["x_axis"],
            preprocessed_feature_names=preprocessed_names,
            transformer=_DummyTransformer(),
        )

        # Act
        result = inspector._get_preprocessed_x_axis()

        # Assert
        np.testing.assert_array_equal(result, preprocessed_names)

    def test_get_preprocessed_x_axis_with_feature_selection(self, sample_spectra_data):
        """Test _get_preprocessed_x_axis when feature selection reduces features."""
        # Arrange
        original_x_axis = np.linspace(4000, 400, 100)
        preprocessed_x_axis = np.linspace(3500, 500, 80)  # Simulating feature selection

        inspector = DummySpectraInspector(
            raw_data=sample_spectra_data["raw_data"],
            preprocessed_data=sample_spectra_data["preprocessed_data"],
            x_axis=original_x_axis,
            preprocessed_feature_names=preprocessed_x_axis,
            transformer=_DummyTransformer(),
        )

        # Act
        result = inspector._get_preprocessed_x_axis()

        # Assert
        np.testing.assert_array_equal(result, preprocessed_x_axis)
        assert len(result) < len(original_x_axis)


class TestSpectraMixinIntegration:
    """Integration tests for SpectraMixin with realistic scenarios."""

    def test_inspect_spectra_test_dataset(self, sample_spectra_data):
        """Test inspect_spectra with test dataset."""
        # Arrange
        inspector = DummySpectraInspector(
            raw_data=sample_spectra_data["raw_data"],
            preprocessed_data=sample_spectra_data["preprocessed_data"],
            x_axis=sample_spectra_data["x_axis"],
            preprocessed_feature_names=sample_spectra_data[
                "preprocessed_feature_names"
            ],
            feature_names=sample_spectra_data["x_axis"],
            transformer=_DummyTransformer(),
        )

        # Act
        figures = inspector.inspect_spectra(dataset="test")

        # Assert
        assert len(figures) == 2
        assert "raw_spectra" in figures
        assert "preprocessed_spectra" in figures

        # Cleanup
        for fig in figures.values():
            plt.close(fig)

    def test_inspect_spectra_without_feature_names(self, sample_spectra_data):
        """Test inspect_spectra when feature_names is None."""
        # Arrange
        inspector = DummySpectraInspector(
            raw_data=sample_spectra_data["raw_data"],
            preprocessed_data=sample_spectra_data["preprocessed_data"],
            x_axis=sample_spectra_data["x_axis"],
            preprocessed_feature_names=sample_spectra_data[
                "preprocessed_feature_names"
            ],
            feature_names=None,  # No feature names
            transformer=_DummyTransformer(),
        )

        # Act
        figures = inspector.inspect_spectra()

        # Assert
        assert len(figures) == 2

        # Cleanup
        for fig in figures.values():
            plt.close(fig)

    def test_inspect_spectra_with_list_dataset_input(self, sample_spectra_data):
        """Test inspect_spectra with list of datasets."""
        # Arrange
        inspector = DummySpectraInspector(
            raw_data=sample_spectra_data["raw_data"],
            preprocessed_data=sample_spectra_data["preprocessed_data"],
            x_axis=sample_spectra_data["x_axis"],
            preprocessed_feature_names=sample_spectra_data[
                "preprocessed_feature_names"
            ],
            feature_names=sample_spectra_data["x_axis"],
            transformer=_DummyTransformer(),
        )

        # Act
        figures = inspector.inspect_spectra(dataset=["train", "test"])

        # Assert
        assert isinstance(figures, dict)
        assert len(figures) == 2

        # Cleanup
        for fig in figures.values():
            plt.close(fig)

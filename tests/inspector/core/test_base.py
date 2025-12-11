import numpy as np
import pytest
from unittest import mock
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin
import matplotlib.pyplot as plt

from chemotools.inspector.core.base import (
    _BaseInspector,
    InspectorDataset,
)


# Concrete implementation for testing the abstract base class
class ConcreteInspector(_BaseInspector):
    """Concrete implementation of _BaseInspector for testing."""

    def plot_scores(self, components=(0, 1), datasets=["train"], **kwargs):
        """Simple implementation for testing."""
        fig, ax = plt.subplots()
        for dataset in datasets:
            if dataset in self.datasets_:
                X_preprocessed = self._get_preprocessed_data(dataset)
                scores = self.estimator_.transform(X_preprocessed)
                ax.scatter(scores[:, components[0]], scores[:, components[1]])
        return fig


class TestBaseInspectorInitialization:
    """Test initialization of _BaseInspector."""

    def test_init_with_fitted_pca(self, fitted_pca, dummy_data_loader):
        """Test initialization with fitted PCA model."""
        # Arrange
        X, _ = dummy_data_loader

        # Act
        inspector = ConcreteInspector(model=fitted_pca, X_train=X)

        # Assert
        assert inspector.estimator_ is fitted_pca
        assert inspector.transformer_ is None
        assert inspector.n_components_ == 2
        assert inspector.n_features_in_ == 3
        assert "train" in inspector.datasets_
        assert np.array_equal(inspector.datasets_["train"].X, X)

    def test_init_with_fitted_pls(self, fitted_pls, dummy_data_loader):
        """Test initialization with fitted PLS model."""
        # Arrange
        X, y = dummy_data_loader

        # Act
        inspector = ConcreteInspector(model=fitted_pls, X_train=X, y_train=y)

        # Assert
        assert inspector.estimator_ is fitted_pls
        assert inspector.transformer_ is None
        assert inspector.n_components_ == 2
        assert inspector.n_features_in_ == 3
        assert np.array_equal(inspector.datasets_["train"].X, X)
        assert np.array_equal(inspector.datasets_["train"].y, y)

    def test_init_with_pipeline(self, fitted_pipeline_pca, dummy_data_loader):
        """Test initialization with fitted pipeline."""
        # Arrange
        X, _ = dummy_data_loader

        # Act
        inspector = ConcreteInspector(model=fitted_pipeline_pca, X_train=X)

        # Assert
        assert isinstance(inspector.estimator_, PCA)
        assert inspector.transformer_ is not None
        assert isinstance(inspector.transformer_.steps[0][1], StandardScaler)
        assert inspector.n_components_ == 2
        assert inspector.n_features_in_ == 3

    def test_init_with_test_data(self, fitted_pca, dummy_data_loader):
        """Test initialization with test data."""
        # Arrange
        X, _ = dummy_data_loader
        X_train, X_test = X[:80], X[80:]

        # Act
        inspector = ConcreteInspector(model=fitted_pca, X_train=X_train, X_test=X_test)

        # Assert
        assert "train" in inspector.datasets_
        assert "test" in inspector.datasets_
        assert np.array_equal(inspector.datasets_["train"].X, X_train)
        assert np.array_equal(inspector.datasets_["test"].X, X_test)

    def test_init_with_validation_data(self, fitted_pca, dummy_data_loader):
        """Test initialization with validation data."""
        # Arrange
        X, _ = dummy_data_loader
        X_train, X_val = X[:80], X[80:]

        # Act
        inspector = ConcreteInspector(model=fitted_pca, X_train=X_train, X_val=X_val)

        # Assert
        assert "train" in inspector.datasets_
        assert "val" in inspector.datasets_
        assert np.array_equal(inspector.datasets_["train"].X, X_train)
        assert np.array_equal(inspector.datasets_["val"].X, X_val)

    def test_init_with_all_datasets(self, fitted_pls, dummy_data_loader):
        """Test initialization with all dataset splits."""
        # Arrange
        X, y = dummy_data_loader
        X_train, X_test, X_val = X[:60], X[60:80], X[80:]
        y_train, y_test, y_val = y[:60], y[60:80], y[80:]

        # Act
        inspector = ConcreteInspector(
            model=fitted_pls,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            X_val=X_val,
            y_val=y_val,
        )

        # Assert
        assert "train" in inspector.datasets_
        assert "test" in inspector.datasets_
        assert "val" in inspector.datasets_
        assert np.array_equal(inspector.datasets_["train"].X, X_train)
        assert np.array_equal(inspector.datasets_["test"].X, X_test)
        assert np.array_equal(inspector.datasets_["val"].X, X_val)

    def test_init_with_feature_names(self, fitted_pca, dummy_data_loader):
        """Test initialization with feature names."""
        # Arrange
        X, _ = dummy_data_loader
        feature_names = ["Feature 1", "Feature 2", "Feature 3"]

        # Act
        inspector = ConcreteInspector(
            model=fitted_pca, X_train=X, feature_names=feature_names
        )

        # Assert
        assert inspector.feature_names is not None
        assert np.array_equal(inspector.feature_names, np.asarray(feature_names))

    def test_init_with_sample_labels(self, fitted_pca, dummy_data_loader):
        """Test initialization with sample labels."""
        # Arrange
        X, _ = dummy_data_loader
        sample_labels = {"train": np.array(["A"] * 50 + ["B"] * 50)}

        # Act
        inspector = ConcreteInspector(
            model=fitted_pca, X_train=X, sample_labels=sample_labels
        )

        # Assert - labels are stored in InspectorDataset
        assert inspector.datasets_["train"].labels is not None
        assert len(inspector.datasets_["train"].labels) == 100

    def test_init_with_unfitted_model(self, unfitted_pca, dummy_data_loader):
        """Test that unfitted model raises error."""
        # Arrange
        X, _ = dummy_data_loader

        # Assert
        with pytest.raises(Exception):  # NotFittedError
            ConcreteInspector(model=unfitted_pca, X_train=X)

    def test_init_with_invalid_model(self, fitted_invalid_model, dummy_data_loader):
        """Test that invalid model type raises error."""
        # Arrange
        X, _ = dummy_data_loader

        # Assert
        with pytest.raises(TypeError):
            ConcreteInspector(model=fitted_invalid_model, X_train=X)


class TestBaseInspectorOrganizeDatasets:
    """Test _organize_datasets method."""

    def test_organize_train_only(self, fitted_pca, dummy_data_loader):
        """Test organizing training data only."""
        # Arrange
        X, y = dummy_data_loader

        # Act
        inspector = ConcreteInspector(model=fitted_pca, X_train=X, y_train=y)
        datasets = inspector.datasets_

        # Assert
        assert len(datasets) == 1
        assert "train" in datasets
        assert "test" not in datasets
        assert "val" not in datasets

    def test_organize_train_and_test(self, fitted_pca, dummy_data_loader):
        """Test organizing training and test data."""
        # Arrange
        X, _ = dummy_data_loader
        X_train, X_test = X[:80], X[80:]

        # Act
        inspector = ConcreteInspector(model=fitted_pca, X_train=X_train, X_test=X_test)
        datasets = inspector.datasets_

        # Assert
        assert len(datasets) == 2
        assert "train" in datasets
        assert "test" in datasets
        assert "val" not in datasets

    def test_organize_all_datasets(self, fitted_pca, dummy_data_loader):
        """Test organizing all dataset splits."""
        # Arrange
        X, _ = dummy_data_loader
        X_train, X_test, X_val = X[:60], X[60:80], X[80:]

        # Act
        inspector = ConcreteInspector(
            model=fitted_pca, X_train=X_train, X_test=X_test, X_val=X_val
        )
        datasets = inspector.datasets_

        # Assert
        assert len(datasets) == 3
        assert "train" in datasets
        assert "test" in datasets
        assert "val" in datasets


class TestBaseInspectorGetNComponents:
    """Test _get_n_components method."""

    def test_get_n_components_from_pca(self, fitted_pca, dummy_data_loader):
        """Test getting n_components from PCA model."""
        # Arrange
        X, _ = dummy_data_loader
        inspector = ConcreteInspector(model=fitted_pca, X_train=X)

        # Act
        n_components = inspector.n_components_

        # Assert
        assert n_components == 2

    def test_get_n_components_from_pls(self, fitted_pls, dummy_data_loader):
        """Test getting n_components from PLS model."""
        # Arrange
        X, y = dummy_data_loader
        inspector = ConcreteInspector(model=fitted_pls, X_train=X, y_train=y)

        # Act
        n_components = inspector.n_components_

        # Assert
        assert n_components == 2

    def test_get_n_components_different_values(self, dummy_data_loader):
        """Test getting n_components with different values."""
        # Arrange
        X, y = dummy_data_loader
        pls_3 = PLSRegression(n_components=3).fit(X, y)

        # Act
        inspector = ConcreteInspector(model=pls_3, X_train=X, y_train=y)

        # Assert
        assert inspector.n_components_ == 3


class TestBaseInspectorTransformData:
    """Test _transform_data method."""

    def test_transform_without_pipeline(self, fitted_pca, dummy_data_loader):
        """Test transform without preprocessing pipeline."""
        # Arrange
        X, _ = dummy_data_loader
        inspector = ConcreteInspector(model=fitted_pca, X_train=X)

        # Act
        X_transformed = inspector._transform_data(X)

        # Assert
        assert np.array_equal(X_transformed, X)

    def test_transform_with_pipeline(self, fitted_pipeline_pca, dummy_data_loader):
        """Test transform with preprocessing pipeline."""
        # Arrange
        X, _ = dummy_data_loader
        inspector = ConcreteInspector(model=fitted_pipeline_pca, X_train=X)

        # Act
        X_transformed = inspector._transform_data(X)

        # Assert
        # Should be scaled (mean ~0, std ~1)
        assert not np.array_equal(X_transformed, X)
        assert np.allclose(np.mean(X_transformed, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(X_transformed, axis=0), 1, atol=1e-10)


class TestBaseInspectorPlotScores:
    """Test plot_scores method."""

    def test_plot_scores_is_abstract(self):
        """Test that plot_scores must be implemented by subclasses."""
        # The ConcreteInspector should implement it
        assert hasattr(ConcreteInspector, "plot_scores")
        assert callable(getattr(ConcreteInspector, "plot_scores"))

    def test_plot_scores_basic(self, fitted_pca, dummy_data_loader):
        """Test basic plotting functionality."""
        # Arrange
        X, _ = dummy_data_loader
        inspector = ConcreteInspector(model=fitted_pca, X_train=X)

        # Act
        fig = inspector.plot_scores()

        # Assert
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_scores_multiple_datasets(self, fitted_pca, dummy_data_loader):
        """Test plotting multiple datasets."""
        # Arrange
        X, _ = dummy_data_loader
        X_train, X_test = X[:80], X[80:]
        inspector = ConcreteInspector(model=fitted_pca, X_train=X_train, X_test=X_test)

        # Act
        fig = inspector.plot_scores(datasets=["train", "test"])

        # Assert
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_scores_different_components(self, dummy_data_loader):
        """Test plotting different component pairs."""
        # Arrange
        X, _ = dummy_data_loader
        pca_3 = PCA(n_components=3).fit(X)
        inspector = ConcreteInspector(model=pca_3, X_train=X)

        # Act
        fig1 = inspector.plot_scores(components=(0, 1))
        fig2 = inspector.plot_scores(components=(1, 2))
        fig3 = inspector.plot_scores(components=(0, 2))

        # Assert
        assert isinstance(fig1, plt.Figure)
        assert isinstance(fig2, plt.Figure)
        assert isinstance(fig3, plt.Figure)
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)


class TestBaseInspectorProperties:
    """Test inspector properties and attributes."""

    def test_n_features_in(self, fitted_pca, dummy_data_loader):
        """Test n_features_in_ attribute."""
        # Arrange
        X, _ = dummy_data_loader
        inspector = ConcreteInspector(model=fitted_pca, X_train=X)

        # Assert
        assert inspector.n_features_in_ == 3

    def test_datasets_structure(self, fitted_pca, dummy_data_loader):
        """Test datasets_ structure."""
        # Arrange
        X, y = dummy_data_loader
        inspector = ConcreteInspector(model=fitted_pca, X_train=X, y_train=y)

        # Assert
        assert isinstance(inspector.datasets_, dict)
        assert "train" in inspector.datasets_
        assert hasattr(inspector.datasets_["train"], "X")
        assert hasattr(inspector.datasets_["train"], "y")
        assert np.array_equal(inspector.datasets_["train"].X, X)
        assert np.array_equal(inspector.datasets_["train"].y, y)

    def test_estimator_attribute(self, fitted_pca, dummy_data_loader):
        """Test estimator_ attribute."""
        # Arrange
        X, _ = dummy_data_loader
        inspector = ConcreteInspector(model=fitted_pca, X_train=X)

        # Assert
        assert inspector.estimator_ is fitted_pca
        assert isinstance(inspector.estimator_, PCA)

    def test_transformer_attribute_none(self, fitted_pca, dummy_data_loader):
        """Test transformer_ attribute when None."""
        # Arrange
        X, _ = dummy_data_loader
        inspector = ConcreteInspector(model=fitted_pca, X_train=X)

        # Assert
        assert inspector.transformer_ is None

    def test_transformer_attribute_pipeline(
        self, fitted_pipeline_pca, dummy_data_loader
    ):
        """Test transformer_ attribute with pipeline."""
        # Arrange
        X, _ = dummy_data_loader
        inspector = ConcreteInspector(model=fitted_pipeline_pca, X_train=X)

        # Assert
        assert inspector.transformer_ is not None
        from sklearn.pipeline import Pipeline

        assert isinstance(inspector.transformer_, Pipeline)


class TestInspectorDataset:
    """Tests for InspectorDataset class."""

    def test_dataset_attributes(self):
        """Test basic dataset attribute access and n_samples property."""
        # Arrange
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        labels = np.array(["a", "b"])
        dataset = InspectorDataset(X=X, y=y, labels=labels)

        # Act & Assert
        # Test direct attribute access
        assert np.array_equal(dataset.X, X)
        assert np.array_equal(dataset.y, y)
        assert np.array_equal(dataset.labels, labels)

        # Test n_samples
        assert dataset.n_samples == 2

    def test_dataset_defaults(self):
        """Test that y and labels default to None."""
        # Arrange
        X = np.array([[1, 2], [3, 4]])
        dataset = InspectorDataset(X=X)

        # Act & Assert
        assert np.array_equal(dataset.X, X)
        assert dataset.y is None
        assert dataset.labels is None

    def test_dataset_is_frozen(self):
        """Test that InspectorDataset is immutable (frozen dataclass)."""
        # Arrange
        X = np.array([[1, 2], [3, 4]])
        dataset = InspectorDataset(X=X)

        # Act & Assert
        with pytest.raises(AttributeError, match="cannot assign"):
            dataset.X = np.array([[5, 6]])


class TestInspectorErrors:
    """Tests for error handling in _BaseInspector initialization."""

    def test_prepare_labels_length_mismatch(self):
        """Test ValueError when sample labels length does not match data length."""
        # Arrange
        rng = np.random.default_rng(42)
        X = rng.random((5, 2))
        model = PCA(n_components=2)
        model.fit(X)

        # Act & Assert
        with pytest.raises(
            ValueError, match="Sample labels for 'train' must have length 5"
        ):
            ConcreteInspector(
                model=model,
                X_train=X,
                sample_labels={"train": ["a", "b"]},  # Length 2 != 5
            )

    def test_resolve_n_components_error(self):
        """Test AttributeError when estimator has no n_components."""

        # Arrange
        class MockEstimator(BaseEstimator):
            def fit(self, X, y=None):
                self.fitted_ = True
                return self

            def transform(self, X):
                return X

            # No n_components attribute

        model = MockEstimator()
        rng = np.random.default_rng(42)
        X = rng.random((5, 2))
        model.fit(X)

        # Patch _validate_and_extract_model to bypass type checks
        with mock.patch(
            "chemotools.inspector.core.base._validate_and_extract_model",
            return_value=(model, None),
        ):
            # Act & Assert
            with pytest.raises(
                AttributeError, match="Cannot determine number of components"
            ):
                ConcreteInspector(
                    model=model,
                    X_train=X,
                )

    def test_get_dataset_errors(self):
        """Test _get_dataset raises appropriate errors."""
        # Arrange
        rng = np.random.default_rng(42)
        X = rng.random((5, 2))
        model = PCA(n_components=2)
        model.fit(X)
        inspector = ConcreteInspector(
            model=model,
            X_train=X,
        )

        # Act & Assert
        # Test missing test data
        with pytest.raises(ValueError, match="Test data not provided"):
            inspector._get_dataset("test")

        # Test missing validation data
        with pytest.raises(ValueError, match="Validation data not provided"):
            inspector._get_dataset("val")

        # Test invalid dataset name
        with pytest.raises(ValueError, match="Invalid dataset 'invalid'"):
            inspector._get_dataset("invalid")

    def test_nan_in_X_train_raises_error(self):
        """Test ValueError when X_train contains NaN values."""
        # Arrange
        rng = np.random.default_rng(42)
        X_valid = rng.random((5, 2))
        model = PCA(n_components=2)
        model.fit(X_valid)

        X_with_nan = rng.random((5, 2))
        X_with_nan[2, 1] = np.nan

        # Act & Assert
        with pytest.raises(ValueError, match="Input X_train contains NaN"):
            ConcreteInspector(
                model=model,
                X_train=X_with_nan,
            )

    def test_inf_in_X_train_raises_error(self):
        """Test ValueError when X_train contains Inf values."""
        # Arrange
        rng = np.random.default_rng(42)
        X_valid = rng.random((5, 2))
        model = PCA(n_components=2)
        model.fit(X_valid)

        X_with_inf = rng.random((5, 2))
        X_with_inf[1, 0] = np.inf

        # Act & Assert
        with pytest.raises(ValueError, match="Input X_train contains infinity"):
            ConcreteInspector(
                model=model,
                X_train=X_with_inf,
            )

    def test_nan_in_X_test_raises_error(self):
        """Test ValueError when X_test contains NaN values."""
        # Arrange
        rng = np.random.default_rng(42)
        X_train = rng.random((5, 2))
        model = PCA(n_components=2)
        model.fit(X_train)

        X_test_with_nan = rng.random((3, 2))
        X_test_with_nan[0, 0] = np.nan

        # Act & Assert
        with pytest.raises(ValueError, match="Input X_test contains NaN"):
            ConcreteInspector(
                model=model,
                X_train=X_train,
                X_test=X_test_with_nan,
            )

    def test_nan_in_y_train_raises_error(self):
        """Test ValueError when y_train contains NaN values."""
        # Arrange
        rng = np.random.default_rng(42)
        X_train = rng.random((5, 2))
        model = PCA(n_components=2)
        model.fit(X_train)

        y_with_nan = rng.random(5)
        y_with_nan[2] = np.nan

        # Act & Assert
        with pytest.raises(ValueError, match="Input target contains NaN"):
            ConcreteInspector(
                model=model,
                X_train=X_train,
                y_train=y_with_nan,
            )


class TestFeatureSelection:
    """Tests for feature selection and mask handling."""

    def test_get_feature_mask_with_pipeline_selector(self):
        """Test get_feature_mask with a pipeline containing a selector."""
        # Arrange
        X = np.random.rand(10, 5)
        y = np.array([0] * 5 + [1] * 5)

        # Create pipeline with selector
        selector = SelectKBest(f_classif, k=2)
        pca = PCA(n_components=2)
        pipeline = Pipeline([("select", selector), ("pca", pca)])
        pipeline.fit(X, y)

        # Act
        inspector = ConcreteInspector(
            model=pipeline, X_train=X, y_train=y, supervised=True
        )
        mask = inspector._get_feature_mask()

        # Assert
        assert mask is not None
        assert mask.sum() == 2

        # Also test _get_preprocessed_feature_names with mask
        feature_names = np.array([f"feat_{i}" for i in range(5)])
        inspector = ConcreteInspector(
            model=pipeline,
            X_train=X,
            y_train=y,
            supervised=True,
            feature_names=feature_names,
        )
        selected_names = inspector._get_preprocessed_feature_names()
        assert len(selected_names) == 2
        assert np.all(np.isin(selected_names, feature_names))

    def test_get_feature_mask_with_selector_model(self):
        """Test get_feature_mask when the transformer is a selector instance."""
        # Arrange
        X = np.random.rand(10, 5)
        y = np.array([0] * 5 + [1] * 5)

        class MockSelector(BaseEstimator, SelectorMixin):
            def fit(self, X, y=None):
                return self

            def transform(self, X):
                return X[:, :2]

            def _get_support_mask(self):
                return np.array([True, True, False, False, False])

        selector = MockSelector()
        pca = PCA(n_components=2)
        pca.fit(X)  # Fit PCA separately as we are mocking the extraction

        # Patch _validate_and_extract_model to return our selector as transformer
        with mock.patch(
            "chemotools.inspector.core.base._validate_and_extract_model",
            return_value=(pca, selector),
        ):
            inspector = ConcreteInspector(
                model=pca, X_train=X, y_train=y, supervised=True
            )
            mask = inspector._get_feature_mask()

            assert mask is not None
            assert np.array_equal(mask, np.array([True, True, False, False, False]))


class TestBaseInspectorMethods:
    """Tests for _BaseInspector methods."""

    def test_get_preprocessing_steps_none(self):
        """Test _get_preprocessing_steps returns empty list when no transformer."""
        # Arrange
        rng = np.random.default_rng(42)
        X = rng.random((5, 2))
        model = PCA(n_components=2)
        model.fit(X)
        inspector = ConcreteInspector(model=model, X_train=X)

        # Act
        steps = inspector._get_preprocessing_steps()

        # Assert
        assert steps == []

    def test_get_preprocessed_feature_names_with_names_no_mask(self):
        """Test _get_preprocessed_feature_names returns feature names when provided."""
        # Arrange
        rng = np.random.default_rng(42)
        X = rng.random((5, 2))
        model = PCA(n_components=2)
        model.fit(X)
        feature_names = ["f1", "f2"]
        inspector = ConcreteInspector(
            model=model, X_train=X, feature_names=feature_names
        )

        # Act
        names = inspector._get_preprocessed_feature_names()

        # Assert
        assert np.array_equal(names, np.array(feature_names))


class TestBaseInspectorFigureManagement:
    """Tests for figure tracking and cleanup functionality."""

    def test_initial_tracked_figures_empty(self, fitted_pca, dummy_data_loader):
        """Test that tracked figures list is empty on initialization."""
        # Arrange
        X, _ = dummy_data_loader

        # Act
        inspector = ConcreteInspector(model=fitted_pca, X_train=X)

        # Assert
        assert inspector._tracked_figures == []

    def test_track_figures_stores_figures(self, fitted_pca, dummy_data_loader):
        """Test that _track_figures stores figure references."""
        # Arrange
        X, _ = dummy_data_loader
        inspector = ConcreteInspector(model=fitted_pca, X_train=X)
        fig1, _ = plt.subplots()
        fig2, _ = plt.subplots()
        figures = {"fig1": fig1, "fig2": fig2}

        # Act
        result = inspector._track_figures(figures)

        # Assert
        assert len(inspector._tracked_figures) == 2
        assert fig1 in inspector._tracked_figures
        assert fig2 in inspector._tracked_figures
        assert result is figures  # Returns the same dict

    def test_close_figures_closes_all_tracked(self, fitted_pca, dummy_data_loader):
        """Test that close_figures closes all tracked figures."""
        # Arrange
        X, _ = dummy_data_loader
        inspector = ConcreteInspector(model=fitted_pca, X_train=X)
        fig1, _ = plt.subplots()
        fig2, _ = plt.subplots()
        inspector._track_figures({"fig1": fig1, "fig2": fig2})

        # Act
        inspector.close_figures()

        # Assert
        assert inspector._tracked_figures == []
        # Verify figures are actually closed (number should be 0)
        assert fig1.number not in plt.get_fignums()
        assert fig2.number not in plt.get_fignums()

    def test_cleanup_previous_figures_clears_old_figures(
        self, fitted_pca, dummy_data_loader
    ):
        """Test that _cleanup_previous_figures closes existing tracked figures."""
        # Arrange
        X, _ = dummy_data_loader
        inspector = ConcreteInspector(model=fitted_pca, X_train=X)
        fig1, _ = plt.subplots()
        inspector._track_figures({"fig1": fig1})
        old_fig_num = fig1.number

        # Act
        inspector._cleanup_previous_figures()

        # Assert
        assert inspector._tracked_figures == []
        assert old_fig_num not in plt.get_fignums()

    def test_multiple_track_calls_accumulate(self, fitted_pca, dummy_data_loader):
        """Test that multiple _track_figures calls accumulate figures."""
        # Arrange
        X, _ = dummy_data_loader
        inspector = ConcreteInspector(model=fitted_pca, X_train=X)
        fig1, _ = plt.subplots()
        fig2, _ = plt.subplots()

        # Act
        inspector._track_figures({"fig1": fig1})
        inspector._track_figures({"fig2": fig2})

        # Assert
        assert len(inspector._tracked_figures) == 2

    def test_close_figures_is_idempotent(self, fitted_pca, dummy_data_loader):
        """Test that calling close_figures multiple times is safe."""
        # Arrange
        X, _ = dummy_data_loader
        inspector = ConcreteInspector(model=fitted_pca, X_train=X)
        fig, _ = plt.subplots()
        inspector._track_figures({"fig": fig})

        # Act & Assert - should not raise
        inspector.close_figures()
        inspector.close_figures()
        assert inspector._tracked_figures == []

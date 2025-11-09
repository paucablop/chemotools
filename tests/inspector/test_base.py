import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from chemotools.inspector._base import _BaseInspector


# Concrete implementation for testing the abstract base class
class ConcreteInspector(_BaseInspector):
    """Concrete implementation of _BaseInspector for testing."""

    def plot_scores(self, components=(0, 1), datasets=["train"], **kwargs):
        """Simple implementation for testing."""
        fig, ax = plt.subplots()
        for dataset in datasets:
            if dataset in self.datasets_:
                scores = self._get_scores(dataset)
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
        assert np.array_equal(inspector.datasets_["train"]["X"], X)

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
        assert np.array_equal(inspector.datasets_["train"]["X"], X)
        assert np.array_equal(inspector.datasets_["train"]["y"], y)

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
        assert np.array_equal(inspector.datasets_["train"]["X"], X_train)
        assert np.array_equal(inspector.datasets_["test"]["X"], X_test)

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
        assert np.array_equal(inspector.datasets_["train"]["X"], X_train)
        assert np.array_equal(inspector.datasets_["val"]["X"], X_val)

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
        assert np.array_equal(inspector.datasets_["train"]["X"], X_train)
        assert np.array_equal(inspector.datasets_["test"]["X"], X_test)
        assert np.array_equal(inspector.datasets_["val"]["X"], X_val)

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

        # Assert
        assert "train" in inspector.sample_labels
        assert len(inspector.sample_labels["train"]) == 100

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


class TestBaseInspectorGetScores:
    """Test _get_scores method."""

    def test_get_scores_pca(self, fitted_pca, dummy_data_loader):
        """Test getting scores from PCA model."""
        # Arrange
        X, _ = dummy_data_loader
        inspector = ConcreteInspector(model=fitted_pca, X_train=X)

        # Act
        scores = inspector._get_scores(X)

        # Assert
        assert scores.shape == (100, 2)
        # Verify it matches direct transform
        expected_scores = fitted_pca.transform(X)
        assert np.allclose(scores, expected_scores)

    def test_get_scores_pls(self, fitted_pls, dummy_data_loader):
        """Test getting scores from PLS model."""
        # Arrange
        X, y = dummy_data_loader
        inspector = ConcreteInspector(model=fitted_pls, X_train=X, y_train=y)

        # Act
        scores = inspector._get_scores(X)

        # Assert
        assert scores.shape == (100, 2)
        # Verify it matches direct transform
        expected_scores = fitted_pls.transform(X)
        assert np.allclose(scores, expected_scores)

    def test_get_scores_with_pipeline(self, fitted_pipeline_pca, dummy_data_loader):
        """Test getting scores with preprocessing pipeline."""
        # Arrange
        X, _ = dummy_data_loader
        inspector = ConcreteInspector(model=fitted_pipeline_pca, X_train=X)

        # Act
        scores = inspector._get_scores(X)

        # Assert
        assert scores.shape == (100, 2)
        # Verify it matches direct transform
        expected_scores = fitted_pipeline_pca.transform(X)
        assert np.allclose(scores, expected_scores)

    def test_get_scores_different_dataset(self, fitted_pca, dummy_data_loader):
        """Test getting scores for different dataset split."""
        # Arrange
        X, _ = dummy_data_loader
        X_train, X_test = X[:80], X[80:]
        inspector = ConcreteInspector(model=fitted_pca, X_train=X_train, X_test=X_test)

        # Act
        train_scores = inspector._get_scores(X_train, "train")
        test_scores = inspector._get_scores(X_test, "test")

        # Assert
        assert train_scores.shape == (80, 2)
        assert test_scores.shape == (20, 2)


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
        assert "X" in inspector.datasets_["train"]
        assert "y" in inspector.datasets_["train"]

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

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.exceptions import NotFittedError

from chemotools.inspector.core.validation import (
    _validate_and_extract_model,
    _validate_datasets_consistency,
)


class TestValidateAndExtractModel:
    """Test _validate_and_extract_model function."""

    def test_validate_fitted_pca(self, fitted_pca):
        """Test validation with a fitted PCA model."""
        # Arrange
        # (fitted_pca fixture provides the fitted model)

        # Act
        estimator, transformer = _validate_and_extract_model(fitted_pca)

        # Assert
        assert estimator is fitted_pca
        assert transformer is None

    def test_validate_fitted_pls(self, fitted_pls):
        """Test validation with a fitted PLS model."""
        # Arrange
        # (fitted_pls fixture provides the fitted model)

        # Act
        estimator, transformer = _validate_and_extract_model(fitted_pls)

        # Assert
        assert estimator is fitted_pls
        assert transformer is None

    def test_validate_fitted_pipeline_pca(self, fitted_pipeline_pca):
        """Test validation with a fitted pipeline containing PCA."""
        # Arrange
        # (fitted_pipeline_pca fixture provides the fitted pipeline)

        # Act
        estimator, transformer = _validate_and_extract_model(fitted_pipeline_pca)

        # Assert
        assert isinstance(estimator, PCA)
        assert transformer is not None
        assert len(transformer.steps) == 1
        assert isinstance(transformer.steps[0][1], StandardScaler)

    def test_unfitted_pca_raises_error(self, unfitted_pca):
        """Test that unfitted PCA raises NotFittedError."""
        # Arrange
        # (unfitted_pca fixture provides the unfitted model)

        # Act & Assert
        with pytest.raises(NotFittedError):
            _validate_and_extract_model(unfitted_pca)

    def test_unfitted_pls_raises_error(self, unfitted_pls):
        """Test that unfitted PLS raises NotFittedError."""
        # Arrange
        # (unfitted_pls fixture provides the unfitted model)

        # Act & Assert
        with pytest.raises(NotFittedError):
            _validate_and_extract_model(unfitted_pls)

    def test_unfitted_pipeline_raises_error(self, unfitted_pipeline):
        """Test that unfitted pipeline raises NotFittedError."""
        # Arrange
        # (unfitted_pipeline fixture provides the unfitted pipeline)

        # Act & Assert
        with pytest.raises(NotFittedError):
            _validate_and_extract_model(unfitted_pipeline)

    def test_invalid_model_type_raises_error(self, fitted_invalid_model):
        """Test that invalid model type raises TypeError."""
        # Arrange
        # (fitted_invalid_model fixture provides a model of wrong type)

        # Act & Assert
        with pytest.raises(TypeError, match="Model must be _BasePCA, _PLS"):
            _validate_and_extract_model(fitted_invalid_model)

    def test_pipeline_with_invalid_final_step(self, dummy_data_loader):
        """Test that pipeline with invalid final step raises TypeError."""
        # Arrange
        X, _ = dummy_data_loader
        pipeline = make_pipeline(StandardScaler(), SVR()).fit(X, np.ones(len(X)))

        # Act & Assert
        with pytest.raises(TypeError, match="Model must be _BasePCA, _PLS"):
            _validate_and_extract_model(pipeline)

    def test_single_step_pipeline(self, dummy_data_loader):
        """Test pipeline with only one step (no preprocessing)."""
        # Arrange
        X, _ = dummy_data_loader
        pipeline = make_pipeline(PCA(n_components=2)).fit(X)

        # Act
        estimator, transformer = _validate_and_extract_model(pipeline)

        # Assert
        assert isinstance(estimator, PCA)
        assert transformer is None

    def test_multi_step_pipeline(self, dummy_data_loader):
        """Test pipeline with multiple preprocessing steps."""
        # Arrange
        X, _ = dummy_data_loader
        pipeline = make_pipeline(
            StandardScaler(), StandardScaler(), PCA(n_components=2)
        ).fit(X)

        # Act
        estimator, transformer = _validate_and_extract_model(pipeline)

        # Assert
        assert isinstance(estimator, PCA)
        assert transformer is not None
        assert len(transformer.steps) == 2


class TestValidateDatasetsConsistency:
    """Test _validate_datasets_consistency function."""

    def test_valid_train_only_unsupervised(self):
        """Test valid training data only for unsupervised learning."""
        # Arrange
        X_train = np.random.rand(100, 10)

        # Act
        _validate_datasets_consistency(
            X_train=X_train,
            y_train=None,
            X_test=None,
            y_test=None,
            X_val=None,
            y_val=None,
            supervised=False,
        )

        # Assert
        # (no exception raised means validation passed)

    def test_valid_train_test_unsupervised(self):
        """Test valid training and test data for unsupervised learning."""
        # Arrange
        X_train = np.random.rand(100, 10)
        X_test = np.random.rand(50, 10)

        # Act
        _validate_datasets_consistency(
            X_train=X_train,
            y_train=None,
            X_test=X_test,
            y_test=None,
            X_val=None,
            y_val=None,
            supervised=False,
        )

        # Assert
        # (no exception raised means validation passed)

    def test_valid_all_datasets_unsupervised(self):
        """Test valid training, test, and validation data for unsupervised."""
        # Arrange
        X_train = np.random.rand(100, 10)
        X_test = np.random.rand(50, 10)
        X_val = np.random.rand(30, 10)

        # Act
        _validate_datasets_consistency(
            X_train=X_train,
            y_train=None,
            X_test=X_test,
            y_test=None,
            X_val=X_val,
            y_val=None,
            supervised=False,
        )

        # Assert
        # (no exception raised means validation passed)

    def test_valid_train_only_supervised(self):
        """Test valid training data only for supervised learning."""
        # Arrange
        X_train = np.random.rand(100, 10)
        y_train = np.random.rand(100)

        # Act
        _validate_datasets_consistency(
            X_train=X_train,
            y_train=y_train,
            X_test=None,
            y_test=None,
            X_val=None,
            y_val=None,
            supervised=True,
        )

        # Assert
        # (no exception raised means validation passed)

    def test_valid_train_test_supervised(self):
        """Test valid training and test data for supervised learning."""
        # Arrange
        X_train = np.random.rand(100, 10)
        y_train = np.random.rand(100)
        X_test = np.random.rand(50, 10)
        y_test = np.random.rand(50)

        # Act
        _validate_datasets_consistency(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            X_val=None,
            y_val=None,
            supervised=True,
        )

        # Assert
        # (no exception raised means validation passed)

    def test_inconsistent_features_in_test(self):
        """Test that inconsistent features in test set raises ValueError."""
        # Arrange
        X_train = np.random.rand(100, 10)
        X_test = np.random.rand(50, 15)  # Wrong number of features

        # Act & Assert
        with pytest.raises(
            ValueError, match="X_test must have same number of features as X_train"
        ):
            _validate_datasets_consistency(
                X_train=X_train,
                y_train=None,
                X_test=X_test,
                y_test=None,
                X_val=None,
                y_val=None,
                supervised=False,
            )

    def test_inconsistent_features_in_val(self):
        """Test that inconsistent features in validation set raises ValueError."""
        # Arrange
        X_train = np.random.rand(100, 10)
        X_val = np.random.rand(30, 8)  # Wrong number of features

        # Act & Assert
        with pytest.raises(
            ValueError, match="X_val must have same number of features as X_train"
        ):
            _validate_datasets_consistency(
                X_train=X_train,
                y_train=None,
                X_test=None,
                y_test=None,
                X_val=X_val,
                y_val=None,
                supervised=False,
            )

    def test_supervised_missing_y_train(self):
        """Test that supervised learning without y_train raises ValueError."""
        # Arrange
        X_train = np.random.rand(100, 10)

        # Act & Assert
        with pytest.raises(ValueError, match="y_train required for supervised models"):
            _validate_datasets_consistency(
                X_train=X_train,
                y_train=None,
                X_test=None,
                y_test=None,
                X_val=None,
                y_val=None,
                supervised=True,
            )

    def test_supervised_missing_y_test(self):
        """Test that supervised learning with X_test but no y_test raises error."""
        # Arrange
        X_train = np.random.rand(100, 10)
        y_train = np.random.rand(100)
        X_test = np.random.rand(50, 10)

        # Act & Assert
        with pytest.raises(ValueError, match="y_test required when X_test is provided"):
            _validate_datasets_consistency(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=None,
                X_val=None,
                y_val=None,
                supervised=True,
            )

    def test_supervised_missing_y_val(self):
        """Test that supervised learning with X_val but no y_val raises error."""
        # Arrange
        X_train = np.random.rand(100, 10)
        y_train = np.random.rand(100)
        X_val = np.random.rand(30, 10)

        # Act & Assert
        with pytest.raises(ValueError):
            _validate_datasets_consistency(
                X_train=X_train,
                y_train=y_train,
                X_test=None,
                y_test=None,
                X_val=X_val,
                y_val=None,
                supervised=True,
            )

    def test_train_y_length_mismatch_raises(self):
        """Train y length must match X_train regardless of supervised flag."""
        # Arrange
        X_train = np.random.rand(20, 5)
        y_train = np.random.rand(19)

        # Act & Assert
        with pytest.raises(ValueError, match="same number of samples"):
            _validate_datasets_consistency(
                X_train=X_train,
                y_train=y_train,
                X_test=None,
                y_test=None,
                X_val=None,
                y_val=None,
                supervised=False,
            )

    def test_test_y_length_mismatch_raises(self):
        """Test y length must match X_test when provided."""
        # Arrange
        X_train = np.random.rand(30, 4)
        y_train = np.random.rand(30)
        X_test = np.random.rand(15, 4)
        y_test = np.random.rand(14)

        # Act & Assert
        with pytest.raises(ValueError, match="same number of samples"):
            _validate_datasets_consistency(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                X_val=None,
                y_val=None,
                supervised=True,
            )

    def test_val_y_length_mismatch_raises(self):
        """Validation y length must match X_val when provided."""
        # Arrange
        X_train = np.random.rand(30, 4)
        y_train = np.random.rand(30)
        X_val = np.random.rand(10, 4)
        y_val = np.random.rand(9)

        # Act & Assert
        with pytest.raises(ValueError, match="same number of samples"):
            _validate_datasets_consistency(
                X_train=X_train,
                y_train=y_train,
                X_test=None,
                y_test=None,
                X_val=X_val,
                y_val=y_val,
                supervised=True,
            )

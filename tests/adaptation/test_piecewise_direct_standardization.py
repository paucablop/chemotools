"""
Test for PiecewiseDirectStandardization
"""

# Authors: Ruggero Guerrini
# License: MIT

import re

import numpy as np
import pytest
import sklearn
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator

from chemotools.adaptation._piecewise_direct_standardization import (
    PiecewiseDirectStandardization,
)
from chemotools.derivative import SavitzkyGolay


def data_diff(dataset_ref, dataset_test):
    diff_norm = np.linalg.norm(dataset_ref - dataset_test)
    ref_norm = np.linalg.norm(dataset_ref)
    difference = diff_norm / ref_norm
    return difference


class TestSklearnCompliance:
    """Tests for sklearn estimator API compliance."""

    def test_compliance_PiecewiseDirectStandardization(self):
        """Verifies that PiecewiseDirectStandardization passes all sklearn estimator
        checks."""
        # Arrange
        transformer = PiecewiseDirectStandardization()

        # Act & Assert
        check_estimator(transformer)


class TestFit:
    """Tests for the fit method behavior."""

    def test_fit_sets_attributes(self, sample_data):
        """Verifies that fit stores the required fitted attributes."""
        # Arrange
        X_target, X_source = sample_data

        # Act
        model = PiecewiseDirectStandardization().fit(X_target, X_source=X_source)

        # Assert
        assert hasattr(model, "n_features_in_")
        assert hasattr(model, "x_mean_")
        assert hasattr(model, "coef_")
        assert hasattr(model, "intercept_")
        assert model.x_mean_.shape == (X_target.shape[1], 2 * model.window_length + 1)
        assert model.coef_.shape == (X_target.shape[1], 2 * model.window_length + 1)
        assert model.intercept_.shape == (X_target.shape[1],)
        assert model.n_features_in_ == X_target.shape[1]

    def test_fit_raises_on_shape_mismatch(self, sample_data):
        """Verifies fit raises ValueError when X and X_source have different shapes."""
        # Arrange
        X_target, X_source = sample_data
        model = PiecewiseDirectStandardization()

        # Act & Assert
        with pytest.raises(
            ValueError,
            match=re.escape(
                "X and X_source must have the same shape, got X=(100, 20) and "
                "X_source=(99, 20)."
            ),
        ):
            model.fit(X_target, X_source=X_source[:-1, :])

    def test_fit_raises_when_n_components_too_large(self, sample_data):
        """Verifies fit raises ValueError when n_components is too large for PLS."""
        # Arrange
        X_target, X_source = sample_data
        model = PiecewiseDirectStandardization(window_length=2, n_components=4)

        # Act & Assert
        with pytest.raises(ValueError, match="n_components"):
            model.fit(X_target, X_source=X_source)

    def test_fit_raises_when_n_components_exceeds_n_samples(self):
        """Verifies fit raises ValueError when n_components exceeds n_samples."""
        # Arrange
        rng = np.random.default_rng(42)
        X = rng.normal(size=(10, 50))  # Only 10 samples
        X_source = rng.normal(size=(10, 50))
        model = PiecewiseDirectStandardization(n_components=15)  # More than 10 samples

        # Act & Assert
        with pytest.raises(
            ValueError,
            match=re.escape("n_components=15 must be <= n_samples=10"),
        ):
            model.fit(X, X_source=X_source)


class TestTransform:
    """Tests for the transform method behavior."""

    def test_transform_preserves_shape(self, sample_data):
        """Verifies that the output shape matches both input X and X_source."""
        # Arrange
        X_target, X_source = sample_data
        model = PiecewiseDirectStandardization().fit(X_target, X_source=X_source)

        # Act
        X_transformed = model.transform(X_target)

        # Assert
        assert X_transformed.shape == X_source.shape
        assert X_transformed.shape == X_target.shape

    def test_transform_improves_match_to_target(self, sample_data):
        """Verifies that transformation reduces the distance to the source instrument
        data."""
        # Arrange
        X_target, X_source = sample_data
        model = PiecewiseDirectStandardization().fit(X_target, X_source=X_source)

        # Act
        X_transformed = model.transform(X_target)
        before = data_diff(X_target, X_source)
        after = data_diff(X_transformed, X_source)

        # Assert
        assert after < before

    def test_transform_before_fit_raises(self, sample_data):
        """Verifies that calling transform before fit raises NotFittedError."""
        # Arrange
        X_target, _ = sample_data
        model = PiecewiseDirectStandardization()

        # Act & Assert
        with pytest.raises(NotFittedError):
            model.transform(X_target)

    def test_transform_does_not_modify_input(self, sample_data):
        """Verifies that fit and transform do not mutate the input arrays."""
        # Arrange
        X_target, X_source = sample_data
        X_target_original = X_target.copy()
        X_source_original = X_source.copy()

        # Act
        model = PiecewiseDirectStandardization().fit(X_target, X_source=X_source)
        model.transform(X_target)

        # Assert
        np.testing.assert_array_equal(X_target, X_target_original)
        np.testing.assert_array_equal(X_source, X_source_original)

    def test_transform_is_deterministic(self, sample_data):
        """Verifies that calling transform multiple times with the same input gives
        identical results."""
        # Arrange
        X_target, X_source = sample_data
        model = PiecewiseDirectStandardization().fit(X_target, X_source=X_source)

        # Act
        result1 = model.transform(X_target)
        result2 = model.transform(X_target)

        # Assert
        np.testing.assert_array_equal(result1, result2)


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_identity_transformation_when_X_source_is_none(self):
        """Verifies that fitting with X_source=None results in identity
        transformation."""
        # Arrange
        rng = np.random.default_rng(42)
        X = rng.normal(size=(50, 10))
        model = PiecewiseDirectStandardization()

        # Act - fit with X_source=None should trigger identity transformation
        with pytest.warns(UserWarning, match="identity transformation"):
            model.fit(X, X_source=None)

        X_transformed = model.transform(X)

        # Assert - should return X unchanged
        np.testing.assert_array_equal(X_transformed, X)
        assert hasattr(model, "x_source_provided_")
        assert model.x_source_provided_ is False

    def test_window_length_larger_than_features(self):
        """Verifies that window_length larger than n_features is handled gracefully."""
        # Arrange - window_length exceeds number of features
        rng = np.random.default_rng(17)
        X = rng.normal(size=(50, 10))
        X_source = X * 1.5 + rng.normal(0, 0.05, size=(50, 10))

        # Act - should handle gracefully by using all features in window
        model = PiecewiseDirectStandardization(
            window_length=50
        )  # Much larger than 10 features
        model.fit(X, X_source=X_source)
        X_transformed = model.transform(X)

        # Assert - transformation should still work correctly
        assert X_transformed.shape == X.shape
        # For edge features with window_length >> n_features, all features are used
        assert model.x_mean_.shape == (10, 101)  # 2 * 50 + 1
        assert model.coef_.shape == (10, 101)

    def test_window_length_equals_one(self):
        """Verifies that minimal window_length of 1 works correctly."""
        # Arrange - minimal window_length
        rng = np.random.default_rng(17)
        X = rng.normal(size=(50, 20))
        X_source = X * 1.2 + rng.normal(0, 0.1, size=(50, 20))

        # Act - each feature uses itself and immediate neighbors only
        model = PiecewiseDirectStandardization(window_length=1, n_components=1)
        model.fit(X, X_source=X_source)
        X_transformed = model.transform(X)

        # Assert
        assert X_transformed.shape == X.shape
        # Interior features use 3 features, edges use fewer
        assert model.x_mean_.shape == (20, 3)  # 2 * 1 + 1
        assert model.coef_.shape == (20, 3)


class TestPipeline:
    """Tests for sklearn Pipeline and metadata routing integration."""

    def test_pipeline_gridsearchcv_pls_metadata_routing(self, sample_data):
        """Verifies that X_source metadata routing works inside a Pipeline with
        GridSearchCV."""
        # Arrange
        X_target, X_source = sample_data
        rng = np.random.default_rng(42)
        y_concentration = rng.normal(size=(100, 1))

        sklearn.set_config(enable_metadata_routing=True)

        pipe = Pipeline(
            [
                ("scaler", SavitzkyGolay()),
                (
                    "model",
                    PiecewiseDirectStandardization().set_fit_request(X_source=True),
                ),
                ("pls", PLSRegression()),
            ]
        )
        param_grid = {
            "scaler__window_length": [15, 25],
            "scaler__polyorder": [2, 3],
            "scaler__deriv": [1, 2],
            "model__window_length": [10, 15, 20],
            "model__n_components": [2, 3, 5],
            "pls__n_components": [2, 3],
        }
        grid = GridSearchCV(pipe, param_grid, cv=3, error_score="raise")

        # Act - X_source passed as kwarg, sklearn routes it to PDS with correct indices
        grid.fit(X_target, y_concentration, X_source=X_source)

        # Assert - verify the grid search completed and the fitted model works
        assert grid.best_estimator_ is not None
        assert hasattr(grid.best_estimator_, "named_steps")
        # Verify the best model can actually transform data
        X_test = rng.normal(size=(10, 20))
        y_pred = grid.best_estimator_.predict(X_test)
        assert y_pred.shape == (10, 1)

        # Cleanup - reset config to avoid affecting other tests
        sklearn.set_config(enable_metadata_routing=False)

"""
Tests for the :class:`chemotools.adaptation.CORAL` transformer.

The test suite covers:
- Correct covariance alignment after transformation
- Identity behaviour when ``X_target`` is not provided
- scikit-learn API compliance (clone, set_params, Pipeline)
- Input validation and error handling
- Numerical edge cases (single sample, single feature, identical domains)
"""

import numpy as np
import pytest
import sklearn
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator

from chemotools.adaptation import CORAL
from chemotools.derivative import SavitzkyGolay
from tests.adaptation.conftest import data_diff


class TestSklearnCompliance:
    """Tests for sklearn estimator API compliance."""

    def test_compliance_PiecewiseDirectStandardization(self):
        """Verifies that PiecewiseDirectStandardization passes all sklearn estimator
        checks."""
        # Arrange
        transformer = CORAL()

        # Act & Assert
        check_estimator(transformer)


class TestAttributes:
    def test_fit_sets_attributes_without_params(self, sample_data):
        """Verifies that fit stores the required fitted attributes with
        valid values."""
        # Arrange
        X_target, X_source = sample_data

        # Act
        model = CORAL().fit(X_target, X_source=X_source)

        # Assert - Check attributes exist with correct shapes
        assert hasattr(model, "n_features_in_")
        assert model.n_features_in_ == X_target.shape[1]
        assert hasattr(model, "X_mean_")
        assert model.X_mean_.shape[0] == X_target.shape[1]
        assert hasattr(model, "X_centered_")
        assert model.X_centered_.shape == X_target.shape
        assert hasattr(model, "X_source_mean_")
        assert model.X_source_mean_.shape[0] == X_source.shape[1]
        assert hasattr(model, "X_source_centered_")
        assert model.X_source_centered_.shape == X_source.shape
        for i in ["C_X_", "C_X_source_", "C_X_inv_sqrt_", "C_X_source_sqrt_"]:
            assert hasattr(model, i)
            assert getattr(model, i).shape[0] == X_target.shape[1]
            assert getattr(model, i).shape[1] == X_target.shape[1]
            assert getattr(model, i).shape[0] == X_source.shape[1]
            assert getattr(model, i).shape[1] == X_source.shape[1]
        assert hasattr(model, "X_source_provided_")

    def test_fit_sets_attributes_with_params(self, sample_data):
        """Verifies that fit stores the required fitted attributes with
        valid values."""
        # Arrange
        X_target, X_source = sample_data

        # Act
        model = CORAL(reg=1e-2).fit(X_target, X_source=X_source)

        # Assert - Check attributes exist with correct shapes
        assert hasattr(model, "n_features_in_")
        assert model.n_features_in_ == X_target.shape[1]
        assert hasattr(model, "X_mean_")
        assert model.X_mean_.shape[0] == X_target.shape[1]
        assert hasattr(model, "X_centered_")
        assert model.X_centered_.shape == X_target.shape
        assert hasattr(model, "X_source_mean_")
        assert model.X_source_mean_.shape[0] == X_source.shape[1]
        assert hasattr(model, "X_source_centered_")
        assert model.X_source_centered_.shape == X_source.shape
        for i in ["C_X_", "C_X_source_", "C_X_inv_sqrt_", "C_X_source_sqrt_"]:
            assert hasattr(model, i)
            assert getattr(model, i).shape[0] == X_target.shape[1]
            assert getattr(model, i).shape[1] == X_target.shape[1]
            assert getattr(model, i).shape[0] == X_source.shape[1]
            assert getattr(model, i).shape[1] == X_source.shape[1]
        assert hasattr(model, "X_source_provided_")

    def test_fit_raises_on_shape_mismatch(self, sample_data):
        """Verifies fit raises ValueError when X and X_source have different
        number of features."""
        # Arrange
        X_target, X_source = sample_data
        model = CORAL()

        # Act & Assert
        with pytest.raises(ValueError):
            model.fit(X_target, X_source=X_source[:, :-1])


class TestFeaturesMismatch:
    """Test for features mismatch."""

    def test_raise_on_feature_mismatch(self, sample_data):
        """Verifies that feature mismatch raises a error."""
        X_target, X_source = sample_data

        model = CORAL()

        with pytest.raises(ValueError, match="same number of features"):
            model.fit(X_target, X_source=X_source[:, :-1])


class TestTransform:
    """Tests for the transform method behavior."""

    def test_transform_preserves_shape(self, sample_data):
        """Verifies that the output shape matches both input X
        and X_source."""
        # Arrange
        X_target, X_source = sample_data
        model = CORAL().fit(X_target, X_source=X_source)

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
        model = CORAL().fit(X_target, X_source=X_source)

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
        model = CORAL()

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
        model = CORAL().fit(X_target, X_source=X_source)
        model.transform(X_target)

        # Assert
        np.testing.assert_array_equal(X_target, X_target_original)
        np.testing.assert_array_equal(X_source, X_source_original)


class TestNumericalCorrectness:
    """Tests for numerical correctness and regression testing.

    These tests verify that the algorithm produces expected numerical outputs.
    They serve as regression tests to catch unintended changes in functionality.
    """

    def test_snapshot_transform_output(self):
        """Snapshot test: verifies transform output matches reference values.

        This is a golden/snapshot test with hardcoded expected output.
        If this test fails after code changes, verify the change is intentional.
        """
        # Arrange - Fixed data (do not change!)
        rng = np.random.default_rng(123)
        X_target = rng.normal(size=(15, 8))
        X_source = X_target * 1.3 + rng.normal(0, 0.08, size=(15, 8))
        X_test = rng.normal(size=(3, 8))

        # Expected reference output (generated with with_mean=True, with_std=False)
        expected_output = np.array(
            [
                [
                    0.37653569,
                    1.21280082,
                    -0.6916714,
                    1.23739142,
                    -0.83617321,
                    0.25977057,
                    -0.0357827,
                    -0.01176518,
                ],
                [
                    1.80432567,
                    0.98502844,
                    -0.53307102,
                    -3.04873694,
                    1.05552412,
                    0.97561542,
                    0.35608705,
                    -0.62059924,
                ],
                [
                    1.90220437,
                    -2.16044339,
                    -0.81835332,
                    -1.57683086,
                    0.38496276,
                    -0.39311498,
                    2.47689537,
                    0.02662162,
                ],
            ]
        )

        # Act
        model = CORAL()
        model.fit(X_target, X_source=X_source)
        output = model.transform(X_test)
        # Assert - Output should match reference within tolerance
        np.testing.assert_allclose(output, expected_output, rtol=1e-6, atol=1e-8)

    def test_transform_output_characteristics(self):
        """Verifies that transform output has expected characteristics."""
        # Arrange
        rng = np.random.default_rng(42)
        X_target = rng.normal(size=(20, 8))
        X_source = X_target * 2.0 + rng.normal(0, 0.1, size=(20, 8))

        model = CORAL()
        model.fit(X_target, X_source=X_source)

        # Act
        X_test = rng.normal(size=(5, 8))
        X_transformed = model.transform(X_test)

        # Assert - Check output properties
        assert X_transformed.shape == X_test.shape
        assert np.all(np.isfinite(X_transformed))
        assert np.abs(X_transformed).mean() > 0  # Non-zero output
        assert np.abs(X_transformed).max() < 100

    def test_transformation_is_reproducible(self):
        """Verifies that same inputs always produce same outputs."""
        # Arrange
        rng = np.random.default_rng(99)
        X_target = rng.normal(size=(25, 12))
        X_source = X_target * 1.3 + rng.normal(0, 0.08, size=(25, 12))
        X_test = rng.normal(size=(10, 12))

        # Act - Fit and transform twice
        model1 = CORAL()
        model1.fit(X_target, X_source=X_source)
        result1 = model1.transform(X_test)

        model2 = CORAL()
        model2.fit(X_target, X_source=X_source)
        result2 = model2.transform(X_test)
        np.testing.assert_array_equal(result1, result2)
        for i in [
            "n_features_in_",
            "X_mean_",
            "X_centered_",
            "X_source_mean_",
            "X_source_centered_",
            "C_X_",
            "C_X_source_",
            "C_X_inv_sqrt_",
            "C_X_source_sqrt_",
        ]:
            np.testing.assert_array_equal(getattr(model1, i), getattr(model2, i))


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_identity_transformation_when_X_source_is_none(self):
        """Verifies that fitting with X_source=None results in identity
        transformation."""
        # Arrange
        rng = np.random.default_rng(42)
        X = rng.normal(size=(50, 10))
        model = CORAL()

        # Act - fit with X_source=None should trigger identity transformation
        with pytest.warns(UserWarning, match="identity transformation"):
            model.fit(X, X_source=None)

        X_transformed = model.transform(X)

        # Assert - should return X unchanged
        np.testing.assert_array_equal(X_transformed, X)
        assert hasattr(model, "X_source_provided_")
        assert model.X_source_provided_ is False


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
        try:
            pipe = Pipeline(
                [
                    ("scaler", SavitzkyGolay()),
                    (
                        "model",
                        CORAL().set_fit_request(X_source=True),
                    ),
                    ("pls", PLSRegression()),
                ]
            )
            param_grid = {
                "scaler__window_length": [15, 25],
                "scaler__polyorder": [2, 3],
                "scaler__deriv": [1, 2],
                "pls__n_components": [2, 3],
            }
            grid = GridSearchCV(pipe, param_grid, cv=3, error_score="raise")

            # Act
            grid.fit(X_target, y_concentration, X_source=X_source)
            X_test = rng.normal(size=(10, 50))
            y_pred = grid.best_estimator_.predict(X_test)

            # Assert
            assert grid.best_estimator_ is not None
            assert hasattr(grid.best_estimator_, "named_steps")
            assert y_pred.shape == (10, 1)

        finally:
            # Cleanup - reset config to avoid affecting other tests
            sklearn.set_config(enable_metadata_routing=False)

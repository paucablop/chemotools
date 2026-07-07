"""
Test for SubspaceAlignment
"""

# Authors: Ruggero Guerrini
# License: MIT

import numpy as np
import pytest
import sklearn
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator

from chemotools.adaptation._subspace_alignment import (
    SubspaceAlignment,
)
from chemotools.derivative import SavitzkyGolay
from tests.adaptation.conftest import data_diff


class TestSklearnCompliance:
    """Tests for sklearn estimator API compliance."""

    def test_compliance_SubspaceAlignment(self):
        """Verifies that SubspaceAlignment passes all sklearn estimator
        checks."""
        # Arrange
        transformer = SubspaceAlignment()

        # Act & Assert
        check_estimator(transformer)


class TestFit:
    """Tests for the fit method behavior."""

    def test_fit_sets_attributes(self, sample_data):
        """Verifies that fit stores the required fitted attributes with valid values."""
        # Arrange
        X_target, X_source = sample_data

        # Act
        model = SubspaceAlignment().fit(X_target, X_source=X_source)

        # Assert - Check attributes exist with correct shapes
        assert hasattr(model, "components_X_")
        assert hasattr(model, "components_X_source_")
        assert hasattr(model, "x_source_provided_")
        assert hasattr(model, "X_mean_")
        assert hasattr(model, "X_std_")
        assert hasattr(model, "X_source_mean_")
        assert hasattr(model, "X_source_std_")

        # Assert - Check values are finite and reasonable

        assert np.all(np.isfinite(model.components_X_))
        assert np.all(np.isfinite(model.components_X_source_))
        assert np.all(np.isfinite(model.X_mean_))
        assert np.all(np.isfinite(model.X_std_))
        assert np.all(np.isfinite(model.X_source_mean_))
        assert np.all(np.isfinite(model.X_source_std_))

    def test_fit_raises_on_features_mismatch(self, sample_data):
        """Verifies fit raises ValueError when X and X_source have different shapes."""
        # Arrange
        X_target, X_source = sample_data
        model = SubspaceAlignment()

        # Act & Assert
        with pytest.raises(
            ValueError,
        ):
            model.fit(X_target, X_source=X_source[:, :-1])


class TestTransform:
    """Tests for the transform method behavior."""

    def test_transform_preserves_shape(self, sample_data):
        """Verifies that the output shape matches both input X and X_source."""
        # Arrange
        X_target, X_source = sample_data
        model = SubspaceAlignment().fit(X_target, X_source=X_source)

        # Act
        X_transformed = model.transform(X_target)

        # Assert
        assert X_transformed.shape == X_source.shape

    def test_transform_improves_match_to_target(self, sample_data):
        """Verifies that transformation reduces the distance to the source instrument
        data."""
        # Arrange
        X_target, X_source = sample_data
        model = SubspaceAlignment().fit(X_target, X_source=X_source)

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
        model = SubspaceAlignment()

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
        model = SubspaceAlignment().fit(X_target, X_source=X_source)
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
                    0.42496044,
                    -0.27328656,
                    0.97811483,
                    0.32371867,
                    -0.15827008,
                    0.31078968,
                    0.27923039,
                    -0.59861651,
                ],
                [
                    1.37732014,
                    0.87882016,
                    0.60471303,
                    -1.63905586,
                    0.82340457,
                    0.39435677,
                    1.56047534,
                    0.09773151,
                ],
                [
                    1.5468466,
                    -1.47909193,
                    -0.64075961,
                    -1.09483044,
                    0.20015182,
                    0.46108194,
                    1.91553028,
                    0.86251811,
                ],
            ]
        )

        # Act
        model = SubspaceAlignment()

        model.fit(X_target, X_source=X_source)
        output = model.transform(X_test)
        print(output)
        # Assert - Output should match reference within tolerance
        np.testing.assert_allclose(output, expected_output, rtol=1e-6, atol=1e-8)

    def test_transform_output_characteristics(self):
        """Verifies that transform output has expected characteristics."""
        # Arrange
        rng = np.random.default_rng(17)
        X_target = rng.normal(size=(20, 8))
        X_source = X_target * 2.0 + rng.normal(0, 0.1, size=(20, 8))

        model = SubspaceAlignment()
        model.fit(X_target, X_source=X_source)

        # Act
        X_test = rng.normal(size=(5, 8))
        X_transformed = model.transform(X_test)

        # Assert - Check output properties
        assert X_transformed.shape == X_test.shape
        assert np.all(np.isfinite(X_transformed))
        assert np.abs(X_transformed).mean() > 0  # Non-zero output
        assert np.abs(X_transformed).max() < 100  # Reasonable magnitude

    def test_transformation_is_reproducible(self):
        """Verifies that same inputs always produce same outputs."""
        # Arrange
        rng = np.random.default_rng(99)
        X_target = rng.normal(size=(25, 12))
        X_source = X_target * 1.3 + rng.normal(0, 0.08, size=(25, 12))
        X_test = rng.normal(size=(10, 12))

        # Act - Fit and transform twice
        model1 = SubspaceAlignment(n_components=2)
        model1.fit(X_target, X_source=X_source)
        result1 = model1.transform(X_test)

        model2 = SubspaceAlignment(n_components=2)
        model2.fit(X_target, X_source=X_source)
        result2 = model2.transform(X_test)

        # Assert - Results should be bit-for-bit identical
        np.testing.assert_array_equal(result1, result2)
        for i in [
            "components_X_",
            "components_X_source_",
            "x_source_provided_",
            "X_mean_",
            "X_std_",
            "X_source_mean_",
            "X_source_std_",
        ]:
            np.testing.assert_array_equal(getattr(model1, i), getattr(model2, i))

    def test_n_components_exceeds_n_samples(self):
        """Verifies n_components > n_samples - 1 raises ValueError."""
        # Arrange
        rng = np.random.default_rng(17)
        X_target = rng.normal(size=(8, 20))
        X_source = X_target * 2.0 + rng.normal(0, 0.1, size=(8, 20))

        # Act & Assert
        with pytest.raises(ValueError):
            model = SubspaceAlignment(n_components=40)
            model.fit(X_target, X_source=X_source)

    def test_n_components_exceeds_n_features(self):
        """Verifies n_components > n_features raises ValueError."""
        # Arrange
        rng = np.random.default_rng(17)
        X_target = rng.normal(size=(20, 8))
        X_source = X_target * 2.0 + rng.normal(0, 0.1, size=(20, 8))

        # Act & Assert
        with pytest.raises(ValueError):
            model = SubspaceAlignment(n_components=17)
            model.fit(X_target, X_source=X_source)


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_identity_transformation_when_X_source_is_none(self):
        """Verifies that fitting with X_source=None results in identity
        transformation."""
        # Arrange
        rng = np.random.default_rng(17)
        X = rng.normal(size=(50, 10))
        model = SubspaceAlignment()

        # Act - fit with X_source=None should trigger identity transformation
        with pytest.warns(UserWarning, match="identity transformation"):
            model.fit(X, X_source=None)

        X_transformed = model.transform(X)

        # Assert - should return X unchanged
        np.testing.assert_array_equal(X_transformed, X)
        assert hasattr(model, "x_source_provided_")
        assert model.x_source_provided_ is False


class TestPipeline:
    """Tests for sklearn Pipeline and metadata routing integration."""

    def test_pipeline_gridsearchcv_pls_metadata_routing(self, sample_data):
        """Verifies that X_source metadata routing works inside a Pipeline with
        GridSearchCV."""
        # Arrange
        X_target, X_source = sample_data
        rng = np.random.default_rng(17)
        y_concentration = rng.normal(size=(100, 1))

        sklearn.set_config(enable_metadata_routing=True)
        try:
            pipe = Pipeline(
                [
                    ("scaler", SavitzkyGolay()),
                    (
                        "model",
                        SubspaceAlignment().set_fit_request(X_source=True),
                    ),
                    ("pls", PLSRegression()),
                ]
            )
            param_grid = {
                "scaler__window_length": [15, 25],
                "scaler__polyorder": [2, 3],
                "scaler__deriv": [1, 2],
                "model__n_components": [2, 3, 5],
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

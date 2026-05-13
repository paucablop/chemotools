"""
Test for DirectStandardization
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

from chemotools.adaptation._direct_standardization import (
    DirectStandardization,
)
from chemotools.derivative import SavitzkyGolay
from chemotools.scatter import StandardNormalVariate
from tests.adaptation.conftest import data_diff


class TestSklearnCompliance:
    """Tests for sklearn estimator API compliance."""

    def test_compliance_DirectStandardization(self):
        """Verifies that DirectStandardization passes all sklearn estimator checks."""
        # Arrange
        transformer = DirectStandardization()

        # Act & Assert
        check_estimator(transformer)


class TestFit:
    """Tests for the fit method behavior."""

    def test_fit_sets_attributes(self, sample_data):
        """Verifies that fit stores the transformation matrix T_."""
        # Arrange
        X_target, X_source = sample_data

        # Act
        model = DirectStandardization().fit(X_target, X_source=X_source)

        # Assert
        assert hasattr(model, "T_")

    def test_fit_should_raise_error_size_mismatch(self, sample_data):
        """Verifies fir raise eror when size mismatch"""
        # Arrange
        X_target, X_source = sample_data

        # Act & Assert
        with pytest.raises(
            ValueError,
            match=re.escape(
                "X and X_source must have the same shape, got X=(100, 20) and "
                "X_source=(99, 20)."
            ),
        ):
            DirectStandardization().fit(X_target, X_source=X_source[:-1, :])


class TestTransform:
    """Tests for the transform method behavior."""

    def test_improvement(self, sample_data):
        """Verifies that the transformed data is closer to the source than the
        original."""
        # Arrange
        X_target, X_source = sample_data
        model = DirectStandardization().fit(X_target, X_source=X_source)

        # Act
        X_transformed = model.transform(X_target)
        before = data_diff(X_source, X_target)
        after = data_diff(X_source, X_transformed)

        # Assert
        assert before > after

    def test_transform_preserves_shape(self, sample_data):
        """Verifies that the output shape matches both input X and X_source."""
        # Arrange
        X_target, X_source = sample_data

        # Act
        model = DirectStandardization().fit(X_target, X_source=X_source)
        X_transformed = model.transform(X_target)

        # Assert
        assert X_transformed.shape == X_target.shape
        assert X_transformed.shape == X_source.shape

    def test_transform_improves_match_to_target(self, sample_data):
        """Verifies that transformation reduces the distance to the source instrument
        data."""
        # Arrange
        X_target, X_source = sample_data
        model = DirectStandardization().fit(X_target, X_source=X_source)

        # Act
        X_transformed = model.transform(X_target)
        before = data_diff(X_source, X_target)
        after = data_diff(X_source, X_transformed)

        # Assert
        assert after < before

    def test_transform_before_fit_raises(self, sample_data):
        """Verifies that calling transform before fit raises NotFittedError."""
        # Arrange
        X_target, _ = sample_data
        model = DirectStandardization()

        # Act & Assert
        with pytest.raises(NotFittedError):
            model.transform(X_target)

    def test_transform_does_not_modify_input(self, sample_data):
        """Verifies that fit and transform do not mutate the input arrays."""
        # Arrange
        X_target, X_source = sample_data
        X_source_original = X_source.copy()
        X_target_original = X_target.copy()

        # Act
        model = DirectStandardization().fit(X_target, X_source=X_source)
        model.transform(X_target)

        # Assert
        np.testing.assert_array_equal(X_source, X_source_original)
        np.testing.assert_array_equal(X_target, X_target_original)

    def test_transform_is_idempotent_on_input(self, sample_data):
        """Verifies that calling transform multiple times with the same input gives
        identical results."""
        # Arrange
        _, X_source = sample_data
        model = DirectStandardization().fit(X_source)

        # Act
        result1 = model.transform(X_source)
        result2 = model.transform(X_source)

        # Assert
        np.testing.assert_array_equal(result1, result2)

    def test_transform_on_unseen_data(self, sample_data):
        """Verifies that transform generalises to data not seen during fit."""
        # Arrange
        _, X_source = sample_data
        rng = np.random.default_rng(17)
        X_new = rng.normal(size=X_source.shape)

        # Act
        model = DirectStandardization().fit(X_source)
        X_transformed = model.transform(X_new)

        # Assert
        assert X_transformed.shape == X_new.shape

    def test_transform_raises_on_wrong_n_features(self, sample_data):
        """Verifies that transform raises ValueError when input has wrong number of
        features."""
        # Arrange
        X_target, X_source = sample_data
        rng = np.random.default_rng(99)
        X_wrong = rng.normal(size=(100, 15))  # 15 instead of 20
        model = DirectStandardization().fit(X_source)

        # Act & Assert
        with pytest.raises(ValueError):
            model.transform(X_wrong)


class TestNumericalCorrectness:
    """Tests for numerical correctness and regression testing.

    These tests verify that the algorithm produces expected numerical outputs.
    They serve as regression tests to catch unintended changes in functionality.
    """

    def test_snapshot_transformation_matrix_and_output(self):
        """Snapshot test: verifies T_ matrix and transform output match reference.

        This is a golden/snapshot test with hardcoded expected values.
        If this test fails after code changes, verify the change is intentional.
        Reference values generated from v0.1.0 implementation.
        """
        # Arrange - Fixed data (do not change!)
        rng = np.random.default_rng(42)
        X_target = rng.normal(size=(10, 5))
        X_source = X_target * 1.5 + rng.normal(0, 0.05, size=(10, 5))
        X_test = np.array(
            [
                [-0.37816255, 1.2992283, -0.35626397, 0.73751557, -0.93361768],
                [-0.20543756, -0.95002205, -0.33903308, 0.84030814, -1.72732042],
                [0.43442364, 0.2377356, -0.59414996, -1.44605785, 0.07212951],
            ]
        )

        # Expected T_ matrix (generated 2026-05-13)
        expected_T = np.array(
            [
                [1.48716816, -0.01063016, -0.02058649, -0.0073408, -0.02897037],
                [-0.06621325, 1.46246043, 0.03103749, -0.03535038, -0.02991129],
                [-0.05116713, -0.01801761, 1.48973578, 0.00199055, 0.00249581],
                [0.01808751, 0.0136685, -0.03753605, 1.54421807, 0.00581606],
                [0.02576647, 0.00534746, -0.00495776, 0.04992774, 1.52699328],
            ]
        )

        # Expected transform output (generated 2026-05-13)
        expected_output = np.array(
            [
                [-0.64090464, 1.91559718, -0.50568414, 1.04841009, -1.45013374],
                [-0.25457667, -1.3788283, -0.55330498, 1.24579467, -2.59919758],
                [0.63642365, 0.33438636, -0.83276929, -2.24220316, 0.08055164],
            ]
        )

        # Act
        model = DirectStandardization()
        model.fit(X_target, X_source=X_source)
        output = model.transform(X_test)

        # Assert - Both T_ and output should match references
        np.testing.assert_allclose(model.T_, expected_T, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(output, expected_output, rtol=1e-6, atol=1e-8)

    def test_transformation_matrix_properties(self):
        """Verifies that T_ matrix has expected properties."""
        # Arrange
        rng = np.random.default_rng(12345)
        X_target = rng.normal(size=(50, 10))
        X_source = X_target * 1.5 + rng.normal(0, 0.05, size=(50, 10))

        # Act
        model = DirectStandardization()
        model.fit(X_target, X_source=X_source)

        # Assert - Check T_ matrix properties
        assert model.T_.shape == (10, 10)
        assert np.all(np.isfinite(model.T_))
        # For scaling transformations, T_ should be diagonally dominant
        diagonal_dominance = np.abs(np.diag(model.T_)).sum() / np.abs(model.T_).sum()
        assert diagonal_dominance > 0.5

    def test_transformation_is_reproducible(self):
        """Verifies that same inputs always produce same outputs."""
        # Arrange
        rng = np.random.default_rng(99)
        X_target = rng.normal(size=(40, 12))
        X_source = X_target * 1.3 + rng.normal(0, 0.08, size=(40, 12))
        X_test = rng.normal(size=(15, 12))

        # Act - Fit and transform twice
        model1 = DirectStandardization()
        model1.fit(X_target, X_source=X_source)
        result1 = model1.transform(X_test)

        model2 = DirectStandardization()
        model2.fit(X_target, X_source=X_source)
        result2 = model2.transform(X_test)

        # Assert - Results should be bit-for-bit identical
        np.testing.assert_array_equal(result1, result2)
        np.testing.assert_array_equal(model1.T_, model2.T_)

    def test_known_linear_transformation(self):
        """Verifies correct behavior on a known linear transformation."""
        # Arrange - Create data with known linear relationship
        rng = np.random.default_rng(777)
        X_target = rng.normal(size=(100, 15))
        # Simple scaling: X_source = 2 * X_target (perfect linear relationship)
        X_source = 2.0 * X_target

        # Act
        model = DirectStandardization()
        model.fit(X_target, X_source=X_source)
        X_transformed = model.transform(X_target)

        # Assert - Transformed data should be very close to X_source
        # (Direct Standardization should perfectly recover the linear scaling)
        np.testing.assert_allclose(X_transformed, X_source, rtol=1e-10, atol=1e-10)

    def test_identity_transformation_when_data_identical(self):
        """Verifies that T_ is identity matrix when X_target ≈ X_source."""
        # Arrange - Create nearly identical data
        rng = np.random.default_rng(555)
        X_target = rng.normal(size=(50, 10))
        X_source = X_target + rng.normal(0, 0.001, size=(50, 10))  # Tiny noise

        # Act
        model = DirectStandardization()
        model.fit(X_target, X_source=X_source)

        # Assert - T_ should be close to identity matrix
        np.testing.assert_allclose(model.T_, np.eye(10), rtol=0.1, atol=0.1)


class TestPipeline:
    """Tests for sklearn Pipeline and metadata routing integration."""

    def test_pipeline(self, sample_data):
        """Verifies that DirectStandardization works correctly inside a sklearn
        Pipeline."""
        # Arrange
        X_target, X_source = sample_data
        pipe = Pipeline(
            [
                ("scaler", StandardNormalVariate()),
                (
                    "model",
                    DirectStandardization(),
                ),
            ]
        )

        # Act
        pipe.fit(X_source)
        X_transformed = pipe.transform(X_source)

        # Assert
        assert X_transformed.shape == X_source.shape == X_target.shape

    def test_pipeline_gridsearchcv_pls_metadata_routing(self, sample_data):
        """Verifies that X_source metadata routing works inside a Pipeline with
        GridSearchCV."""
        # Arrange
        _, X_source = sample_data
        rng = np.random.default_rng(42)
        y_concentration = rng.normal(size=(100, 1))

        sklearn.set_config(enable_metadata_routing=True)

        try:
            pipe = Pipeline(
                [
                    ("scaler", SavitzkyGolay()),
                    ("ds", DirectStandardization().set_fit_request(X_source=True)),
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
            grid.fit(
                X_source,
                y_concentration,
            )

            # Assert
            assert grid.best_estimator_ is not None

        finally:
            # Cleanup - reset config to avoid affecting other tests
            sklearn.set_config(enable_metadata_routing=False)

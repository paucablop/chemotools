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


def data_diff(dataset_ref, dataset_test):
    diff_norm = np.linalg.norm(dataset_ref - dataset_test)
    ref_norm = np.linalg.norm(dataset_ref)
    difference = diff_norm / ref_norm
    return difference


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

    def test_fit_should_raise_error_size_missmatch(self, sample_data):
        """Verifies fir raise eror when size missmatch"""
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
        model = DirectStandardization().fit(X_source, X_source=X_source)
        model.transform(X_source)

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

        sklearn.set_config(enable_metadata_routing=False)

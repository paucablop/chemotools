"""Tests for enhanced PLSRegression with automatic variance calculation."""

import pytest
import numpy as np
from sklearn.cross_decomposition import PLSRegression as SklearnPLSRegression
from sklearn.utils.estimator_checks import check_estimator

from chemotools.models import PLSRegression


# Test compliance with scikit-learn
def test_compliance_pls_regression():
    # Arrange
    transformer = PLSRegression()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
class TestPLSRegressionCompatibility:
    """Test that enhanced PLSRegression maintains sklearn API compatibility."""

    def test_same_predictions_as_sklearn(self):
        """Test that predictions match sklearn's PLSRegression exactly."""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        # Fit both models with same parameters
        sklearn_pls = SklearnPLSRegression(n_components=5, scale=False)
        chemotools_pls = PLSRegression(n_components=5, scale=False)

        sklearn_pls.fit(X, y)
        chemotools_pls.fit(X, y)

        # Act
        sklearn_pred = sklearn_pls.predict(X)
        chemotools_pred = chemotools_pls.predict(X)

        # Assert - predictions should be identical
        np.testing.assert_array_almost_equal(
            sklearn_pred,
            chemotools_pred,
            decimal=10,
            err_msg="Predictions should match sklearn exactly",
        )

    def test_same_transform_as_sklearn(self):
        """Test that transform() produces same scores as sklearn."""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        sklearn_pls = SklearnPLSRegression(n_components=5)
        chemotools_pls = PLSRegression(n_components=5)

        sklearn_pls.fit(X, y)
        chemotools_pls.fit(X, y)

        # Act
        sklearn_scores = sklearn_pls.transform(X)
        chemotools_scores = chemotools_pls.transform(X)

        # Assert
        np.testing.assert_array_almost_equal(
            sklearn_scores,
            chemotools_scores,
            decimal=10,
            err_msg="Transform scores should match sklearn exactly",
        )

    def test_same_attributes_as_sklearn(self):
        """Test that all sklearn attributes are present and identical."""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        sklearn_pls = SklearnPLSRegression(n_components=5)
        chemotools_pls = PLSRegression(n_components=5)

        sklearn_pls.fit(X, y)
        chemotools_pls.fit(X, y)

        # Assert - check all important sklearn attributes match
        sklearn_attributes = [
            "x_weights_",
            "y_weights_",
            "x_loadings_",
            "y_loadings_",
            "x_scores_",
            "y_scores_",
            "x_rotations_",
            "y_rotations_",
            "coef_",
            "intercept_",
            "n_features_in_",
        ]

        for attr in sklearn_attributes:
            sklearn_val = getattr(sklearn_pls, attr)
            chemotools_val = getattr(chemotools_pls, attr)
            np.testing.assert_array_almost_equal(
                sklearn_val,
                chemotools_val,
                decimal=10,
                err_msg=f"Attribute {attr} should match sklearn exactly",
            )

    def test_same_score_as_sklearn(self):
        """Test that score() method produces same R² as sklearn."""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(100) * 0.1

        sklearn_pls = SklearnPLSRegression(n_components=5)
        chemotools_pls = PLSRegression(n_components=5)

        sklearn_pls.fit(X, y)
        chemotools_pls.fit(X, y)

        # Act
        sklearn_r2 = sklearn_pls.score(X, y)
        chemotools_r2 = chemotools_pls.score(X, y)

        # Assert
        np.testing.assert_almost_equal(
            sklearn_r2,
            chemotools_r2,
            decimal=10,
            err_msg="R² score should match sklearn exactly",
        )

    def test_works_with_scale_true(self):
        """Test that scaling parameter works correctly."""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(100, 50) * 100  # Large scale
        y = np.random.randn(100) * 10

        sklearn_pls = SklearnPLSRegression(n_components=5, scale=True)
        chemotools_pls = PLSRegression(n_components=5, scale=True)

        sklearn_pls.fit(X, y)
        chemotools_pls.fit(X, y)

        # Act
        sklearn_pred = sklearn_pls.predict(X)
        chemotools_pred = chemotools_pls.predict(X)

        # Assert
        np.testing.assert_array_almost_equal(
            sklearn_pred,
            chemotools_pred,
            decimal=10,
            err_msg="Predictions with scale=True should match sklearn",
        )

    def test_works_with_multivariate_y(self):
        """Test that it works with multiple y variables."""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100, 3)  # 3 y variables

        sklearn_pls = SklearnPLSRegression(n_components=5)
        chemotools_pls = PLSRegression(n_components=5)

        sklearn_pls.fit(X, y)
        chemotools_pls.fit(X, y)

        # Act
        sklearn_pred = sklearn_pls.predict(X)
        chemotools_pred = chemotools_pls.predict(X)

        # Assert
        np.testing.assert_array_almost_equal(
            sklearn_pred,
            chemotools_pred,
            decimal=10,
            err_msg="Multivariate predictions should match sklearn",
        )


class TestPLSRegressionVarianceCalculation:
    """Test the new variance calculation features."""

    def test_has_explained_variance_attributes(self):
        """Test that explained variance attributes are created after fitting."""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        # Act
        pls = PLSRegression(n_components=50)
        pls.fit(X, y)

        # Assert
        assert hasattr(pls, "explained_x_variance_ratio_")
        assert hasattr(pls, "explained_y_variance_ratio_")
        assert len(pls.explained_x_variance_ratio_) == 50
        assert len(pls.explained_y_variance_ratio_) == 50

    def test_x_variance_sums_to_one(self):
        """Test that X-space variance ratios sum to ~1.0 (100%)."""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        # Act
        pls = PLSRegression(n_components=50)
        pls.fit(X, y)

        # Assert
        # X-space variance should sum to approximately 1.0
        np.testing.assert_almost_equal(
            pls.explained_x_variance_ratio_.sum(),
            1.0,
            decimal=2,
            err_msg="X-space variance should sum to ~1.0",
        )

    def test_y_variance_high_with_strong_correlation(self):
        """Test Y variance calculation against known literature example.

        Abdi, H. (2003). Partial Least Squares (PLS) Regression.
        In Lewis-Beck M., Bryman A., Futing T. (Eds.),
        Encyclopedia of Social Sciences Research Methods.
        Thousand Oaks (CA): Sage.
        """
        # Arrange - Known example from literature
        X = np.array(
            [
                [7, 7, 13, 7],
                [4, 3, 14, 7],
                [10, 5, 12, 5],
                [16, 7, 11, 3],
                [13, 3, 10, 3],
            ],
            dtype=float,
        )

        y = np.array(
            [[14, 7, 8], [10, 7, 6], [8, 5, 5], [2, 4, 7], [6, 2, 4]], dtype=float
        )

        # Act
        pls = PLSRegression(n_components=3)
        pls.fit(X, y)

        # Assert - Expected values from literature
        expected_y_var = np.array([0.63331666, 0.22064505, 0.10437163])

        np.testing.assert_array_almost_equal(
            pls.explained_y_variance_ratio_,
            expected_y_var,
            decimal=5,
            err_msg="Y variance should match literature values",
        )

    def test_x_variance_all_positive(self):
        """Test that X-space variance ratios are all positive."""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        # Act
        pls = PLSRegression(n_components=5)
        pls.fit(X, y)

        # Assert
        assert np.all(pls.explained_x_variance_ratio_ >= 0), (
            "X-space variance should be non-negative"
        )

    def test_y_variance_all_positive(self):
        """Test that Y-space variance ratios are all positive."""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        # Act
        pls = PLSRegression(n_components=5)
        pls.fit(X, y)

        # Assert
        assert np.all(pls.explained_y_variance_ratio_ >= 0), (
            "Y-space variance should be non-negative"
        )

    def test_y_variance_is_float_array(self):
        """Test that Y-space variance is a proper numpy array."""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(100) * 0.1

        # Act
        pls = PLSRegression(n_components=3)
        pls.fit(X, y)

        # Assert
        assert isinstance(pls.explained_y_variance_ratio_, np.ndarray)
        assert pls.explained_y_variance_ratio_.dtype == np.float64

    def test_variance_calculation_with_pandas(self):
        """Test that variance calculation works with pandas DataFrame/Series."""
        # Arrange
        pytest.importorskip("pandas")
        import pandas as pd

        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 50))
        y = pd.Series(np.random.randn(100))

        # Act
        pls = PLSRegression(n_components=5)
        pls.fit(X, y)

        # Assert
        assert hasattr(pls, "explained_x_variance_ratio_")
        assert hasattr(pls, "explained_y_variance_ratio_")
        assert len(pls.explained_x_variance_ratio_) == 5
        assert len(pls.explained_y_variance_ratio_) == 5

    def test_variance_with_different_n_components(self):
        """Test variance calculation with different numbers of components."""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        # Act & Assert for different component counts
        for n_comp in [2, 5, 10]:
            pls = PLSRegression(n_components=n_comp)
            pls.fit(X, y)

            assert len(pls.explained_x_variance_ratio_) == n_comp
            assert len(pls.explained_y_variance_ratio_) == n_comp

    def test_variance_calculation_preserves_sklearn_behavior(self):
        """Test that adding variance calculation doesn't change predictions."""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(100) * 0.1

        sklearn_pls = SklearnPLSRegression(n_components=5)
        chemotools_pls = PLSRegression(n_components=5)

        # Fit both
        sklearn_pls.fit(X, y)
        chemotools_pls.fit(X, y)

        # Calculate predictions before accessing variance
        sklearn_pred = sklearn_pls.predict(X)
        chemotools_pred = chemotools_pls.predict(X)

        # Access variance (this shouldn't change anything)
        _ = chemotools_pls.explained_x_variance_ratio_
        _ = chemotools_pls.explained_y_variance_ratio_

        # Recalculate predictions
        chemotools_pred_after = chemotools_pls.predict(X)

        # Assert
        np.testing.assert_array_almost_equal(
            sklearn_pred,
            chemotools_pred,
            decimal=10,
            err_msg="Predictions should match sklearn",
        )
        np.testing.assert_array_almost_equal(
            chemotools_pred,
            chemotools_pred_after,
            decimal=10,
            err_msg="Variance calculation shouldn't change predictions",
        )


class TestPLSRegressionEdgeCases:
    """Test edge cases and error handling."""

    def test_works_with_minimum_samples(self):
        """Test that it works with minimum number of samples."""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(10, 5)  # Only 10 samples
        y = np.random.randn(10)

        # Act
        pls = PLSRegression(n_components=3)
        pls.fit(X, y)

        # Assert
        assert hasattr(pls, "explained_x_variance_ratio_")
        assert hasattr(pls, "explained_y_variance_ratio_")

    def test_repr_shows_variance_info(self):
        """Test that __repr__ shows variance information after fitting."""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        # Act
        pls = PLSRegression(n_components=5)
        pls.fit(X, y)
        repr_str = repr(pls)

        # Assert
        assert "X-space variance explained" in repr_str
        assert "Y-space variance explained" in repr_str

    def test_repr_before_fitting(self):
        """Test that __repr__ works before fitting (no variance info)."""
        # Arrange
        pls = PLSRegression(n_components=5)

        # Act
        repr_str = repr(pls)

        # Assert
        assert "n_components=5" in repr_str
        assert "X-space variance" not in repr_str  # Not fitted yet

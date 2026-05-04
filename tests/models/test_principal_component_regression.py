"""Test for PCR"""

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.utils.estimator_checks import check_estimator

from chemotools.models._principal_component_regression import (
    PrincipalComponentRegression,
)
from chemotools.scatter import StandardNormalVariate


# Test compliance with scikit-learn
def test_compliance_PrincipalComponentRegression():
    # Arrange
    transformer = PrincipalComponentRegression()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
class TestPrincipalComponentRegressionCompatibility:
    """
    Test that enhanced PrincipalComponentRegression maintains sklearn API compatibility.
    """

    def test_same_predictions_as_sklearn(self):
        # Arrange
        np.random.seed(17)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        # Fit both models with same parameters
        sklearn_pcr = make_pipeline(PCA(n_components=5), LinearRegression())
        chemotools_pcr = PrincipalComponentRegression(n_components=5)

        sklearn_pcr.fit(X, y)
        chemotools_pcr.fit(X, y)

        # Act
        sklearn_pred = sklearn_pcr.predict(X)
        chemotools_pred = chemotools_pcr.predict(X)

        # Assert - prediction should be identical
        np.testing.assert_array_almost_equal(
            sklearn_pred,
            chemotools_pred,
            decimal=10,
            err_msg="Predictions should match sklearn exactly",
        )

    def test_y_dimensions(self):
        """
        Test that y dimesions predicted is the right one.
        """

        # Arrange
        np.random.seed(17)
        X = np.random.randn(100, 50)

        # Act & Assert - Test 1D y
        y = np.random.randn(100)
        chemotools_pcr = PrincipalComponentRegression(n_components=5)
        chemotools_pcr.fit(X, y)
        assert chemotools_pcr.predict(X).shape == y.shape

        # Act & Assert - Test 2D y
        for n_y in [1, 2, 3, 4]:
            y = np.random.randn(100, n_y)
            chemotools_pcr = PrincipalComponentRegression(n_components=5)
            chemotools_pcr.fit(X, y)
            assert chemotools_pcr.predict(X).shape == y.shape

    def test_attributs(self):
        """
        Test all the attributes are presents
        """
        # Arrange
        np.random.seed(17)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        # Act
        sklearn_pcr = make_pipeline(PCA(n_components=5), LinearRegression())
        chemotools_pcr = PrincipalComponentRegression(n_components=5)
        sklearn_pcr.fit(X, y)
        chemotools_pcr.fit(X, y)

        # Assert - PCA
        sklearn_attributes = [
            "components_",
            "explained_variance_",
            "explained_variance_ratio_",
        ]

        for attr in sklearn_attributes:
            sklearn_pca = sklearn_pcr.named_steps["pca"]
            sklearn_val = getattr(sklearn_pca, attr)
            chemotools_val = getattr(chemotools_pcr, attr)
            np.testing.assert_array_almost_equal(
                sklearn_val,
                chemotools_val,
                decimal=10,
                err_msg=f"Attribute {attr} should match sklearn exactly",
            )
        # Assert - linear regression
        sklearn_attributes = [
            "coef_",
            "intercept_",
            "rank_",
            "singular_",
        ]

        for attr in sklearn_attributes:
            sklearn_lr = sklearn_pcr.named_steps["linearregression"]
            sklearn_val = getattr(sklearn_lr, attr)
            chemotools_val = getattr(chemotools_pcr, attr)
            np.testing.assert_array_almost_equal(
                sklearn_val,
                chemotools_val,
                decimal=10,
                err_msg=f"Attribute {attr} should match sklearn exactly",
            )

    def test_tranform_as_sklearn(self):
        """
        Test that the transform function match with the PCA transform
        """

        # Arrange
        np.random.seed(17)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        # Act
        sklearn_pcr = make_pipeline(PCA(n_components=5), LinearRegression())
        chemotools_pcr = PrincipalComponentRegression(n_components=5)
        sklearn_pcr.fit(X, y)
        chemotools_pcr.fit(X, y)

        # Assert - PCA
        assert (
            chemotools_pcr.transform(X).shape
            == PCA(n_components=5).fit_transform(X).shape
        )
        np.testing.assert_array_almost_equal(
            chemotools_pcr.transform(X),
            PCA(n_components=5).fit_transform(X),
            decimal=10,
            err_msg="The transformed data should match sklearn exactly",
        )

    def test_same_attributes_as_sklearn(self):
        """
        Test that all sklearn attributes are present and identical.
        """

        # Arrange
        np.random.seed(17)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        # Act
        sklearn_pcr = make_pipeline(PCA(n_components=5), LinearRegression())
        chemotools_pcr = PrincipalComponentRegression(n_components=5)

        sklearn_pcr.fit(X, y)
        chemotools_pcr.fit(X, y)

        sklearn_pca = sklearn_pcr.named_steps["pca"]
        sklearn_lr = sklearn_pcr.named_steps["linearregression"]

        # Assert - PCA attributes
        pca_attributes = [
            "components_",
            "explained_variance_",
            "explained_variance_ratio_",
            "mean_",
            "noise_variance_",
        ]

        for attr in pca_attributes:
            sklearn_val = getattr(sklearn_pca, attr)
            chemotools_val = getattr(chemotools_pcr, attr)

            np.testing.assert_array_almost_equal(
                sklearn_val,
                chemotools_val,
                decimal=10,
                err_msg=f"Attribute {attr} should match sklearn exactly",
            )

        # Assert - LinearRegression attributes
        lr_attributes = [
            "coef_",
            "rank_",
            "singular_",
            "intercept_",
        ]

        for attr in lr_attributes:
            sklearn_val = getattr(sklearn_lr, attr)
            chemotools_val = getattr(chemotools_pcr, attr)

            np.testing.assert_array_almost_equal(
                sklearn_val,
                chemotools_val,
                decimal=10,
                err_msg=f"Attribute {attr} should match sklearn exactly",
            )

    def test_same_score_as_sklearn(self):
        """
        Test that score() method produces same R² as sklearn.
        """
        # Arrange
        np.random.seed(17)
        X = np.random.randn(100, 50)
        y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(100)

        # Fit both models with same parameters
        sklearn_pcr = make_pipeline(PCA(n_components=5), LinearRegression())
        chemotools_pcr = PrincipalComponentRegression(n_components=5)

        sklearn_pcr.fit(X, y)
        chemotools_pcr.fit(X, y)

        # Act
        sklearn_r2 = sklearn_pcr.score(X, y)
        chemotools_r2 = chemotools_pcr.score(X, y)

        # Assert
        np.testing.assert_almost_equal(
            sklearn_r2,
            chemotools_r2,
            decimal=10,
            err_msg="R² score should match sklearn exactly",
        )

    def test_works_with_multivariate_y(self):
        """
        Test that it works with multiple y variables
        """
        # Arrange
        np.random.seed(17)
        X = np.random.randn(100, 50)
        y = np.random.randn(100, 3)

        # Fit both models with same parameters
        sklearn_pcr = make_pipeline(PCA(n_components=5), LinearRegression())
        chemotools_pcr = PrincipalComponentRegression(n_components=5)

        sklearn_pcr.fit(X, y)
        chemotools_pcr.fit(X, y)

        # Act
        sklearn_pred = sklearn_pcr.predict(X)
        chemotools_pred = chemotools_pcr.predict(X)

        # Assert
        np.testing.assert_array_almost_equal(
            sklearn_pred,
            chemotools_pred,
            decimal=10,
            err_msg="Multivariate predictions should match sklearn",
        )


class TestPrincipalComponentRegressionVarianceCalculation:
    """
    Test the new variance calculation features.
    """

    def test_has_explained_variance_attributes(self):
        """
        Test that explained variance attributes are created after fitting.
        """
        # Arrange
        np.random.seed(17)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        # Act
        chemotools_pcr = PrincipalComponentRegression(n_components=50)
        chemotools_pcr.fit(X, y)

        # Assert
        # X-space variance should sum to approximately 1.0
        np.testing.assert_almost_equal(
            chemotools_pcr.explained_variance_ratio_.sum(),
            1.0,
            decimal=2,
            err_msg="X-space variance should sum to ~1.0",
        )

    def test_variance_all_positive(self):
        """
        Test the X-space variance ratios are all positive
        """
        # Arrange
        np.random.seed(17)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        # Act
        chemotools_pcr = PrincipalComponentRegression(n_components=50)
        chemotools_pcr.fit(X, y)

        # Assert
        assert np.all(chemotools_pcr.explained_variance_ratio_ >= 0), (
            "X-space variance should be non-negative"
        )

    def test_variance_calculation_with_pandas(self):
        """
        Test that variance calculation works with pandas DataFrame/Series.
        """
        # Arrange
        pytest.importorskip("pandas")
        import pandas as pd

        np.random.seed(17)
        X = pd.DataFrame(np.random.randn(100, 50))
        y = pd.Series(np.random.randn(100))

        # Act
        chemotools_pcr = PrincipalComponentRegression(n_components=5)
        chemotools_pcr.fit(X, y)

        # Assert
        assert hasattr(chemotools_pcr, "mean_")
        assert hasattr(chemotools_pcr, "explained_variance_ratio_")
        assert len(chemotools_pcr.explained_variance_ratio_) == 5

    def test_variance_with_different_n_components(self):
        """
        Test variance calculation with different numbers of components.
        """
        # Arrange
        np.random.seed(17)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        # Act & Assert for different counts
        for n_comp in [2, 5, 10]:
            chemotools_pcr = PrincipalComponentRegression(n_components=n_comp)
            chemotools_pcr.fit(X, y)

            assert len(chemotools_pcr.explained_variance_ratio_) == n_comp

    def test_variance_calculation_preserves_sklearn_behavior(self):
        """
        Test that adding variance calculation doesn't change predictions.
        """
        # Arrange
        np.random.seed(17)
        X = np.random.randn(100, 50)
        y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(100) * 0.1

        # Fit both models with same parameters
        sklearn_pcr = make_pipeline(PCA(n_components=5), LinearRegression())
        chemotools_pcr = PrincipalComponentRegression(n_components=5)

        sklearn_pcr.fit(X, y)
        chemotools_pcr.fit(X, y)

        # Act - calculate prediction before accessing variance
        sklearn_pred = sklearn_pcr.predict(X)
        chemotools_pred = chemotools_pcr.predict(X)

        # Act - Access variance (this shouldn't change anything)
        _ = chemotools_pcr.explained_variance_ratio_

        # Act - recalculate the prediction
        chemotools_pred_after = chemotools_pcr.predict(X)

        # Assert
        np.testing.assert_array_almost_equal(
            sklearn_pred,
            chemotools_pred,
            decimal=10,
            err_msg="Prediction should match sklearn",
        )
        np.testing.assert_array_almost_equal(
            chemotools_pred,
            chemotools_pred_after,
            decimal=10,
            err_msg="Variance calculation shouldn't change predictions",
        )


class TestPrincipalComponentRegressionEdgeCases:
    """
    Test edge cases and error handling
    """

    def test_works_with_minimum_samples(self):
        """
        Test that it works with minimum number of samples.
        """
        # Arrange
        np.random.seed(17)
        X = np.random.randn(10, 5)  # Only 10 samples
        y = np.random.randn(10)

        # Act
        chemotools_pcr = PrincipalComponentRegression(n_components=3)
        chemotools_pcr.fit(X, y)

        # Assert
        assert hasattr(chemotools_pcr, "mean_")
        assert hasattr(chemotools_pcr, "coef_")


def test_pipeline_gridsearchcv():
    # Arrange
    np.random.seed(17)
    X = np.random.randn(10, 5)  # Only 10 samples
    y = np.random.randn(10)

    pipe = Pipeline(
        [
            ("scaler", StandardNormalVariate()),
            ("model", PrincipalComponentRegression()),
        ]
    )

    param_grid = {
        "model__n_components": np.arange(1, 6),
    }

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=3,
        scoring="r2",
        error_score="raise",
    )

    # Act
    grid.fit(X, y)

    # Assert
    assert grid.best_estimator_ is not None
    assert np.isfinite(grid.best_score_)
    assert "model__n_components" in grid.best_params_

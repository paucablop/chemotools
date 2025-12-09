import numpy as np
from sklearn.cross_decomposition import PLSRegression
from chemotools.inspector import PLSRegressionInspector


class TestPredictionSummary:
    def test_regression_summary_structure(self):
        # Arrange
        X = np.random.rand(10, 5)
        y = np.random.rand(10)
        pls = PLSRegression(n_components=2).fit(X, y)
        inspector = PLSRegressionInspector(pls, X, y)

        # Act
        summary = inspector.regression_summary()

        # Assert
        assert isinstance(summary, dict)
        assert "train" in summary
        assert summary["train"].rmse is not None
        assert summary["train"].r2 is not None
        assert isinstance(summary["train"].rmse, float)
        assert isinstance(summary["train"].r2, float)

    def test_regression_summary_multiple_datasets(self):
        # Arrange
        X_train = np.random.rand(10, 5)
        y_train = np.random.rand(10)
        X_test = np.random.rand(5, 5)
        y_test = np.random.rand(5)

        pls = PLSRegression(n_components=2).fit(X_train, y_train)
        inspector = PLSRegressionInspector(
            pls, X_train, y_train, X_test=X_test, y_test=y_test
        )

        # Act
        summary = inspector.regression_summary()

        # Assert
        assert "train" in summary
        assert "test" in summary
        assert "val" not in summary

    def test_regression_summary_bias_calculation(self):
        # Arrange
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3])

        pls = PLSRegression(n_components=1)
        pls.fit(X, y)

        inspector = PLSRegressionInspector(pls, X, y)

        # Calculate expected bias manually
        y_pred = pls.predict(X).ravel()
        expected_bias = np.mean(y_pred - y)

        # Act
        summary = inspector.regression_summary()
        bias = summary["train"].bias

        # Assert
        assert summary["train"].bias is not None
        assert isinstance(bias, float)
        np.testing.assert_almost_equal(bias, expected_bias)

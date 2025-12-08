import numpy as np
from sklearn.cross_decomposition import PLSRegression
from chemotools.inspector import PLSRegressionInspector


class TestPredictionSummary:
    def test_prediction_summary_structure(self):
        # Arrange
        X = np.random.rand(10, 5)
        y = np.random.rand(10)
        pls = PLSRegression(n_components=2).fit(X, y)
        inspector = PLSRegressionInspector(pls, X, y)

        # Act
        summary = inspector.prediction_summary()

        # Assert
        assert isinstance(summary, dict)
        assert "train" in summary
        assert "RMSE" in summary["train"]
        assert "R2" in summary["train"]
        assert isinstance(summary["train"]["RMSE"], float)
        assert isinstance(summary["train"]["R2"], float)

    def test_prediction_summary_multiple_datasets(self):
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
        summary = inspector.prediction_summary()

        # Assert
        assert "train" in summary
        assert "test" in summary
        assert "val" not in summary

    def test_prediction_summary_bias_calculation(self):
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
        summary = inspector.prediction_summary()
        bias = summary["train"]["Bias"]

        # Assert
        assert "Bias" in summary["train"]
        assert isinstance(bias, float)
        np.testing.assert_almost_equal(bias, expected_bias)

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from chemotools.utils.check_inputs import check_input


class LinearCorrection(BaseEstimator, TransformerMixin):

    def _drift_correct_spectrum(self, x: np.ndarray) -> np.ndarray:

        # Can take any array and returns with a linear baseline correction
        # Find the x values at the edges of the spectrum
        y1: float = x[0]
        y2: float = x[-1]

        # Find the max and min wavenumebrs
        x1 = 0
        x2 = len(x)
        x_range = np.linspace(x1, x2, x2)

        # Calculate the straight line initial and end point
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        drift_correction = slope * x_range + intercept

        # Return the drift corrected spectrum
        return x - drift_correction

    def fit(self, X: np.ndarray, y=None) -> "LinearCorrection":
        # Check that X is a 2D array and has only finite values
        X = check_input(X)

        # Set the number of features
        self.n_features_in_ = X.shape[1]

        # Set the fitted attribute to True
        self._is_fitted = True

        return self

    def transform(self, X: np.ndarray, y=0, copy=True) -> np.ndarray:
        # Check that the estimator is fitted
        check_is_fitted(self, "_is_fitted")

        # Check that X is a 2D array and has only finite values
        X = check_input(X)
        X_ = X.copy()

        # Check that the number of features is the same as the fitted data
        if X_.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features but got {X_.shape[1]}")

        # Calculate non-negative values
        for i, x in enumerate(X_):
            X_[i, :] = self._drift_correct_spectrum(x)
        return X_.reshape(-1, 1) if X_.ndim == 1 else X_
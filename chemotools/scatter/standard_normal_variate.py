import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from chemotools.utils.check_inputs import check_input


class StandardNormalVariate(BaseEstimator, TransformerMixin):
    def fit(self, X: np.ndarray, y=None) -> "StandardNormalVariate":
        # Check that X is a 2D array and has only finite values
        X = check_input(X)

        # Set the number of features
        self.n_features_in_ = X.shape[1]

        # Set the fitted attribute to True
        self._is_fitted = True

        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        # Check that the estimator is fitted
        check_is_fitted(self, "_is_fitted")

        # Check that X is a 2D array and has only finite values
        X = check_input(X)
        X_ = X.copy()

        # Check that the number of features is the same as the fitted data
        if X_.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features but got {X_.shape[1]}")

        # Calculate the standard normal variate
        for i, x in enumerate(X_):
            X_[i] = self._calculate_standard_normal_variate(x)

        return X_.reshape(-1, 1) if X_.ndim == 1 else X_

    def _calculate_standard_normal_variate(self, x) -> np.ndarray:
        return (x - x.mean()) / x.std()
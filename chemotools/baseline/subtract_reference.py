import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from chemotools.utils.check_inputs import check_input


class SubtractReference(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        reference: np.ndarray = None,
    ):
        self.reference = reference

    def fit(self, X: np.ndarray, y=None) -> "SubtractReference":
        # Check that X is a 2D array and has only finite values
        X = check_input(X)

        # Set the number of features
        self.n_features_in_ = X.shape[1]

        # Set the fitted attribute to True
        self._is_fitted = True

        # Set the reference

        if self.reference is not None:
            self.reference_ = self.reference.copy()
            return self
        
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

        if self.reference is None:
            return X_.reshape(-1, 1) if X_.ndim == 1 else X_

        # Subtract the reference
        for i, x in enumerate(X_):
            X_[i] = self._subtract_reference(x)
        return X_.reshape(-1, 1) if X_.ndim == 1 else X_

    def _subtract_reference(self, x) -> np.ndarray:
        return x - self.reference_

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from chemotools.utils.check_inputs import check_input


class ConstantBaselineCorrection(BaseEstimator, TransformerMixin):
    def __init__(
        self, wavenumbers: np.ndarray = None, start: int = 0, end: int = 1
    ) -> None:
        self.wavenumbers = wavenumbers
        self.start = self._find_index(start)
        self.end = self._find_index(end)

    def fit(self, X: np.ndarray, y=None) -> "ConstantBaselineCorrection":
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
            raise ValueError(
                f"Expected {self.n_features_in_} features but got {X_.shape[1]}"
            )

        # Base line correct the spectra
        for i, x in enumerate(X_):
            mean_baseline = np.mean(x[self.start : self.end+1])
            X_[i, :] = x - mean_baseline
        return X_.reshape(-1, 1) if X_.ndim == 1 else X_

    def _find_index(self, target: float) -> int:
        if self.wavenumbers is None:
            return target
        wavenumbers = np.array(self.wavenumbers)
        return np.argmin(np.abs(wavenumbers - target))

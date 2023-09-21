import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted

from chemotools.utils.check_inputs import check_input


class ConstantBaselineCorrection(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    """
    A transformer that corrects a baseline by subtracting a constant value.
    The constant value is taken by the mean of the features between the start
    and end indices. This is a common preprocessing technique for UV-Vis spectra.

    Parameters
    ----------
    wavenumbers : np.ndarray, optional
        The wavenumbers corresponding to each feature in the input data.

    start : int, optional
        The index of the first feature to use for the baseline correction.

    end : int, optional
        The index of the last feature to use for the baseline correction.

    Attributes
    ----------
    start_index_ : int
        The index of the start of the range. It is 0 if the wavenumbers are not provided.

    end_index_ : int
        The index of the end of the range. It is 1 if the wavenumbers are not provided.

    n_features_in_ : int
        The number of features in the input data.

    _is_fitted : bool
        Whether the transformer has been fitted to data.

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the input data.

    transform(X, y=0, copy=True)
        Transform the input data by subtracting the constant baseline value.
    """

    def __init__(
        self, start: int = 0, end: int = 1, wavenumbers: np.ndarray = None,
    ) -> None:
        self.start = start
        self.end = end
        self.wavenumbers = wavenumbers

    def fit(self, X: np.ndarray, y=None) -> "ConstantBaselineCorrection":
        """
        Fit the transformer to the input data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to fit the transformer to.

        y : None
            Ignored.

        Returns
        -------
        self : ConstantBaselineCorrection
            The fitted transformer.
        """
        # Check that X is a 2D array and has only finite values
        X = check_input(X)

        # Set the number of features
        self.n_features_in_ = X.shape[1]

        # Set the fitted attribute to True
        self._is_fitted = True

        # Set the start and end indices
        if self.wavenumbers is None:
            self.start_index_ = self.start
            self.end_index_ = self.end
        else:
            self.start_index_ = self._find_index(self.start)
            self.end_index_ = self._find_index(self.end)

        return self

    def transform(self, X: np.ndarray, y=0, copy=True) -> np.ndarray:
        """
        Transform the input data by subtracting the constant baseline value.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to transform.

        y : int or None, optional
            Ignored.

        copy : bool, optional
            Whether to copy the input data before transforming it.

        Returns
        -------
        X_ : np.ndarray of shape (n_samples, n_features)
            The transformed input data.
        """
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
            mean_baseline = np.mean(x[self.start_index_ : self.end_index_ + 1])
            X_[i, :] = x - mean_baseline
        return X_.reshape(-1, 1) if X_.ndim == 1 else X_

    def _find_index(self, target: float) -> int:
        wavenumbers = np.array(self.wavenumbers)
        return np.argmin(np.abs(wavenumbers - target))

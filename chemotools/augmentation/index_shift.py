from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class IndexShift(TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """
    Shift the spectrum a given number of indices between - shift and + shift drawn
    from a discrete uniform distribution.

    Parameters
    ----------
    shift : float, default=0.0
        Shifts the data by a random integer between -shift and shift.

    random_state : int, default=None
        The random state to use for the random number generator.

    Attributes
    ----------
    n_features_in_ : int
        The number of features in the input data.

    _is_fitted : bool
        Whether the transformer has been fitted to data.

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the input data.

    transform(X, y=0, copy=True)
        Transform the input data by shifting the spectrum.
    """

    def __init__(self, shift: int = 0, random_state: Optional[int] = None):
        self.shift = shift
        self.random_state = random_state

    def fit(self, X: np.ndarray, y=None) -> "IndexShift":
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
        self : IndexShift
            The fitted transformer.
        """
        # Check that X is a 2D array and has only finite values
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        # Set the number of features
        self.n_features_in_ = X.shape[1]

        # Set the fitted attribute to True
        self._is_fitted = True

        # Instantiate the random number generator
        self._rng = np.random.default_rng(self.random_state)

        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Transform the input data by shifting the spectrum.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to transform.

        y : None
            Ignored.

        Returns
        -------
        X_ : np.ndarray of shape (n_samples, n_features)
            The transformed data.
        """
        # Check that the estimator is fitted
        check_is_fitted(self, "_is_fitted")

        # Check that X is a 2D array and has only finite values
        X_ = validate_data(
            self,
            X,
            y="no_validation",
            ensure_2d=True,
            copy=True,
            reset=False,
            dtype=np.float64,
        )

        # Check that the number of features is the same as the fitted data
        if X_.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features but got {X_.shape[1]}"
            )

        # Calculate the standard normal variate
        for i, x in enumerate(X_):
            X_[i] = self._shift_spectrum(x)

        return X_.reshape(-1, 1) if X_.ndim == 1 else X_

    def _shift_spectrum(self, x) -> np.ndarray:
        shift_amount = self._rng.integers(-self.shift, self.shift, endpoint=True)
        return np.roll(x, shift_amount)

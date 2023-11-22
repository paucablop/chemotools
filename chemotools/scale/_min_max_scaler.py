import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted

from chemotools.utils.check_inputs import check_input


class MinMaxScaler(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    """
    A transformer that scales the input data by subtracting the minimum and dividing by
    the difference between the maximum and the minimum. When the use_min parameter is False,
    the data is scaled by the maximum.

    Parameters
    ----------
    use_min : bool, default=True
        The normalization to use. If True, the data is subtracted by the minimum and
        scaled by the maximum. If False, the data is scaled by the maximum.

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the input data.

    transform(X, y=0, copy=True)
        Transform the input data by scaling by the maximum value.
    """

    def __init__(self, use_min: bool = True):
        self.use_min = use_min

    def fit(self, X: np.ndarray, y=None) -> "MinMaxScaler":
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
        self : MinMaxScaler
            The fitted transformer.
        """
        # Check that X is a 2D array and has only finite values
        X = self._validate_data(X)

        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Transform the input data by scaling it.

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
        check_is_fitted(self, "n_features_in_")

        # Check that X is a 2D array and has only finite values
        X = check_input(X)
        X_ = X.copy()

        # Check that the number of features is the same as the fitted data
        if X_.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features but got {X_.shape[1]}"
            )

        # Normalize the data by the maximum value
        if self.use_min:
            X_ = (X_ - np.min(X_, axis=1, keepdims=True)) / (
                np.max(X_, axis=1, keepdims=True) - np.min(X_, axis=1, keepdims=True)
            )

        else:
            X_ = X_ / np.max(X_, axis=1, keepdims=True)

        return X_.reshape(-1, 1) if X_.ndim == 1 else X_

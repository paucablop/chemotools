import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted

from chemotools.utils.check_inputs import check_input


class NormScaler(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    """
    A transformer that scales the input data by the L-norm of the spectrum.

    Parameters
    ----------
    l_norm : int, optional
        The L-norm to use. Default is 2. 

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
        Transform the input data by scaling by the L-norm.
    """
    def __init__(self, l_norm: int = 2):
        self.l_norm = l_norm

    def fit(self, X: np.ndarray, y=None) -> "NormScaler":
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
        self : NormScaler
            The fitted transformer.
        """
        # Check that X is a 2D array and has only finite values
        X = check_input(X)

        # Set the number of features
        self.n_features_in_ = X.shape[1]

        # Set the fitted attribute to True
        self._is_fitted = True

        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Transform the input data by scaling by the L-norm.

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
        X = check_input(X)
        X_ = X.copy()

        # Check that the number of features is the same as the fitted data
        if X_.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features but got {X_.shape[1]}"
            )

        # Normalize the data by the maximum value
        for i, x in enumerate(X_):
            X_[i] = x / np.linalg.norm(x, ord=self.l_norm)

        return X_.reshape(-1, 1) if X_.ndim == 1 else X_

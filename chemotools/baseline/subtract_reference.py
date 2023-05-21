import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted

from chemotools.utils.check_inputs import check_input


class SubtractReference(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    """
    A transformer that subtracts a reference spectrum from the input data.

    Parameters
    ----------
    reference : np.ndarray, optional
        The reference spectrum to subtract from the input data. If None, the original spectrum
        is returned.

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
        Transform the input data by subtracting the reference spectrum.

    _subtract_reference(x)
        Subtract the reference spectrum from a single spectrum.
    """
    def __init__(
        self,
        reference: np.ndarray = None,
    ):
        self.reference = reference

    def fit(self, X: np.ndarray, y=None) -> "SubtractReference":
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
        self : SubtractReference
            The fitted transformer.
        """
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
        """
        Transform the input data by subtracting the reference spectrum.

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
            raise ValueError(f"Expected {self.n_features_in_} features but got {X_.shape[1]}")

        if self.reference is None:
            return X_.reshape(-1, 1) if X_.ndim == 1 else X_

        # Subtract the reference
        for i, x in enumerate(X_):
            X_[i] = self._subtract_reference(x)
        return X_.reshape(-1, 1) if X_.ndim == 1 else X_

    def _subtract_reference(self, x) -> np.ndarray:
        """
        Subtract the reference spectrum from a single spectrum.

        Parameters
        ----------
        x : np.ndarray
            The spectrum to subtract the reference from.
        """
        return x - self.reference_

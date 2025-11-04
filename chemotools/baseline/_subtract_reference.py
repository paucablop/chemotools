"""
The :mod:`chemotools.baseline._subtract_reference` module implements
a reference spectrum subtraction transformer.
"""

# Author: Pau Cabaneros
# License: MIT

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class SubtractReference(TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """
    A transformer that subtracts a reference spectrum from the input data.

    Parameters
    ----------
    reference : np.ndarray, optional, default=None
        The reference spectrum to subtract from the input data. If None, the original spectrum
        is returned.

    Attributes
    ----------
    n_features_in_ : int
        The number of features in the input data.

    reference_ : np.ndarray
        The reference spectrum to subtract from the input data if the reference parameter is not None.

    Examples
    --------
    >>> from chemotools.baseline import SubtractReference
    >>> from chemotools.datasets import load_fermentation_train
    >>> # Load sample data
    >>> X, _ = load_fermentation_train()
    >>> # Convert X to a numpy array
    >>> X = np.array(X)
    >>> # Instantiate the transformer with a reference spectrum
    >>> reference = X[0]
    >>> transformer = SubtractReference(reference=reference)
    SubtractReference()
    >>> transformer.fit(X)
    >>> # Generate baseline-corrected data
    >>> X_corrected = transformer.transform(X)
    """

    _parameter_constraints: dict = {
        "reference": ["array-like", None],
    }

    def __init__(
        self,
        reference: Optional[np.ndarray] = None,
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
            Ignored to align with API.

        Returns
        -------
        self : SubtractReference
            The fitted transformer.
        """
        # Check that X is a 2D array and has only finite values
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )
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
            Ignored to align with API.

        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_features)
            The transformed data.
        """
        # Check that the estimator is fitted
        check_is_fitted(self, "n_features_in_")

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

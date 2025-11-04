"""
The :mod:`chemotools.scatter._multiplicative_scatter_correction` module implements a Multiplicative Scatter Correction transformer.
"""

# Authors: Pau Cabaneros
# License: MIT

from typing import Literal, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils._param_validation import StrOptions


class MultiplicativeScatterCorrection(
    TransformerMixin, OneToOneFeatureMixin, BaseEstimator
):
    """
    Multiplicative scatter correction (MSC) is a preprocessing technique for
    removing scatter effects from spectra. It is based on fitting a linear
    regression model to the spectrum using a reference spectrum. The reference
    spectrum is usually a mean or median spectrum of a set of spectra.

    Parameters
    ----------
    reference : np.ndarray of shape (n_freatures), optional, default=None
        The reference spectrum to use for the correction. If None, the mean
        spectrum will be used. The default is None.

    use_mean : bool, optional, default=True
        Whether to use the mean spectrum as the reference. The default is True.

    use_median : bool, optional, default=False
        Whether to use the median spectrum as the reference. The default is False.

    Attributes
    ----------
    n_features_in_ : int
        The number of features in the training data.

    reference_ : np.ndarray
        The reference spectrum used for the correction.

    Raises
    ------
    ValueError
        If no reference is provided.

    References
    ----------
    [1] Åsmund Rinnan, Frans van den Berg, Søren Balling Engelsen,
        "Review of the most common pre-processing techniques for near-infrared spectra,"
        TrAC Trends in Analytical Chemistry 28 (10) 1201-1222 (2009).

    Examples
    --------
    >>> from chemotools.datasets import load_fermentation_train
    >>> from chemotools.scatter import MultiplicativeScatterCorrection
    >>> # Load sample data
    >>> X, _ = load_fermentation_train()
    >>> # Initialize MultiplicativeScatterCorrection
    >>> msc = MultiplicativeScatterCorrection()
    MultiplicativeScatterCorrection()
    >>> # Fit and transform the data
    >>> X_scaled = msc.fit_transform(X)
    """

    ALLOWED_METHODS = ["mean", "median"]

    _parameter_constraints: dict = {
        "method": [StrOptions({"mean", "median"})],
        "reference": ["array-like", None],
        "weights": ["array-like", None],
    }
    # TODO: Check method is valid in instantiation. Right now it is check on fit because it breaks the scikitlearn check_estimator()

    def __init__(
        self,
        method: Literal["mean", "median"] = "mean",
        reference: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ):
        self.method = method
        self.reference = reference
        self.weights = weights

    def fit(self, X: np.ndarray, y=None) -> "MultiplicativeScatterCorrection":
        """
        Fit the transformer to the input data. If no reference is provided, the
        mean or median spectrum will be calculated from the input data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to fit the transformer to.

        y : None
            Ignored to align with API.

        Returns
        -------
        self : MultiplicativeScatterCorrection
            The fitted transformer.
        """
        # Check that X is a 2D array and has only finite values
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )
        # Check that the length of the reference is the same as the number of features
        if self.reference is not None:
            if len(self.reference) != self.n_features_in_:
                raise ValueError(
                    f"Expected {self.n_features_in_} features in reference but got {len(self.reference)}"
                )

        if self.weights is not None:
            if len(self.weights) != self.n_features_in_:
                raise ValueError(
                    f"Expected {self.n_features_in_} features in weights but got {len(self.weights)}"
                )

        # Set the reference
        if self.reference is not None:
            self.reference_ = np.array(self.reference)
            self.A_ = self._calculate_A(self.reference_)
            self.weights_ = np.array(self.weights)
            return self

        if self.method == "mean":
            self.reference_ = X.mean(axis=0)
            self.A_ = self._calculate_A(self.reference_)
            self.weights_ = np.array(self.weights)
            return self

        elif self.method == "median":
            self.reference_ = np.median(X, axis=0)
            self.A_ = self._calculate_A(self.reference_)
            self.weights_ = np.array(self.weights)
            return self

        else:
            raise ValueError(
                f"Invalid method: {self.method}. Must be one of {self.ALLOWED_METHODS}"
            )

        raise ValueError("No reference was provided")

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Transform the input data by applying the multiplicative scatter
        correction.

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

        # Check that the number of features is the same as the fitted data
        if X_.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features but got {X_.shape[1]}"
            )

        # Calculate the multiplicative signal correction
        if self.weights is None:
            for i, x in enumerate(X_):
                X_[i] = self._calculate_multiplicative_correction(x)
            return X_.reshape(-1, 1) if X_.ndim == 1 else X_

        if self.weights is not None:
            for i, x in enumerate(X_):
                X_[i] = self._calculate_weighted_multiplicative_correction(x)
            return X_.reshape(-1, 1) if X_.ndim == 1 else X_

    def _calculate_weighted_multiplicative_correction(self, x) -> np.ndarray:
        m, c = np.linalg.lstsq(
            np.diag(self.weights_) @ self.A_, x * self.weights_, rcond=None
        )[0]
        return (x - c) / m

    def _calculate_multiplicative_correction(self, x) -> np.ndarray:
        m, c = np.linalg.lstsq(self.A_, x, rcond=None)[0]
        return (x - c) / m

    def _calculate_A(self, reference):
        ones = np.ones(reference.shape[0])
        return np.vstack([reference, ones]).T

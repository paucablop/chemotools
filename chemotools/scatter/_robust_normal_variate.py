"""
The :mod:`chemotools.scatter._robust_normal_variate` module implements the Robust Normal Variate (RNV) transformation.
"""

# Authors: Pau Cabaneros
# License: MIT

import warnings
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils._param_validation import Interval, Real


class RobustNormalVariate(TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """
    A transformer that calculates the robust normal variate of the input data.

    Parameters
    ----------
    percentile : float, optional, default=25
        The percentile to use for the robust normal variate. The value should be
        between 0 and 100. The default is 25.

    epsilon : float, optional, default=1e-10
        A small value added to the denominator to avoid numerical instability
        (division by zero). The default is 1e-10.

    Attributes
    ----------
    n_features_in_ : int
        The number of features in the training data.

    Raises
    ------
    UserWarning
        If the standard deviation of the values below the specified percentile is zero,
        a warning and a small epsilon is added to the denominator to avoid NaNs.

    References
    ----------
    [1] Q. Guo, W. Wu, D.L. Massart.
        "The robust normal variate transform for pattern
        recognition with near-infrared data." doi:10.1016/S0003-2670(98)00737-5

    Examples
    --------
    >>> from chemotools.datasets import load_fermentation_train
    >>> from chemotools.scatter import RobustNormalVariate
    >>> # Load sample data
    >>> X, _ = load_fermentation_train()
    >>> # Initialize RobustNormalVariate
    >>> rnv = RobustNormalVariate()
    RobustNormalVariate()
    >>> # Fit and transform the data
    >>> X_scaled = rnv.fit_transform(X)
    """

    _parameter_constraints: dict = {
        "percentile": [Interval(Real, 0, None, closed="both")],
        "epsilon": [Interval(Real, 0, None, closed="both")],
    }

    def __init__(self, percentile: float = 25, epsilon: float = 1e-10):
        self.percentile = percentile
        self.epsilon = epsilon

    def fit(self, X: np.ndarray, y=None) -> "RobustNormalVariate":
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
        self : RobustNormalVariate
            The fitted transformer.
        """
        # Check that X is a 2D array and has only finite values
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Transform the input data by calculating the standard normal variate.

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

        # Calculate the standard normal variate
        for i, x in enumerate(X_):
            X_[i] = self._calculate_robust_normal_variate(x)

        return X_.reshape(-1, 1) if X_.ndim == 1 else X_

    def _calculate_robust_normal_variate(self, x) -> np.ndarray:
        percentile = np.percentile(x, self.percentile)
        denom = np.std(x[x <= percentile])
        if denom == 0:
            warnings.warn(
                "Denominator is zero in RNV. Adding epsilon to avoid NaNs.",
                UserWarning,
                stacklevel=2,
            )
        return (x - percentile) / (denom + self.epsilon)

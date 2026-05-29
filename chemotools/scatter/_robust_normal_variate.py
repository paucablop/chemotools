"""
The :mod:`chemotools.scatter._robust_normal_variate` module
implements the Robust Normal Variate (RNV) transformation.
"""

# Authors: Pau Cabaneros
# License: MIT

import warnings
from numbers import Integral

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils._param_validation import Interval, Real
from sklearn.utils.validation import check_is_fitted, validate_data

from chemotools._doc_mixin import DocLinkMixin
from chemotools._parallel import parallel_apply_by_rows


class RobustNormalVariate(
    DocLinkMixin, TransformerMixin, OneToOneFeatureMixin, BaseEstimator
):
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

    n_jobs : int, optional, default=1
        Number of parallel jobs used to process samples independently.
        Uses serial execution when set to 1.

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
        "percentile": [Interval(Real, 0, 100, closed="both")],
        "epsilon": [Interval(Real, 0, None, closed="both")],
        "n_jobs": [
            Interval(Integral, None, -1, closed="right"),
            Interval(Integral, 1, None, closed="left"),
        ],
    }

    def __init__(self, percentile: float = 25, epsilon: float = 1e-10, n_jobs: int = 1):
        self.percentile = percentile
        self.epsilon = epsilon
        self.n_jobs = n_jobs

    def __setstate__(self, state: dict) -> None:
        """Restore state while keeping backward compatibility with old pickles."""
        super().__setstate__(state)
        if "n_jobs" not in self.__dict__:
            self.n_jobs = 1

    def _calculate_rnv(self, X_block: np.ndarray) -> tuple[np.ndarray, bool]:
        """Compute RNV transform for a block of rows.

        This helper is intentionally block-oriented so it can be reused later in
        chunked/parallel execution without changing numerical behavior.
        """
        percentile = np.percentile(X_block, self.percentile, axis=1, keepdims=True)
        mask = X_block <= percentile
        denom = np.std(X_block, axis=1, keepdims=True, where=mask)
        transformed = (X_block - percentile) / (denom + self.epsilon)
        return transformed, bool(np.any(denom == 0))

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
        # Validate the input parameters
        self._validate_params()

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

        zero_denom_detected = False

        def block_fn(X_block: np.ndarray) -> np.ndarray:
            nonlocal zero_denom_detected
            transformed, has_zero_denom = self._calculate_rnv(X_block)
            if has_zero_denom:
                zero_denom_detected = True
            return transformed

        # Calculate robust normal variate for this validated block
        X_transformed = parallel_apply_by_rows(
            X_, n_jobs=self.n_jobs, block_fn=block_fn
        )

        if zero_denom_detected:
            warnings.warn(
                "Denominator is zero in RNV. Adding epsilon to avoid NaNs.",
                UserWarning,
                stacklevel=2,
            )

        return X_transformed

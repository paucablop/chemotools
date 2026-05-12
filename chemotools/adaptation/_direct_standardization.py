"""
The :mod:'chemotools.domain_adaptation:DirectStandardization'
module implements a Direct Standardization transformer
"""

# Authors: Ruggero Guerrini
# License: MIT

import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import (
    check_consistent_length,
    check_is_fitted,
    validate_data,
)


class DirectStandardization(TransformerMixin, BaseEstimator):
    """
    Direct Standardization (DS) is a transformer used for domain adaptation (calibration
    transfer) applications. The transformer used least squares to find a linear map from
    the source space to the target space, following the implementation by [1].

    Attributes
    ----------
    T_ : np.ndarray of shape (n_features, n_features)
        Linear transformation matrix mapping source space to target space.

    Raises
    ------
    ValueError
        If X and X_source do not have the same shape.

    Reference
    ---------
    .. [1] Wang, Yongdong., Veltkamp, D. J., & Kowalski, B. R. (1991),
        Multivariate instrument standardization,
        Analytical Chemistry, 63(23), Pages 2750–2756,
        https://doi.org/10.1021/ac00023a016.

    Examples
    --------

    """

    def fit(
        self, X: np.ndarray, y=None, *, X_source: np.ndarray | None = None
    ) -> "DirectStandardization":
        """
        Fit the direct standardization of new X data  model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data from the target instrument.

        y : None
            Ignored to align with API.

        X_source : np.ndarray of shape (n_samples, n_features), optional
            Target data. Overrides the value provided at initialization.

        Returns
        -------
        self : DirectStandardization
        """
        # Check that X is a 2D array and has only finite values
        X = validate_data(self, X, ensure_2d=True, reset=True, dtype=np.float64)

        # Check that X_target is not None
        if X_source is None:
            X_source = np.eye(X.shape[0], X.shape[1])
            warnings.warn(
                "Input X_source is None, defaulting to identity matrix with X shape"
            )

        # Check that X_target is a 2D array and has only finite values
        X_source = validate_data(
            self, X_source, ensure_2d=True, reset=True, dtype=np.float64
        )

        # Check consistency in between X and X_target
        check_consistent_length(X, X_source)

        self.T_, _, _, _ = np.linalg.lstsq(X, X_source, rcond=None)

        return self

    def transform(self, X) -> np.ndarray:
        """
        Transform the data from the target space to the source space using the map
        self.T_

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to transform

        Returns
        -------
        X_transf : np.ndarray of shape (n_samples, n_features)
            The data transformed
        """
        # Check that the estimator is fitted

        check_is_fitted(self, ["T_"])

        # Validate the input data
        X = validate_data(
            self,
            X,
            ensure_2d=True,
            reset=False,
            dtype=np.float64,
        )

        # Apply the transformation
        return X @ self.T_

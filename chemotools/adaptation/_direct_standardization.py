"""
The :mod:'chemotools.domain_adaption:DirectStandardization'
module implements a Direct Standardization transformer
"""

# Authors: Ruggero Guerrini
# License: MIT

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class DirectStandardization(TransformerMixin, BaseEstimator):
    """
    Implement a direct standardization transformer for the calibration
    transfer application.
    y contains the reference measurements acquired
    on the target instrument.
    X contains the corresponding measurements of the same samples
    acquired on the source instrument.
    The transformer estimates a mapping from the source space to
    the target space.
    After fitting, new X spectra can be transformed into
    the y space.

    Parameters
    ----------
    X_target : np.ndarray of shape (n_samples, n_features)
        Data for which the transformation matrix is calculated
    Attributes
    ----------
    T : np.ndarray of shape (n_features, n_features)
        The pxp matrix that solver the problem X T = y
        using the method of least squares

    Reference
    ---------
    .. [1] Wang, Yongdong., Veltkamp,
        D. J., & Kowalski, B. R. (1991).
        Multivariate instrument standardization.
        Analytical Chemistry, 63(23), 2750–2756.

    Examples
    --------
    **Basic usage**
    >>> # Import necessary libraries
    >>> from chemotools.adaptation import DirectStandardization
    >>> import numpy as np
    >>>
    >>> # Generate sample data
    >>> rng = np.random.default_rng(17)
    >>> y = rng.normal(size=(100, 20))
    >>> # Train the model
    >>> DS = DirectStandardization().fit(X, y)
    >>> # Apply to a new set of data
    >>> X_transf = DS.transform(X_new)

    """

    def __init__(self, X_target: np.ndarray | None = None):

        self.X_target = X_target

    def fit(self, X: np.ndarray, y=None) -> "DirectStandardization":
        """
        Fit the DirectStandardization to the input data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The source data


        Returns
        -------
        self : DirectStandardization
            The fitted model.
        """

        # validate_data
        X = validate_data(self, X, ensure_2d=True, reset=True, dtype=np.float64)

        if self.X_target is None:
            X_target = X.copy()
        else:
            X_target = self.X_target

        self.T_ = np.linalg.pinv(X) @ X_target
        return self

    def transform(self, X) -> np.ndarray:
        """
        Transform the source data

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to transform

        Returns
        -------
        X_transf : np.ndarray of shape (n_samples, n_features)
            The data transformed
        """
        # Validate input data
        check_is_fitted(self, ["T_"])
        X = validate_data(
            self,
            X,
            ensure_2d=True,
            reset=False,
            dtype=np.float64,
        )
        # Apply the transformation
        return X @ self.T_

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
    None

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

    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> "DirectStandardization":
        """
        Fit the DirectStandardization to the input data.
        Use always both "X" and "y", that must share the same
        dimensions.
        IMPORTANT
        To preserve the compatibility with scikitlearn, the case
        where y is None or its shape are not equal to X are accepted.
        In this case the T matrix will be an identity matrix.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The source data
        y : np.ndarray of shape (n_samples, n_features)
            The target data

        Returns
        -------
        self : DirectStandardization
            The fitted model.
        """

        # validate_data
        X = validate_data(self, X, ensure_2d=True, reset=True, dtype=np.float64)

        # To protect from .ndim
        if y is not None:
            y = np.asarray(y)

        # Real case
        if y is not None and y.ndim == 2:
            _, y = validate_data(
                self,
                X,
                y,
                ensure_2d=True,
                reset=False,  # reset=False perché X già validato
                dtype=np.float64,
                multi_output=True,
            )
            self.T_ = np.linalg.pinv(X) @ y
            return self

        # If y is not set, T will be an Identity matrix.
        # It was done to preserve the compatibility with scikitlearn
        self.T_ = np.eye(X.shape[1])
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

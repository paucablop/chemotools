"""
The :mod:`chemotools.cross_decomposition._external_parameter_orthogonalization` module
implements the External Parameter Orthogonalization (EPO) technique for preprocessing
spectral data by removing variations orthogonal to the external parameters.
"""

from numbers import Integral, Real
from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data


class ExternalParameterOrthogonalization(TransformerMixin, BaseEstimator):
    """

    Parameters
    ----------
    n_components : int, default=2
        Number of orthogonal components to remove. Must be a positive integer.



    max_iter : int, default=500
        Maximum number of iterations for the component calculation algorithms.

    tol : float, default=1e-06
        Tolerance for convergence in the iterative algorithms.

    copy : bool, default=True
        Whether to copy X and Y in fit before applying centering.

    Attributes
    ----------
    mean_X_ : ndarray of shape (n_features,)
        The mean of the features in the training data.

    mean_y_ : float or ndarray of shape (n_targets,)
        The mean of the target variable(s) in the training data.

    scores_ : ndarray of shape (n_samples, n_components)
        The scores of the orthogonal components.

    weights_ : ndarray of shape (n_features, n_components)
        The weights of the orthogonal components.

    loadings_ : ndarray of shape (n_features, n_components)
        The loadings of the orthogonal components.

    n_iter_ : ndarray of shape (n_components,)
        The number of iterations taken for each component to converge.


    References
    ----------
    .. [1] Jean-Michel Roger*, Fabien Chauchard, Ve ́ronique Bellon-Maurel (2003),
        EPO–PLS external parameter orthogonalisation of PLS application to
        temperature-independent measurement of sugar content of intact fruits,
        Chemometrics and Intelligent Laboratory Systems,
        Volume 66, Issue 2, Pages 191-204,
        https://doi.org/10.1016/S0169-7439(03)00051-0

    Examples
    --------
    **Basic usage with automatic variance calculation**
    >>> import numpy as np

    Notes
    -----


    See Also
    --------

    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "method": [StrOptions({"wold", "sjoblom", "fearn"})],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left")],
        "copy": ["boolean"],
    }

    def __init__(
        self,
        n_components: int = 2,
        method: Literal["wold", "sjoblom", "fearn"] = "wold",
        max_iter: int = 500,
        tol: float = 1e-06,
        copy: bool = True,
    ):
        """Initialize the External Parameter Orthogonalization (EPO) transformer.

        Parameters
        ----------
        n_components : int, default=2
            Number of orthogonal components to remove. Must be a positive integer.
        copy : bool, default=True
            Whether to copy X and Y in fit before applying centering.
        """
        self.n_components = n_components
        self.method = method
        self.max_iter = max_iter
        self.tol = tol
        self.copy = copy


def fit(
    self, X: np.ndarray, y: np.ndarray, X_external: np.ndarray
) -> "ExternalParameterOrthogonalization":
    self._validate_params()

    # Validate X and X_external
    X, y = validate_data(
        self, X, y, ensure_2d=True, copy=self.copy, dtype=np.float64, multi_output=True
    )
    X_ext = validate_data(
        self, X_external, ensure_2d=True, copy=False, dtype=np.float64
    )

    if X_ext.shape[1] != X.shape[1]:
        raise ValueError("X_external must have the same number of features as X.")

    # 1. Center the external variation matrix
    # Note: Roger et al. often use SVD on the raw difference matrix D,
    # but centering is fine if X_external represents the variation.
    self.mean_X_ = np.mean(X, axis=0)
    X_ext_centered = X_ext - np.mean(X_ext, axis=0)

    # 2. SVD to find the nuisance subspace
    _, _, Vt = np.linalg.svd(X_ext_centered, full_matrices=False)

    # 3. V contains the principal directions of the nuisance variation
    V = Vt[: self.n_components, :].T

    # 4. P = I - V(V^T V)^-1 V^T. Since V is orthonormal from SVD, V^T V = I
    identity = np.eye(X.shape[1])
    self.P_epo_ = identity - (V @ V.T)

    return self


def transform(self, X: np.ndarray):
    check_is_fitted(self)
    X = validate_data(self, X, reset=False)

    # Correct Mathematical Projection: (X - mu) @ P
    return (X - self.mean_X_) @ self.P_epo_

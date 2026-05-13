"""
The :mod:`chemotools.projection._direct_orthogonalization` module implements the Direct
Orthogonalization (DO) technique for preprocessing spectral data by removing variations
orthogonal to the target variable.
"""

# Author: Pau Cabaneros
# License: MIT

from numbers import Integral

import numpy as np
from scipy.linalg import svd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted, validate_data


class DirectOrthogonalization(TransformerMixin, BaseEstimator):
    """
    Remove variation in X that is uncorrelated with the target y using Direct
    Orthogonalization (DO) [1]_ [2]_.

    DO removes from X systematic variation that is independent of y. X is
    orthogonalized with respect to y, PCA is performed on the orthogonalized matrix
    to estimate orthogonal components, and those components are subtracted from X to
    obtain the corrected data.

    The transformer returns the corrected matrix with the same number of features,
    retaining variation relevant for predicting y. Inputs are typically assumed to
    be mean-centered.

    Parameters
    ----------
    n_components : int, default=1
        The number of orthogonal components to compute. This determines how many
        orthogonal variations will be removed from the data.

    copy : bool, default=False
        If True, a copy of the input data is created and used for computations.
        If False, the input data is modified in place.

    Attributes
    ----------
    x_weights_orth_ : ndarray of shape (n_features, n_components)
        The weights of the orthogonal components.

    x_loadings_orth_ : ndarray of shape (n_features, n_components)
        The loadings of the orthogonal components.

    x_scores_orth_ : ndarray of shape (n_samples, n_components)
        The scores of the orthogonal components.

    mean_X_ : ndarray of shape (n_features,)
        The mean of the original data `X` used for centering.

    mean_y_ : float or ndarray of shape (n_targets,)
        The mean of the target variable `y` used for centering.

    retained_variance_ratio_ : float
        The proportion of variance in `X` retained explained by the predictive
        components.

    removed_variance_ratio_ : float
        The proportion of variance in `X` removed explained by the orthogonal
        components.


    References
    ----------
    .. [1] Clause A. Andersson, (1999).
        Direct orthogonalization.
        Chemometrics and Intelligent Laboratory Systems,
        Volume 47, Issue 1, Pages 51-63,
        https://doi.org/10.1016/S0169-7439(98)00158-0.

    .. [2] O. Svensson, T. Kourti, J. F. MacGregor (2002).
        An investigation of orthogonal signal correction algorithms and their
        characteristics.
        Journal of Chemometrics,
        Volume 16, Issue 4, Pages 176-188,
        https://doi.org/10.1002/cem.700

    Examples
    --------
        Fit and apply DirectOrthogonalization to remove variation in `X` that is
        orthogonal to `y`.

    >>> import numpy as np
    >>> from chemotools.projection import DirectOrthogonalization
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y = np.array([1, 2, 3])
    >>> do = DirectOrthogonalization(n_components=1)
    >>> do.fit(X, y)
    DirectOrthogonalization(n_components=1, copy=False)
    >>> X_transformed = do.transform(X, y)
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "copy": ["boolean"],
    }

    def __init__(self, n_components: int = 1, copy=False):
        """Initialize the DirectOrthogonalization transformer.

        Parameters
        ----------
        n_components : int, default=1
            The number of orthogonal components to compute. This determines how many
            orthogonal variations will be removed from the data.

        copy : bool, default=False
            If True, a copy of the input data is created and used for computations.
            If False, the input data is modified in place.
        """
        self.n_components = n_components
        self.copy = copy

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DirectOrthogonalization":
        """Fit the DirectOrthogonalization model to the training data.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to fit the model to.

        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : DirectOrthogonalization
            Fitted estimator.
        """
        # Check that X is a 2D array and has only finite values
        self._validate_params()
        X, y = validate_data(
            self,
            X,
            y=y,
            ensure_2d=True,
            reset=True,
            copy=self.copy,
            dtype=np.float64,
            multi_output=True,
        )
        y = np.asarray(y, dtype=np.float64)

        # Validate that there are at least 2 samples
        n_samples = X.shape[0]
        if n_samples < 2:
            raise ValueError(
                "n_samples=1 is not enough for direct orthogonalization. "
                "At least 2 samples are required."
            )

        # Center the data
        self.mean_X_ = np.mean(X, axis=0)
        self.mean_y_ = np.mean(y, axis=0) if y.ndim == 2 else np.mean(y)
        X_centered = X - self.mean_X_
        y_centered = y - self.mean_y_

        yk = y_centered.reshape(-1, 1) if y_centered.ndim == 1 else y_centered

        # Calculate total sum of squares for X
        total_ss_x = np.sum(X_centered**2)
        if total_ss_x == 0:
            raise ValueError(
                "X has zero variance after mean-centering. OrthogonalPLS requires "
                "X to contain at least some non-constant features."
            )

        # Step 1: Orthogonalize X with respect to y using regression (Step 2 in [1])
        # Generalization to multivariate y. coef_yx is w in [1].
        # This is equivalent to:
        # coef_yx = np.linalg.pinv(yk.T @ yk) @ yk.T @ X_centered,
        # but using lstsq is more numerically stable and handles rank-deficient cases.
        coef_yx, _, _, _ = np.linalg.lstsq(yk, X_centered, rcond=None)

        # Step 2: Deflate X by removing the variation explained by y (Step 2 in [1])
        Xk = X_centered - yk @ coef_yx

        # Step 3: SVD of the deflated Xk to get orthogonal components (Step 3 in [1])
        _, _, Vt = svd(Xk, full_matrices=False)
        self.x_loadings_orth_ = Vt[: self.n_components].T

        # Check that n_components is not greater than the maximum possible components
        max_components = Vt.shape[0]
        if self.n_components > max_components:
            raise ValueError(
                "Number of components must be less than or equal to the number "
                "of features."
            )

        # Step 4: Calculate orthogonal scores (Step 4.1 in [1])
        self.x_scores_orth_ = np.dot(X_centered, self.x_loadings_orth_)

        # Step 5: Orthogonal weights are the same as loadings (Table 1 in [2])
        self.x_weights_orth_ = self.x_loadings_orth_

        # Step 6: Deflate X globally (vectorized version of Step 4.2 in [1])
        X_deflated = X_centered - np.dot(self.x_scores_orth_, self.x_loadings_orth_.T)

        # Step 7: Calculate sum of squares in defleated Xk
        total_ss_x_k = np.sum(X_deflated**2)

        # Step 8: Calculate variance ratio in prediction matrix
        self.retained_variance_ratio_ = total_ss_x_k / total_ss_x
        self.removed_variance_ratio_ = 1 - self.retained_variance_ratio_

        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Apply the Direct Orthogonalization (D) correction to X

        This returns the predictive part of the data, i.e. the variation in X that is
        related to y, after removing the orthogonal part (variation in X that is not
        related to y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to transform.

        y : None
            Ignored to align with API.

        Returns
        -------
        X_transformed : array-like of shape (n_samples, n_features)
            The transformed data.
        """
        # Check that the estimator is fitted
        check_is_fitted(self, "x_weights_orth_")

        # Validate input data
        X = validate_data(
            self,
            X,
            y="no_validation",
            ensure_2d=True,
            reset=False,
            copy=self.copy,
            dtype=np.float64,
        )

        # Mean center the new data
        Xc = X - self.mean_X_

        # Transform the data
        t_ortho = np.dot(Xc, self.x_weights_orth_)
        Xc -= np.dot(t_ortho, self.x_loadings_orth_.T)

        return Xc + self.mean_X_

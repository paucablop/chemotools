"""
The :mod:`chemotools.projection._orthogonal_pls` module implements the Orthogonal
Projection to Latent Structures (OPLS) technique for preprocessing spectral data by
removing variations orthogonal to the target variable.
"""

# Author: Pau Cabaneros
# License: MIT

from numbers import Integral

import numpy as np
from scipy.linalg import svd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted, validate_data


class OrthogonalPLS(TransformerMixin, BaseEstimator):
    """
    A transformer that removes variation in X that is orthogonal to the target y using
    Orthogonal Projection to Latent Structures (OPLS) [1]_.

    OPLS extends PLS by explicitly separating X into predictive variation,
    correlated with y, and orthogonal variation, unrelated to y. At each
    iteration, a predictive weight vector is estimated using the PLS criterion
    (maximizing covariance between X and y). Scores and loadings are then
    computed, and the loading vector is decomposed into a component aligned
    with the predictive weight and a component orthogonal to it. The orthogonal
    component defines an orthogonal score vector, which is used to deflate X.

    This procedure is repeated to remove multiple orthogonal components while
    retaining the predictive structure. Multivariate targets are supported via
    decomposition of the cross-covariance matrix.

    The transformer returns X with orthogonal variation removed, preserving the
    original number of features.

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
    x_weights_ : ndarray of shape (n_features, n_components)
        The weights of the original components.

    x_weights_orth_ : ndarray of shape (n_features, n_components)
        The weights of the orthogonal components.

    x_loadings_ : ndarray of shape (n_features, n_components)
        The loadings of the original components.

    x_loadings_orth_ : ndarray of shape (n_features, n_components)
        The loadings of the orthogonal components.

    x_scores_ : ndarray of shape (n_samples, n_components)
        The scores of the original components.

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
    .. [1] Trygg, J., & Wold, S. (2002).
        Orthogonal projections to latent structures (O-PLS).
        Journal of Chemometrics, Volume 16, Issue 3, Pages 119-128,
        https://doi.org/10.1002/cem.695.

    Examples
    --------
        Fit and apply OrthogonalPLS to remove variation in `X` that is orthogonal to
        `y`.

    >>> import numpy as np
    >>> from chemotools.projection import OrthogonalPLS
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y = np.array([1, 2, 3])
    >>> opls = OrthogonalPLS(n_components=1)
    >>> opls.fit(X, y)
    OrthogonalPLS(n_components=1, copy=False)
    >>> X_transformed = opls.transform(X, y)
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "copy": ["boolean"],
    }

    def __init__(self, n_components: int = 1, copy=False):
        """Initialize the OrthogonalPLS transformer.

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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OrthogonalPLS":
        """Fit the OorthogonalPLS model to the training data.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to fit the model to.

        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : OrthogonalPLS
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
                "n_samples=1 is not enough for orthogonal signal correction. "
                "At least 2 samples are required."
            )

        # Center the data
        self.mean_X_ = np.mean(X, axis=0)
        self.mean_y_ = np.mean(y, axis=0) if y.ndim == 2 else np.mean(y)
        X_centered = X - self.mean_X_
        y_centered = y - self.mean_y_

        # Get the dimensions
        n = X.shape[0]
        p = X.shape[1]

        # Mean center and optionally scale the data
        Xk = X_centered.copy()
        yk = y_centered.copy()
        yk = yk.reshape(-1, 1)

        # Allocate scores and weights
        self.x_weights_ = np.zeros((p, self.n_components))  # w in [1]
        self.x_weights_orth_ = np.zeros((p, self.n_components))  # w_ortho in [1]
        self.x_loadings_ = np.zeros((p, self.n_components))  # p in [1]
        self.x_loadings_orth_ = np.zeros((p, self.n_components))  # p_ortho in [1]
        self.x_scores_ = np.zeros((n, self.n_components))  # t in [1]
        self.x_scores_orth_ = np.zeros((n, self.n_components))  # t_ortho in [1]

        # Calculate total sum of squares for X
        total_ss_x = np.sum(Xk**2)

        # For each component
        for k in range(self.n_components):
            # Step 1: Weights are calculated through SVD to support multi y (Step 1
            # in [1])
            # Step 1.1. Calculate covariance matrix (C)
            C = np.dot(Xk.T, yk)

            # Step 1.2: Calculate the SVD of C
            U, _, _ = svd(C, full_matrices=True)

            # Step 1.3: We just use the first weight
            x_weights = U[:, 0]

            # Step 2. Normalize the weights (Step 2 in [1])
            x_weights /= np.linalg.norm(x_weights)

            # Step 3: Calculate the x_scores (Step 3 in [1])
            x_scores = np.dot(Xk, x_weights) / np.dot(x_weights.T, x_weights)

            # Step 4: Calculate the x_loadings (Step 6 in [1])
            x_loadings = np.dot(x_scores.T, Xk) / np.dot(x_scores.T, x_scores)
            x_loadings = x_loadings.T

            # Step 5: Calculate orthogonal x weights (Step 7 in [1])
            x_weights_orth = (
                x_loadings
                - (np.dot(x_weights.T, x_loadings) / np.dot(x_weights.T, x_weights))
                * x_weights
            )

            # Step 6: Normalize the orthogonal weights (Step 8 in [1])
            x_weights_orth /= np.linalg.norm(x_weights_orth)

            # Step 7: Calculate orthogonal x scores (Step 9 in [1])
            x_scores_orth = np.dot(Xk, x_weights_orth) / np.dot(
                x_weights_orth.T, x_weights_orth
            )

            # Step 7: Calculate orthogonal x loadings (Step 10 in [1])
            x_loadings_orth = np.dot(x_scores_orth.T, Xk) / np.dot(
                x_scores_orth.T, x_scores_orth
            )

            # Step 8: Deflation of X matrix (Step 11 in [1])
            Xk -= np.outer(x_scores_orth, x_loadings_orth)

            # Step 9: Collect the variables
            self.x_weights_[:, k] = x_weights
            self.x_weights_orth_[:, k] = x_weights_orth
            self.x_loadings_[:, k] = x_loadings
            self.x_loadings_orth_[:, k] = x_loadings_orth
            self.x_scores_[:, k] = x_scores
            self.x_scores_orth_[:, k] = x_scores_orth

        # Step 10: Calculate sum of squares in defleated Xk
        total_ss_x_k = np.sum(Xk**2)

        # Step 11: Calculate variance ratio in prediction matrix
        self.retained_variance_ratio_ = total_ss_x_k / total_ss_x
        self.removed_variance_ratio_ = 1 - self.retained_variance_ratio_

        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Apply the OrthoghonalPLS correction to X

        This returns the predictinve part of the data, i.e. the variation in X that is
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

        # Return the transformed data
        for k in range(self.n_components):
            # Calculate scores for the NEW data using learned weights
            scores_orth = np.dot(Xc, self.x_weights_orth_[:, k])
            # Subtract the learned loading pattern
            Xc -= np.outer(scores_orth, self.x_loadings_orth_[:, k])

        return Xc + self.mean_X_

"""
The :mod:`chemotools.cross_decomposition._orthogonal_signal_correction` module
implements the Orthogonal Signal Correction (OSC) technique for preprocessing
spectral data by removing variations orthogonal to the target variable.
"""

import warnings
from typing import Literal

import numpy as np
from scipy.linalg import pinv
from scipy.sparse.linalg import svds
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class OrthogonalSignalCorrection(TransformerMixin, BaseEstimator):
    """

    Parameters
    ----------
    n_components : int, default=2
        Number of orthogonal components to remove. Must be a positive integer.
    method : {'wold', 'sjoblom', 'fearn'}, default='wold'
        Method for calculating orthogonal components:
        - 'wold': Original method by Wold et al. (1998) [1]_
        - 'sjoblom': Method by Sjöblom et al. (1998) [2]_
        - 'fearn': Method by Fearn (2000) [3]_

    copy : bool, default=True
        Whether to copy X and Y in fit before applying centering.

    Attributes
    ----------


    References
    ----------
    .. [1] Svante Wold, Henrik Antti, Fredrik Lindgren, Jerker Öhman (1998),
        Orthogonal signal correction of near-infrared spectra,
        Chemometrics and Intelligent Laboratory Systems,
        Volume 44, Issues 1–2, Pages 175-185,
        https://doi.org/10.1016/S0169-7439(98)00109-9.

    .. [2] Jonas Sjöblom, Olof Svensson, Mats Josefson, Hans Kullberg, Svante Wold
        (1998),
        An evaluation of orthogonal signal correction applied to calibration transfer of
        near infrared spectra,
        Chemometrics and Intelligent Laboratory Systems,
        Volume 44, Issues 1–2, Pages 229-244,
        https://doi.org/10.1016/S0169-7439(98)00112-9.

    .. [3] Tom Fearn (2000),
        On orthogonal signal correction,
        Chemometrics and Intelligent Laboratory Systems,
        Volume 50, Issue 1, Pages 47-52,
        https://doi.org/10.1016/S0169-7439(99)00045-3.

    Examples
    --------
    **Basic usage with automatic variance calculation**
    >>> import numpy as np

    Notes
    -----


    See Also
    --------

    """

    def __init__(
        self,
        n_components: int = 2,
        method: Literal["wold", "sjoblom", "fearn"] = "wold",
        max_iter: int = 500,
        tol: float = 1e-06,
        copy: bool = True,
    ):
        """Initialize the Orthogonal Signal Correction (OSC) transformer.

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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OrthogonalSignalCorrection":
        """Fit the OSC model to calculate the orthogonal components to remove.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors. Accepts numpy arrays, pandas DataFrames.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target vectors. Accepts 1D (univariate) or 2D (multivariate) targets.

        Returns
        -------
        self : OrthogonalSignalCorrection
            Fitted OSC model with calculated orthogonal components.
        """
        # Check that X is a 2D array and has only finite values
        X, y = validate_data(X, y=y, ensure_2d=True, reset=True, dtype=np.float64)

        # Center the data
        self.mean_X_ = np.mean(X, axis=0)
        X_centered = X - self.mean_X_

        # Call parent fit method
        if self.method == "wold":
            self._wold_method(X_centered, y)

        # Calculate the projection matrix for transforming new data
        W, P = self.weights_, self.loadings_
        self.projection_matrix_ = W @ pinv(P.T @ W) @ P.T
        return self

    def transform(self, X: np.ndarray, y: np.ndarray | None = None, copy: bool = True):
        """Apply dimensionality reduction to X.

        Projects X onto the latent components found during fitting.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform.
        y : array-like of shape (n_samples,) or (n_samples, n_targets), optional
            Target vectors. Only used to transform Y when provided.
        copy : bool, default=True
            Whether to copy X and Y, or perform in-place normalization.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
            X transformed with removed orthogonal variation.
        """
        # Check that the estimator is fitted
        check_is_fitted(self, "n_features_in_")

        # Validate input data
        X = validate_data(
            X, y="no_validation", ensure_2d=True, copy=copy, dtype=np.float64
        )

        Xc = X - self.mean_X_
        return Xc - Xc @ self.projection_matrix_

    def _wold_method(self, X: np.ndarray, y: np.ndarray):
        """Calculate orthogonal components using Wold's method."""
        # Calculate the first singular vectors of X
        Xk = X.copy()
        y = np.asarray(y)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Get the features and components
        n_samples, n_features = X.shape

        # Precompute the projection matrix part
        y_pinv = pinv(y)

        self.scores_ = np.zeros((n_samples, self.n_components))
        self.weights_ = np.zeros((n_features, self.n_components))
        self.loadings_ = np.zeros((n_features, self.n_components))

        for k in range(self.n_components):
            # Calculate the first singular vectors of Xk
            U, _, Vt = svds(Xk, k=1, which="LM")
            t = Xk @ Vt.T

            # Initial orthogonalization
            t_star = t - y @ (y_pinv @ t)

            for iteration in range(self.max_iter):
                # Weight calculation (NIPALS step)
                w = Xk.T @ t_star / (t_star.T @ t_star)
                w /= np.linalg.norm(w)

                # Recalculate the scores using w
                t_new = Xk @ w
                t_new_star = t_new - y @ (y_pinv @ t_new)

                # Vectorized convergence check
                if (
                    np.linalg.norm(t_new_star - t_star) / np.linalg.norm(t_star)
                    < self.tol
                ):
                    break

                t_star = t_new_star

            if iteration == self.max_iter:
                warnings.warn(
                    f"Warning: Wold method did not converge after {self.max_iter} "
                    f"iterations."
                )

            # Calculate the loadings p
            p = Xk.T @ t_star / (t_star.T @ t_star)

            # Store the scores, weights and loadings
            self.scores_[:, k] = t_star.flatten()
            self.weights_[:, k] = w.flatten()
            self.loadings_[:, k] = p.flatten()

            # Deflate Xk by removing the contribution of the orthogonal component
            Xk -= t_star @ p.T

        return Xk

    def _sjoblom_method(self, X: np.ndarray, y: np.ndarray):
        """Calculate orthogonal components using Sjöblom's method."""
        # Placeholder for Sjöblom's method implementation
        pass

    def _fearn_method(self, X: np.ndarray, y: np.ndarray):
        """Calculate orthogonal components using Fearn's method."""
        # Placeholder for Fearn's method implementation
        pass

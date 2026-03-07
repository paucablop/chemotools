"""
The :mod:`chemotools.cross_decomposition._orthogonal_signal_correction` module
implements the Orthogonal Signal Correction (OSC) technique for preprocessing
spectral data by removing variations orthogonal to the target variable.
"""

import warnings
from numbers import Integral, Real
from typing import Literal, Tuple

import numpy as np
from scipy.linalg import pinv, svd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted


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
    projection_matrix_ : ndarray of shape (n_features, n_features)
        The projection matrix used to remove orthogonal variation from new data.

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
        X, y = self.validate_data(X, y=y, ensure_2d=True, reset=True, dtype=np.float64)  # type: ignore[unresolved-attribute]

        # Center the data
        self.mean_X_ = np.mean(X, axis=0)
        self.mean_y_ = np.mean(y, axis=0) if y.ndim == 2 else np.mean(y)
        X_centered = X - self.mean_X_
        y_centered = y - self.mean_y_

        # Call parent fit method
        if self.method == "wold":
            self.scores_, self.weights_, self.loadings_ = self._wold_method(
                X_centered, y_centered
            )

        # Calculate the projection matrix for transforming new data
        W, P = self.weights_, self.loadings_
        self.projection_matrix_ = W @ pinv(P.T @ W) @ P.T
        return self

    def transform(self, X: np.ndarray, y = None, copy: bool = True):
        """Apply dimensionality reduction to X.

        Projects X onto the latent components found during fitting.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform.

        y : None
            Ignored to align with API.

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
        X = self.validate_data(  # type: ignore[unresolved-attribute]
            X, y="no_validation", ensure_2d=True, copy=copy, dtype=np.float64
        )

        Xc = X - self.mean_X_

        for k in range(self.n_components):
            Xc -= np.outer(Xc @ self.weights_[:, k], self.loadings_[:, k])

        return Xc + self.mean_X_

    def _wold_method(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

        scores_ = np.zeros((n_samples, self.n_components))
        weights_ = np.zeros((n_features, self.n_components))
        loadings_ = np.zeros((n_features, self.n_components))

        for k in range(self.n_components):
            # Calculate the first singular vectors of Xk
            _, _, Vt = svd(Xk, full_matrices=False)
            t = Xk @ Vt.T[:, 0]

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
                    f"Wold method did not converge after {self.max_iter} iterations.",
                    ConvergenceWarning,
                )

            # Calculate the loadings p
            p = Xk.T @ t_star / (t_star.T @ t_star)

            # Store the scores, weights and loadings
            scores_[:, k] = t_star.flatten()
            weights_[:, k] = w.flatten()
            loadings_[:, k] = p.flatten()

            # Deflate Xk by removing the contribution of the orthogonal component
            Xk -= t_star @ p.T

        return scores_, weights_, loadings_

    def _sjoblom_method(self, X: np.ndarray, y: np.ndarray):
        """Calculate orthogonal components using Sjöblom's method."""
        # Placeholder for Sjöblom's method implementation
        pass

    def _fearn_method(self, X: np.ndarray, y: np.ndarray):
        """Calculate orthogonal components using Fearn's method."""
        # Placeholder for Fearn's method implementation
        pass

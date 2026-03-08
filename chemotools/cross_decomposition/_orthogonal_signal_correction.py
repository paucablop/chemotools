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

    projection_matrix_ : ndarray of shape (n_features, n_features)
        The projection matrix used to remove orthogonal variation from new data.

    n_iter_ : ndarray of shape (n_components,)
        The number of iterations taken for each component to converge.


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

        # Center the data
        self.mean_X_ = np.mean(X, axis=0)
        self.mean_y_ = np.mean(y, axis=0) if y.ndim == 2 else np.mean(y)
        X_centered = X - self.mean_X_
        y_centered = y - self.mean_y_

        # Call parent fit method
        if self.method == "wold":
            self.scores_, self.weights_, self.loadings_, self.n_iter_ = (
                self._wold_method(X_centered, y_centered)
            )

        if self.method == "sjoblom":
            self.scores_, self.weights_, self.loadings_, self.n_iter_ = (
                self._sjoblom_method(X_centered, y_centered)
            )

        # Calculate the projection matrix for transforming new data
        # W, P = self.weights_, self.loadings_
        # self.projection_matrix_ = W @ pinv(P.T @ W) @ P.T
        return self

    def transform(self, X: np.ndarray, y=None):
        """Apply dimensionality reduction to X.

        Projects X onto the latent components found during fitting.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform.

        y : None
            Ignored to align with API.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
            X transformed with removed orthogonal variation.
        """
        # Check that the estimator is fitted
        check_is_fitted(self, "n_features_in_")

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

        Xc = X - self.mean_X_

        for k in range(self.n_components):
            Xc -= np.outer(Xc @ self.weights_[:, k], self.loadings_[:, k])

        return Xc + self.mean_X_

    def _wold_method(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate orthogonal components using Wold's method."""
        # Initialize variables
        Xk = X.copy()
        y = np.asarray(y)

        y = y.reshape(-1, 1) if y.ndim == 1 else y

        # Get the features and components
        n_samples, n_features = X.shape

        # Precompute the projection matrix part
        y_pinv = pinv(y)

        scores = np.zeros((n_samples, self.n_components))
        weights = np.zeros((n_features, self.n_components))
        loadings = np.zeros((n_features, self.n_components))
        n_iter = np.zeros(self.n_components, dtype=int)

        for k in range(self.n_components):
            # Calculate the first singular vectors of Xk
            _, _, Vt = svd(Xk, full_matrices=False)
            t = Xk @ Vt.T[:, 0]

            # Initial orthogonalization
            t_star = t - y @ (y_pinv @ t)

            for iteration in range(self.max_iter):
                # Weight calculation (NIPALS step)
                t_star_norm_sq = t_star.T @ t_star
                if np.isclose(t_star_norm_sq, 0.0):
                    raise ValueError(
                        "Wold method encountered a zero-norm orthogonal score vector."
                    )
                w = Xk.T @ t_star / t_star_norm_sq
                w_norm = np.linalg.norm(w)
                if np.isclose(w_norm, 0.0):
                    raise ValueError(
                        "Wold method encountered a zero-norm weight vector."
                    )
                w /= w_norm

                # Recalculate the scores using w
                t_new = Xk @ w
                t_new_star = t_new - y @ (y_pinv @ t_new)

                # Vectorized convergence check
                if (
                    np.linalg.norm(t_new_star - t_star)
                    / max(np.linalg.norm(t_star), np.finfo(float).eps)
                    < self.tol
                ):
                    t_star = t_new_star
                    break
                t_star = t_new_star

            # Update w for the final iteration
            t_star_norm_sq = t_star.T @ t_star
            if np.isclose(t_star_norm_sq, 0.0):
                raise ValueError(
                    "Wold method encountered a zero-norm orthogonal "
                    "score vector after convergence."
                )
            w = Xk.T @ t_star / t_star_norm_sq
            w_norm = np.linalg.norm(w)
            if np.isclose(w_norm, 0.0):
                raise ValueError(
                    "Wold method encountered a zero-norm weight vector "
                    "after convergence."
                )
            w /= w_norm

            if iteration == self.max_iter - 1:
                warnings.warn(
                    f"Wold method did not converge after {self.max_iter} iterations.",
                    ConvergenceWarning,
                )

            # Calculate the loadings p
            p = Xk.T @ t_star / t_star_norm_sq

            # Store the scores, weights and loadings
            scores[:, k] = t_star.flatten()
            weights[:, k] = w.flatten()
            loadings[:, k] = p.flatten()

            # Deflate Xk by removing the contribution of the orthogonal component
            Xk -= np.outer(t_star, p)

            # Update iteration count
            n_iter[k] = iteration + 1

        return scores, weights, loadings, n_iter

    def _sjoblom_method(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate orthogonal components using Sjöblom's method."""
        # Initialize variables
        Xk = X.copy()
        y = np.asarray(y)

        y = y.reshape(-1, 1) if y.ndim == 1 else y
        y_pinv = pinv(y)

        # Get the features and components
        n_samples, n_features = X.shape

        scores = np.zeros((n_samples, self.n_components))
        weights = np.zeros((n_features, self.n_components))
        loadings = np.zeros((n_features, self.n_components))
        n_iter = np.zeros(self.n_components, dtype=int)

        for k in range(self.n_components):
            # Calculate the first singular vectors of Xk
            _, _, Vt = svd(Xk, full_matrices=False)
            t = Xk @ Vt.T[:, 0]

            for iteration in range(self.max_iter):
                # Center the scores (Equation 4 in Sjöblom et al.)
                t_mean = np.mean(t)
                t_centered = t - t_mean

                # Orthogonalize with respect to y (Equations 5 and 6 in Sjöblom
                # et al.). Keep the score vector as a 1D array of shape
                # (n_samples,) throughout the iteration.
                t_star = t_centered - y @ (y_pinv @ t_centered) + t_mean

                # Calculate loading vector w and scale (Equations 7 and 8 in
                # Sjöblom et al.). This keeps w in feature space with shape
                # (n_features,).
                t_star_norm_sq = t_star @ t_star
                if np.isclose(t_star_norm_sq, 0.0):
                    raise ValueError(
                        "Sjöblom method encountered a zero-norm orthogonal "
                        "score vector."
                    )

                w = Xk.T @ t_star / t_star_norm_sq
                w_norm = np.linalg.norm(w)
                if np.isclose(w_norm, 0.0):
                    raise ValueError(
                        "Sjöblom method encountered a zero-norm weight vector."
                    )
                w /= w_norm

                # Calculate t new from w (Equation 9 in Sjöblom et al.)
                t_new = Xk @ w

                # Vectorized convergence check
                if (
                    np.linalg.norm(t_new - t)
                    / max(np.linalg.norm(t), np.finfo(float).eps)
                    < self.tol
                ):
                    break

                t = t_new

            # Update t_star for the final iteration
            t_mean = np.mean(t)
            t_centered = t - t_mean
            t_star = t_centered - y @ (y_pinv @ t_centered) + t_mean

            if iteration == self.max_iter - 1:
                warnings.warn(
                    f"Sjöblom method did not converge after {self.max_iter} "
                    f"iterations.",
                    ConvergenceWarning,
                )

            # Calculate PLS regression between X and t_star (text after
            # Equation 9 in Sjöblom et al.). Treat t_star as a single-response
            # column vector to keep the SVD shapes explicit.
            # Calculate first singular vectors of X.T @ t_star
            t_star_column = t_star[:, np.newaxis]
            C = Xk.T @ t_star_column
            U, _, Vt = svd(C, full_matrices=False)

            # Calculate the x weights
            x_weights = U[:, 0]

            # Calculate the y weights
            y_weights = Vt.T[:, 0]

            # Calculate the regression vector
            x_rotations_ = np.dot(
                x_weights[:, np.newaxis],
                pinv(
                    np.dot(x_weights[np.newaxis, :], x_weights[:, np.newaxis]),
                    check_finite=False,
                ),
            )
            y_rotations_ = np.dot(
                y_weights[:, np.newaxis],
                pinv(
                    np.dot(y_weights[np.newaxis, :], y_weights[:, np.newaxis]),
                    check_finite=False,
                ),
            )

            coef = np.dot(x_rotations_, y_rotations_.T).ravel()

            # The new weights are the regression vector (Equation 10 in Sjöblom et al.)
            w_star = coef

            # Calculate the scores t_star_star using the new weights (Equation
            # 11 in Sjöblom et al.)
            t_star_star = Xk @ w_star

            # Calculate the x loadings p_star (Equation 12 in Sjöblom et al.)
            t_star_star_norm_sq = t_star_star @ t_star_star
            if np.isclose(t_star_star_norm_sq, 0.0):
                raise ValueError(
                    "Sjöblom method encountered a zero-norm final score vector."
                )
            p_star = Xk.T @ t_star_star / t_star_star_norm_sq

            # Store the scores, weights and loadings
            scores[:, k] = t_star_star.flatten()
            weights[:, k] = w_star.flatten()
            loadings[:, k] = p_star.flatten()

            # Deflate Xk by removing the contribution of the orthogonal component
            Xk -= np.outer(t_star_star, p_star)

            # Update iteration count
            n_iter[k] = iteration + 1

        return scores, weights, loadings, n_iter

    def _fearn_method(self, X: np.ndarray, y: np.ndarray):
        """Calculate orthogonal components using Fearn's method."""
        # Placeholder for Fearn's method implementation
        pass

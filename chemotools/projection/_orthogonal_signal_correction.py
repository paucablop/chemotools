"""
The :mod:`chemotools.projection._orthogonal_signal_correction` module
implements the Orthogonal Signal Correction (OSC) technique for preprocessing
spectral data by removing variations orthogonal to the target variable.
"""

# Author: Pau Cabaneros
# License: MIT

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
    A transformer that removes variation in X that is orthogonal to the target y.

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
        Fit and apply OSC to remove variation in `X` that is orthogonal to `y`.

    >>> import numpy as np
    >>> from chemotools.projection import OrthogonalSignalCorrection
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(8, 5))
    >>> y = np.linspace(0, 1, 8)
    >>> osc = OrthogonalSignalCorrection(n_components=1, method="wold")
    >>> X_osc = osc.fit_transform(X, y)
    >>> X_osc.shape
    (8, 5)

    Multivariate targets are also supported.

    >>> y_multi = np.column_stack([y, y**2])
    >>> osc = OrthogonalSignalCorrection(n_components=2, method="fearn")
    >>> osc.fit(X, y_multi)
    OrthogonalSignalCorrection(method='fearn')

    The transformer can be used inside a scikit-learn pipeline.

    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.cross_decomposition import PLSRegression
    >>> pipe = make_pipeline(
    ...     OrthogonalSignalCorrection(n_components=1, method="sjoblom"),
    ...     PLSRegression(n_components=2),
    ... )
    >>> pipe.fit(X, y)
    Pipeline(steps=[('orthogonalsignalcorrection',
                     OrthogonalSignalCorrection(
                         method='sjoblom', n_components=1
                     )),
                    ('plsregression', PLSRegression())])

    Notes
    -----
        OSC is a supervised preprocessing method: it removes components from `X`
        that are orthogonal to the provided target `y`. Because of this, the target
        used during `fit()` must be representative of the calibration problem.

        The transformed data keep the same shape as the input data. This estimator
        is therefore intended for signal correction rather than classical dimension
        reduction.

        The available methods differ in how orthogonal components are estimated:

        - `wold` and `sjoblom` use iterative updates and may emit
            `ConvergenceWarning` if `max_iter` is reached before convergence.
        - `fearn` uses a direct SVD-based formulation and does not require an
            iterative loop.

        In practice, a small number of components is usually preferred. Removing too
        many orthogonal components may discard structured variation that is still
        useful for the downstream model.


    See Also
    --------
        chemotools.projection.ExternalParameterOrthogonalization : Remove
            variation linked to external nuisance parameters.
        sklearn.pipeline.make_pipeline : Build preprocessing and modelling pipelines.

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

        # Dispatch to the selected OSC method
        if self.method == "wold":
            self.scores_, self.weights_, self.loadings_, self.n_iter_ = (
                self._wold_method(X_centered, y_centered)
            )

        if self.method == "sjoblom":
            self.scores_, self.weights_, self.loadings_, self.n_iter_ = (
                self._sjoblom_method(X_centered, y_centered)
            )

        if self.method == "fearn":
            self.scores_, self.weights_, self.loadings_, self.n_iter_ = (
                self._fearn_method(X_centered, y_centered)
            )

        return self

    def transform(self, X: np.ndarray, y=None):
        """Apply orthogonal signal correction to X.

        Projects X onto the latent components found during fitting.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform.

        y : None
            Ignored to align with API.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
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
            else:
                warnings.warn(
                    f"Wold method did not converge after {self.max_iter} iterations.",
                    ConvergenceWarning,
                )

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
            else:
                warnings.warn(
                    f"Sjöblom method did not converge after {self.max_iter} "
                    f"iterations.",
                    ConvergenceWarning,
                )

            # Update t_star for the final iteration
            t_mean = np.mean(t)
            t_centered = t - t_mean
            t_star = t_centered - y @ (y_pinv @ t_centered) + t_mean

            # Calculate PLS regression between X and t_star (text after
            # Equation 9 in Sjöblom et al.). Treat t_star as a single-response
            # column vector to keep the SVD shapes explicit.
            # Calculate first singular vectors of X.T @ t_star
            w_star = Xk.T @ t_star
            w_star /= np.linalg.norm(w_star)  # Normalize w_star to unit length

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
        # Initialize variables
        X = X.copy()
        y = np.asarray(y)

        y = y.reshape(-1, 1) if y.ndim == 1 else y

        n_samples, n_features = X.shape

        scores = np.zeros((n_samples, self.n_components))
        weights = np.zeros((n_features, self.n_components))
        loadings = np.zeros((n_features, self.n_components))
        n_iter = np.ones(self.n_components, dtype=int)  # Non-iterative per component

        # Calculate the residual matrix M (Equation -2 in Fearn's paper [3])
        Id = np.eye(X.shape[1])
        M = Id - X.T @ y @ pinv(y.T @ X @ X.T @ y) @ y.T @ X

        # Calculate the matrix Z (Equation -1 in Fearn's paper [3])
        Z = X @ M

        # Calculate the leading right singular vectors of Z (Equation 1 in Fearn's
        # paper). Use the first n_components vectors as orthogonal weights.
        _, _, Vt = svd(Z, full_matrices=False)
        weights = Vt.T[:, : self.n_components]

        weight_norms = np.linalg.norm(weights, axis=0)
        if np.any(np.isclose(weight_norms, 0.0)):
            raise ValueError("Fearn method encountered a zero-norm weight vector.")
        weights /= weight_norms

        # Calculate the scores T (Equation 2 in Fearn's paper)
        scores = X @ weights

        score_gram = scores.T @ scores
        if np.any(np.isclose(np.diag(score_gram), 0.0)):
            raise ValueError("Fearn method encountered a zero-norm score vector.")

        # Calculate the loadings P from the projected scores.
        loadings = X.T @ scores @ pinv(score_gram)

        return scores, weights, loadings, n_iter

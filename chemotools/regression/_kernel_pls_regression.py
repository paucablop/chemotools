"""
The :mod:`chemotools.regression._kernel_pls` module implements Kernel PLS regression.
"""

# Authors: Ruggero Guerrini
# License: MIT

from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data


class KernelPLSRegression(RegressorMixin, TransformerMixin, BaseEstimator):
    """
    Kernel Partial Least Squares (KernelPLSRegression) regression.

    Implements non-linear regression by applying the kernel trick to map
    input data into a reproducing kernel Hilbert space (RKHS), followed
    by PLS regression on the resulting kernel matrix, following the
    algorithm described in [1]_.

    Parameters
    ----------
    n_components : int, default=2
        Number of PLS components to extract.

    kernel : str, default="rbf"
        Kernel function to use. Supported: ``"rbf"``, ``"linear"``,
        ``"poly"``, ``"sigmoid"``.

    gamma : float, default=1.0
        Kernel coefficient for ``"rbf"``, ``"poly"``, and ``"sigmoid"``.

    degree : int, default=3
        Degree for the ``"poly"`` kernel. Ignored by other kernels.

    coef0 : float, default=1.0
        Independent term for ``"poly"`` and ``"sigmoid"`` kernels.
        Ignored by other kernels.

    scale_X : bool, default=False
        If True, standardize each feature of X (zero mean, unit variance)
        before computing the kernel. Recommended when features are on
        different scales, e.g. mixed spectral regions.

    scale : bool, default=False
        If True, scale the kernel matrix K and y inside PLSRegression.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.

    X_train_ : ndarray of shape (n_samples, n_features)
        Training data stored for kernel computation at predict time.

    X_mean_ : ndarray of shape (n_features,)
        Per-feature mean of X used for centering before kernel computation.
        Zero vector when ``scale_X=False``.

    X_std_ : ndarray of shape (n_features,)
        Per-feature standard deviation of X used for scaling before kernel
        computation. Ones vector when ``scale_X=False``.

    K_fit_rows_ : ndarray of shape (n_samples,)
        Per-column mean of the training kernel matrix. Used for kernel
        centering at predict time.

    K_fit_all_ : float
        Global mean of the training kernel matrix. Used for kernel
        centering at predict time.

    K_train_c_ : ndarray of shape (n_samples, n_samples)
        Centered training kernel matrix.

    x_mean_ : ndarray of shape (n_samples,)
        Mean of the centered kernel matrix columns.

    x_weights_ : ndarray of shape (n_samples, n_components)
        PLS X weights.

    y_weights_ : ndarray of shape (n_targets, n_components)
        PLS Y weights.

    x_loadings_ : ndarray of shape (n_samples, n_components)
        PLS X loadings.

    y_loadings_ : ndarray of shape (n_targets, n_components)
        PLS Y loadings.

    x_scores_ : ndarray of shape (n_samples, n_components)
        PLS X scores (latent variables) on training data.

    y_scores_ : ndarray of shape (n_samples, n_components)
        PLS Y scores on training data.

    x_rotations_ : ndarray of shape (n_samples, n_components)
        PLS X rotations.

    y_rotations_ : ndarray of shape (n_targets, n_components)
        PLS Y rotations.

    coef_ : ndarray of shape (n_targets, n_samples)
        PLS regression coefficients in kernel space.

    intercept_ : ndarray of shape (n_targets,)
        PLS intercept.

    n_iter_ : list of int
        Number of NIPALS iterations per component.

    y_was_1d_ : bool
        True if y was passed as a 1D array in fit.

    y_mean_ : ndarray of shape (n_targets,)
        Mean of y computed during fit. Stored for
        informational purposes only; the intercept_ already
        absorbs this value for prediction.

    y_std_ : ndarray of shape (n_targets,)
        Standard deviation of y computed during fit.
        Stored for informational purposes only.

    References
    ----------
    .. [1] Rosipal, R. & Trejo, L. J. (2001).
        Kernel Partial Least Squares Regression in Reproducing Kernel
        Hilbert Space. Journal of Machine Learning Research, 2, 97–123.
        http://www.jmlr.org/papers/volume2/rosipal01a/rosipal01a.pdf

    Examples
    -------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> from chemotools.models._kernel_pls import KernelPLS
    >>>
    >>> X = rng.normal(size=(100, 20))
    >>> X_test = rng.normal(size=(10, 20))
    >>> y = np.sin(X[:, 0]) + 0.5 * X[:, 1]**2 + 0.1 * rng.normal(size=100)
    >>>
    >>> model = KernelPLS(n_components=2, kernel="rbf", gamma=0.5)
    >>> model.fit(X, y)
    KernelPLS(gamma=0.5, n_components=2)
    >>> y_hat = model.predict(X_test)
    >>> y_hat.shape
    (10,)
    """

    _parameter_constraints = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "kernel": [StrOptions({"rbf", "linear", "poly", "sigmoid"})],
        "gamma": [Interval(Real, 0, None, closed="neither")],
        "degree": [Interval(Integral, 1, None, closed="left")],
        "coef0": [Interval(Real, None, None, closed="neither")],
        "scale_X": ["boolean"],
        "scale": ["boolean"],
    }

    def __init__(
        self,
        n_components: int = 2,
        kernel: str = "rbf",
        gamma: float = 1.0,
        degree: int = 3,
        coef0: float = 1.0,
        scale_X: bool = False,
        scale: bool = False,
    ):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.scale_X = scale_X
        self.scale = scale

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = True
        return tags

    def apply_kernel(self, X: np.ndarray, X2: np.ndarray) -> np.ndarray:
        if self.kernel in {"rbf"}:
            K = pairwise_kernels(
                X,
                X2,
                metric=self.kernel,
                gamma=self.gamma,
            )
        elif self.kernel in {"poly"}:
            K = pairwise_kernels(
                X,
                X2,
                metric=self.kernel,
                gamma=self.gamma,
                degree=self.degree,
                coef0=self.coef0,
            )
        elif self.kernel in {"sigmoid"}:
            K = pairwise_kernels(
                X,
                X2,
                metric=self.kernel,
                gamma=self.gamma,
                coef0=self.coef0,
            )
        elif self.kernel in {"linear"}:
            K = pairwise_kernels(
                X,
                X2,
                metric="linear",
            )
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")
        return K

    def center_kernel(self, X: np.ndarray) -> np.ndarray:
        K_pred_cols = X.mean(axis=1, keepdims=True)
        return X - self.K_fit_rows_ - K_pred_cols + self.K_fit_all_

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KernelPLSRegression":
        """
        Fit the KernelPLS model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training input data.

        y : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : KernelPLS
            Fitted estimator.

        """
        # Validate the input parameters
        self._validate_params()

        # Check that X is a 2D array and has only finite values
        X, y = validate_data(
            self,
            X,
            y,
            ensure_2d=True,
            dtype=np.float64,
            multi_output=True,
            y_numeric=True,
            reset=True,
        )

        self.y_was_1d_ = y.ndim == 1
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Verify that n_components does not exceed number of samples
        if self.n_components > X.shape[0]:
            raise ValueError(
                f"n_components={self.n_components} is too large. "
                f"It must be <= n_samples={X.shape[0]}."
            )

        # Optional: X scaling before kernel computation
        if self.scale_X:
            self.X_mean_ = X.mean(axis=0)
            self.X_std_ = X.std(axis=0)
        else:
            self.X_mean_ = np.zeros(X.shape[1])
            self.X_std_ = np.ones(X.shape[1])
        X_scaled = (X - self.X_mean_) / self.X_std_
        self.X_train_ = X_scaled

        # Kernel Computation
        K_train = self.apply_kernel(X_scaled, X_scaled)
        self.K_fit_rows_ = K_train.mean(axis=0)
        self.K_fit_all_ = K_train.mean()
        # Kernel Centering
        self.K_train_c_ = self.center_kernel(K_train)

        # Regression
        model_ = PLSRegression(n_components=self.n_components, scale=self.scale)
        model_.fit(self.K_train_c_, y)

        # Attributes from PLS
        self.x_weights_ = model_.x_weights_
        self.y_weights_ = model_.y_weights_
        self.x_loadings_ = model_.x_loadings_
        self.y_loadings_ = model_.y_loadings_
        self.x_scores_ = model_.x_scores_
        self.y_scores_ = model_.y_scores_
        self.x_rotations_ = model_.x_rotations_
        self.y_rotations_ = model_.y_rotations_
        # Mean of centered kernel — replicates PLSRegression
        # internal centering without relying on private
        # attributes (model_._x_mean).
        self.x_mean_ = self.K_train_c_.mean(axis=0)
        self.coef_ = model_.coef_
        self.intercept_ = model_.intercept_
        self.n_iter_ = model_.n_iter_
        # Other attributes
        self.y_mean_ = y.mean(axis=0)
        self.y_std_ = y.std(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project X into the PLS latent space (kernel scores).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        T : ndarray of shape (n_samples, n_components)
            Kernel scores (latent variables).
        """
        check_is_fitted(self, ["X_train_", "K_fit_rows_", "K_fit_all_"])

        X = validate_data(self, X, ensure_2d=True, dtype=np.float64, reset=False)

        X_scaled = (X - self.X_mean_) / self.X_std_
        # Kernel Computation
        K_test = self.apply_kernel(X_scaled, self.X_train_)

        # Kernel centering for test data
        K_test_c_ = self.center_kernel(K_test)

        # Project into latent space
        return K_test_c_ @ self.x_rotations_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the trained model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Predicted values
        """
        check_is_fitted(
            self,
            ["X_train_", "K_fit_rows_", "K_fit_all_", "y_was_1d_"],
        )

        X = validate_data(
            self,
            X,
            ensure_2d=True,
            dtype=np.float64,
            reset=False,
        )

        X_scaled = (X - self.X_mean_) / self.X_std_

        # Kernel Computation
        K_test = self.apply_kernel(X_scaled, self.X_train_)
        # Kernel centering
        K_test_c_ = self.center_kernel(K_test)
        y_pred = (K_test_c_ - self.x_mean_) @ self.coef_.T + self.intercept_

        if self.y_was_1d_:
            return y_pred[:, 0]

        return y_pred

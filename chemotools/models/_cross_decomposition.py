"""Enhanced PLS regression with automatic explained variance calculation.

This module extends sklearn's PLSRegression with chemometrics-specific features,
particularly automatic calculation of explained variance ratios for both X and Y spaces.
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression as _SklearnPLSRegression


class PLSRegression(_SklearnPLSRegression):
    """PLS regression with automatic explained variance calculation.

    This is an enhanced version of sklearn's PLSRegression that automatically
    calculates explained variance ratios for both X-space and Y-space after
    fitting. This makes it much easier to use with diagnostic plots and follow
    the same API as PCA.

    All parameters and methods from sklearn's PLSRegression are available.
    After fitting, two additional attributes are computed:

    - `explained_x_variance_ratio_`: Variance explained in X-space (predictors)
    - `explained_y_variance_ratio_`: Variance explained in Y-space (response)

    Following the PLSRegression implemented in scikit-learn [1] and [2], this
    extension uses the x_scores_ (t) to asymmetrically deflate the Y matrix.

    In PLS, the latent score vector t (from X) is used to model Y via its loading vector c:
        Y_hat = t @ c.T

    Deflation removes the part of Y explained by the current component:
        Y_new = Y - Y_hat

    This process is repeated for each component, using the corresponding t and c vectors.
    Note: Unlike PCA, deflation in PLS is asymmetric—Y is deflated using t-scores derived from X.
        
        
    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep. Should be in [1, min(n_samples, n_features, n_targets)].
    scale : bool, default=True
        Whether to scale X and Y.
    max_iter : int, default=500
        Maximum number of iterations of the power method when algorithm='nipals'.
    tol : float, default=1e-06
        Tolerance used as convergence criteria in the power method.
    copy : bool, default=True
        Whether to copy X and Y in fit before applying centering, and potentially scaling.

    Attributes
    ----------
    explained_x_variance_ratio_ : ndarray of shape (n_components,)
        Explained variance ratio in X-space (predictors) for each component.
        This measures how much variance in the predictor variables each latent
        variable captures. Automatically calculated after fitting.

    explained_y_variance_ratio_ : ndarray of shape (n_components,)
        Explained variance ratio in Y-space (response) for each component.
        This measures the prediction quality - how much variance in the response
        each latent variable explains. Automatically calculated after fitting.

    All other attributes from sklearn.cross_decomposition.PLSRegression:
    x_weights_, y_weights_, x_loadings_, y_loadings_, x_scores_, y_scores_,
    x_rotations_, y_rotations_, coef_, intercept_, n_features_in_, feature_names_in_

    References
    ----------
    .. [1] sklearn.cross_decomposition.PLSRegression
        https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html

    .. [2] Wegelin, J. A. (2000). 
        A Survey of Partial Least Squares (PLS) Methods, with Emphasis on the Two-Block Case. Technical Report No. 371, Department of Statistics, University of Washington, Seattle, WA

    .. [3] Abdi, H. (2003). 
        Partial Least Squares (PLS) Regression. In Lewis-Beck M., Bryman A., Futing T. (Eds.), Encyclopedia of Social Sciences Research Methods. Thousand Oaks (CA): Sage.

    Examples
    --------
    **Example 1: Basic usage with automatic variance calculation**

    >>> from chemotools.models import PLSRegression
    >>> from chemotools.plotting import ExplainedVariancePlot
    >>>
    >>> pls = PLSRegression(n_components=5)
    >>> pls.fit(X_train, y_train)
    >>>
    >>> # Variance ratios are automatically available!
    >>> print(f"LV1 explains {pls.explained_y_variance_ratio_[0]*100:.1f}% of Y variance")

    Notes
    -----
    - **X-space variance** is calculated using sequential deflation method
    - **Y-space variance** is calculated using sequential deflation method
    - For each component, variance is calculated before deflating the matrices
    - This is the standard approach in PLS regression for explained variance
    - Both calculations are performed automatically during the initial fit

    See Also
    --------
    sklearn.cross_decomposition.PLSRegression : Base sklearn PLS implementation
    chemotools.plotting.ExplainedVariancePlot : Visualization for explained variance
    """

    def __init__(
        self,
        n_components: int = 2,
        *,
        scale: bool = True,
        max_iter: int = 500,
        tol: float = 1e-06,
        copy: bool = True,
    ):
        """Initialize PLS Regression model.

        Parameters
        ----------
        n_components : int, default=2
            Number of components to keep.
        scale : bool, default=True
            Whether to scale X and Y.
        max_iter : int, default=500
            Maximum number of iterations of the power method.
        tol : float, default=1e-06
            Tolerance used as convergence criteria.
        copy : bool, default=True
            Whether to copy X and Y in fit before applying centering.

        Attributes (set after fitting)
        -------------------------------
        explained_x_variance_ratio_ : ndarray
            Explained variance ratio in X-space for each component.
        explained_y_variance_ratio_ : ndarray
            Explained variance ratio in Y-space for each component.
        """
        super().__init__(
            n_components=n_components,
            scale=scale,
            max_iter=max_iter,
            tol=tol,
            copy=copy,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PLSRegression":
        """Fit model to data and compute explained variance ratios.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target vectors.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Call parent fit method
        super().fit(X, y)

        # Calculate explained variance ratios automatically
        self._calculate_explained_variance(X, y)

        return self

    def transform(self, X: np.ndarray, y: np.ndarray | None = None, copy: bool = True):
        return super().transform(X, y=y, copy=copy)

    def _calculate_explained_variance(self, X, y):
        """Calculate explained variance ratios for X and Y spaces.

        This is called automatically after fitting and populates:
        - explained_x_variance_ratio_
        - explained_y_variance_ratio_

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target vectors.
        """
        # Convert to arrays if needed (handles pandas DataFrame/Series)
        X_array = np.asarray(X)
        y_array = np.asarray(y)
        y_2d = y_array.reshape(-1, 1) if y_array.ndim == 1 else y_array

        # Calculate X-space and Y-space variance using deflation method
        self.explained_x_variance_ratio_, self.explained_y_variance_ratio_ = (
            self._calculate_variance_deflation(X_array, y_2d)
        )

    def _calculate_variance_deflation(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate explained variance ratios using sequential deflation.

        This method calculates how much variance each component explains by
        sequentially deflating the X and Y matrices. This is the standard
        approach in PLS and provides accurate component-wise variance.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training vectors.
        y : ndarray of shape (n_samples, n_targets)
            Target vectors (2D).

        Returns
        -------
        tuple[ndarray, ndarray]
            - X variance ratios of shape (n_components,)
            - Y variance ratios of shape (n_components,)
        """
        # Ensure data is numeric (handle object dtype from sklearn tests)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Center X and Y (PLS already centers data, but we need the original centered versions)
        X_centered = X - X.mean(axis=0)
        y_centered = y - y.mean(axis=0)

        # Check for scaling
        if self.scale:
            X_std = X.std(axis=0, ddof=1)
            y_std = y.std(axis=0, ddof=1)
            # Avoid division by zero
            X_std[X_std == 0] = 1.0
            y_std[y_std == 0] = 1.0
            X_centered /= X_std
            y_centered /= y_std

        # Total variance in centered data
        X_total_var = np.var(X_centered, axis=0).sum()
        y_total_var = np.var(y_centered, axis=0).sum()

        # Initialize matrices for deflation
        X_current = X_centered.copy()
        y_current = y_centered.copy()

        X_var_ratios = []
        y_var_ratios = []

        # For each component, calculate variance explained then deflate
        for a in range(self.n_components):
            # Get scores and loadings for component a
            t_a = self.x_scores_[:, a].reshape(-1, 1)  # (n_samples, 1)
            p_a = self.x_loadings_[:, a].reshape(-1, 1)  # (n_features_X, 1)
            c_a = self.y_loadings_[:, a].reshape(-1, 1)  # (n_features_y, 1)

            # Reconstruct X and y using current component
            X_hat = t_a @ p_a.T
            y_hat = t_a @ c_a.T

            # Variance of current residual before deflation
            X_var_before = np.var(X_current, axis=0).sum()
            y_var_before = np.var(y_current, axis=0).sum()

            # Deflate X and y
            X_current = X_current - X_hat
            y_current = y_current - y_hat

            # Variance of residual after deflation
            X_var_after = np.var(X_current, axis=0).sum()
            y_var_after = np.var(y_current, axis=0).sum()

            # Variance explained = reduction in variance
            X_var_explained = X_var_before - X_var_after
            y_var_explained = y_var_before - y_var_after

            # Store as ratio of total variance
            X_var_ratios.append(X_var_explained / X_total_var)
            y_var_ratios.append(y_var_explained / y_total_var)

        return np.array(X_var_ratios), np.array(y_var_ratios)

    def __repr__(self):
        """Enhanced repr showing variance info if fitted."""
        base_repr = super().__repr__()

        # Add variance info if model is fitted
        if hasattr(self, "explained_x_variance_ratio_"):
            total_x = self.explained_x_variance_ratio_.sum() * 100
            total_y = self.explained_y_variance_ratio_.sum() * 100
            variance_info = (
                f"\n  X-space variance explained: {total_x:.1f}%"
                f"\n  Y-space variance explained: {total_y:.1f}%"
            )
            # Insert before the closing parenthesis
            base_repr = base_repr.rstrip(")") + "," + variance_info + "\n)"

        return base_repr

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
    **Basic usage with automatic variance calculation**

    >>> from chemotools.models import PLSRegression
    >>> import numpy as np
    >>>
    >>> # Generate sample data
    >>> X = np.random.randn(100, 50)
    >>> y = X[:, 0] + 2*X[:, 1] + np.random.randn(100)*0.1
    >>>
    >>> # Fit model
    >>> pls = PLSRegression(n_components=5)
    >>> pls.fit(X, y)
    >>>
    >>> # Variance ratios are automatically available!
    >>> print(f"LV1 explains {pls.explained_y_variance_ratio_[0]*100:.1f}% of Y variance")
    >>> print(f"Total Y variance: {pls.explained_y_variance_ratio_.sum()*100:.1f}%")
    >>>
    >>> # Use with plotting
    >>> from chemotools.plotting import ExplainedVariancePlot
    >>> plot = ExplainedVariancePlot(pls)
    >>> plot.show()

    Notes
    -----
    **Variance Calculation:**

    - **X-space variance** is calculated using sequential deflation and will sum to ~1.0 (100%)
    - **Y-space variance** is calculated using sequential deflation but may not sum to 1.0
      due to asymmetric deflation (Y deflated with X-scores). The sum depends on X-Y correlation.
    - For each component, variance explained = variance reduction after deflation
    - This follows the standard PLS variance decomposition methodology (Wegelin, 2000)

    **Scaling:**

    - When ``scale=True``, data is standardized before variance calculation
    - The variance ratios reflect the scaled space, not the original data space
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
            Whether to scale X and Y to unit variance.
        max_iter : int, default=500
            Maximum number of iterations of the power method.
        tol : float, default=1e-06
            Tolerance used as convergence criteria.
        copy : bool, default=True
            Whether to copy X and Y in fit before applying centering.
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

        This method extends sklearn's fit() by automatically calculating
        explained variance ratios after fitting.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors. Accepts numpy arrays, pandas DataFrames.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target vectors. Accepts 1D (univariate) or 2D (multivariate) targets.

        Returns
        -------
        self : PLSRegression
            Fitted estimator with populated variance attributes:
            ``explained_x_variance_ratio_`` and ``explained_y_variance_ratio_``.
        """
        # Call parent fit method
        super().fit(X, y)

        # Calculate explained variance ratios automatically
        (
            self.explained_x_variance_ratio_,
            self.explained_y_variance_ratio_,
        ) = self._calculate_explained_variance_deflation(X, y)

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
        X_scores : ndarray of shape (n_samples, n_components)
            X transformed into the latent space (X-scores).
        """
        return super().transform(X, y=y, copy=copy)

    def _calculate_explained_variance_deflation(
        self, X, y
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate explained variance ratios using sequential deflation.

        This implements the variance decomposition for PLS regression following
        the deflation methodology described in Wegelin (2000).

        This method calculates how much variance each component explains by
        sequentially deflating the X and Y matrices. This is the standard
        approach in PLS and provides accurate component-wise variance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors. Accepts numpy arrays, pandas DataFrames.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target vectors. Accepts 1D (univariate) or 2D (multivariate) targets.

        Returns
        -------
        tuple[ndarray, ndarray]
            - X variance ratios of shape (n_components,)
            - Y variance ratios of shape (n_components,)
        """
        # Convert to arrays and ensure y is 2D (handles pandas DataFrame/Series)
        X = np.asarray(X, dtype=float)
        y_array = np.asarray(y, dtype=float)
        y = np.atleast_2d(y_array).T if y_array.ndim == 1 else y_array

        # Center X and Y (PLS already centers data, but we need the original centered versions)
        X_centered = X - X.mean(axis=0)
        y_centered = y - y.mean(axis=0)

        # Check for scaling
        if self.scale:
            # Scale to unit variance, avoiding division by zero
            X_centered /= np.maximum(X.std(axis=0, ddof=1), 1.0)
            y_centered /= np.maximum(y.std(axis=0, ddof=1), 1.0)

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
            # Get scores and loadings for component a (using slicing to keep 2D)
            t_a = self.x_scores_[:, a : a + 1]  # (n_samples, 1)
            p_a = self.x_loadings_[:, a : a + 1]  # (n_features_X, 1)
            q_a = self.y_loadings_[:, a : a + 1]  # (n_features_y, 1)

            # Reconstruct X and y using current component
            X_hat = t_a @ p_a.T
            y_hat = t_a @ q_a.T

            # Variance of current residual before deflation
            X_var_before = np.var(X_current, axis=0).sum()
            y_var_before = np.var(y_current, axis=0).sum()

            # Deflate X and y
            X_current -= X_hat
            y_current -= y_hat

            # Variance of residual after deflation
            X_var_after = np.var(X_current, axis=0).sum()
            y_var_after = np.var(y_current, axis=0).sum()

            # Store variance explained as ratio of total variance
            X_var_ratios.append((X_var_before - X_var_after) / X_total_var)
            y_var_ratios.append((y_var_before - y_var_after) / y_total_var)

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

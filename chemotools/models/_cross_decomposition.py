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

    These can be used directly with `ExplainedVariancePlot` for model diagnostics.

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
    >>>
    >>> # Use directly with plotting
    >>> plot_x = ExplainedVariancePlot(pls.explained_x_variance_ratio_)
    >>> plot_x.show(title='PLS Variance in X-space')
    >>>
    >>> plot_y = ExplainedVariancePlot(pls.explained_y_variance_ratio_)
    >>> plot_y.show(title='PLS Variance in Y-space')

    **Example 2: Side-by-side comparison**

    >>> import matplotlib.pyplot as plt
    >>> fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    >>>
    >>> ExplainedVariancePlot(pls.explained_x_variance_ratio_).render(ax=axes[0])
    >>> axes[0].set_title('X-space (Predictors)')
    >>>
    >>> ExplainedVariancePlot(pls.explained_y_variance_ratio_).render(ax=axes[1])
    >>> axes[1].set_title('Y-space (Response)')

    **Example 3: Just like PCA!**

    >>> from sklearn.decomposition import PCA
    >>>
    >>> # PCA workflow
    >>> pca = PCA(n_components=5)
    >>> pca.fit(X)
    >>> plot = ExplainedVariancePlot(pca.explained_variance_ratio_)
    >>>
    >>> # PLS workflow - now just as easy!
    >>> pls = PLSRegression(n_components=5)
    >>> pls.fit(X, y)
    >>> plot = ExplainedVariancePlot(pls.explained_y_variance_ratio_)

    Notes
    -----
    - **X-space variance** is calculated using score variances (fast, no refitting)
    - **Y-space variance** is calculated using R² from predictions with 1..n components
    - Y-space calculation uses the fitted model's scores and loadings directly,
      no refitting required - just matrix multiplication!
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

        # Calculate X-space variance (using score variances - fast method)
        self.explained_x_variance_ratio_ = self._calculate_x_variance()

        # Calculate Y-space variance (using R² method - accurate for prediction quality)
        self.explained_y_variance_ratio_ = self._calculate_y_variance(X_array, y_2d)

    def _calculate_x_variance(self) -> np.ndarray:
        """Calculate explained variance ratio for X-space.

        Uses the score variance method which is fast and doesn't require refitting.

        Returns
        -------
        ndarray of shape (n_components,)
            Explained variance ratio for each component in X-space.
        """
        # Score variances
        score_variances = np.var(self.x_scores_, axis=0)

        # Total variance in X (using the data the model was trained on)
        # We need to calculate this from the scores and loadings
        # Total variance = sum of all score variances
        total_variance = np.sum(score_variances)

        # Explained variance ratio
        var_individual = score_variances / total_variance

        return var_individual

    def _calculate_y_variance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate explained variance ratio for Y-space.

        Uses the R² method which accurately reflects prediction quality.
        Uses the already-fitted model's scores and loadings to reconstruct
        predictions with different numbers of components (no refitting needed).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training vectors.
        y : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Target vectors.

        Returns
        -------
        ndarray of shape (n_components,)
            Explained variance ratio for each component in Y-space.
        """
        # Ensure y is 2D
        y_2d = y.reshape(-1, 1) if y.ndim == 1 else y

        # Center y
        y_centered = y_2d - y_2d.mean(axis=0)
        total_variance = float(np.sum(y_centered**2))

        # Calculate cumulative R² for each number of components
        # Using the already fitted model's x_scores_ and y_loadings_
        var_ratios = []
        for i in range(1, self.n_components + 1):
            # Reconstruct Y using first i components: Y_pred = X_scores @ Y_loadings.T
            y_pred = self.x_scores_[:, :i] @ self.y_loadings_[:, :i].T + self._y_mean

            # Calculate R² (cumulative explained variance)
            ss_res = np.sum((y_2d - y_pred) ** 2)
            r2 = 1.0 - (ss_res / total_variance)
            var_ratios.append(float(r2))

        # Convert cumulative to individual variance per component
        var_cumulative = np.array([0.0] + var_ratios)
        var_individual = np.diff(var_cumulative)

        return var_individual

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

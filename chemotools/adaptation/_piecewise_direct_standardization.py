"""
Test for PiecewiseDirectStandardization
"""

# Author: Ruggero Guerrini
# Licence: MIT

from numbers import Integral

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted, validate_data


class PiecewiseDirectStandardization(TransformerMixin, BaseEstimator):
    """
    Implement a piecewise direct standardization transformer for
    the calibration transfer
    (domain adaptation) application.
    NOTE
    X_target can be provided either at initialization or during fit.

    - Use initialization when X_target is fixed and reused across multiple fits
    (e.g., same target instrument used repeatedly in pipelines).

    - Use fit-time X_target when working with Pipeline or GridSearchCV,
    or when X_target varies across experiments (e.g., cross-validation splits
    or different target batches).

    Parameters
    ----------
    X_target : np.ndarray of shape (n_samples, n_features), optional
        Target instrument data used to compute the transformation.
        If not provided, X is used as both source and target.

    window_length : int
        Half-width (w) of the local spectral window used in PDS

    n_components : int
        Number of components to keep for PLS model

    scale : bool, default = True
        Whether to scale X and Y in the PLS model

    Reference
    ---------
    .. [1] Wang, Yongdong., Veltkamp,
        D. J., & Kowalski, B. R. (1991).
        Multivariate instrument standardization.
        Analytical Chemistry, 63(23), 2750–2756.

    .. [2] Bouveresse, E.; Massart, D. L. (1996).
        Improvement of the piecewise
        direct standardisation procedure
        for the transfer of NIR spectra
        for multivariate calibration.
        Chemometrics and Intelligent
        Laboratory Systems, 32(2), 201–213.

    Attributes
    ----------
    n_samples_ : int
        Number of samples in the training data (X and X_target).

    n_features_ : int
        Number of features in the traininf data (X and X_target).

    pls_ : list[dict]
        List containing the parameters of the local PLS models for each feature.
        Each dictionary contains:
        - 'x_mean_' : mean of the local X window
        - 'coef_' : regression coefficients
        - 'intercept_' : intercept term

    Raises
    ------
    ValueError
        If X and X_target do not have the same shape.

    Examples
    --------
    **Basic usage**
    >>> # Import necessary libraries
    >>> from chemotools.adaptation import PiecewiseDirectStandardization
    >>> import numpy as np
    >>>
    >>> # Generate sample data
    >>> rng = np.random.default_rng(17)
    >>> X_target = rng.normal(size=(100, 20))
    >>> X_source = X_target * 2 - rng.normal(size=(100, 20)) * 0.02
    >>> # Train the model
    >>> DS = PiecewiseDirectStandardization(X_target = X_target).fit(X_source)
    >>> # Apply to a new set of data
    >>> X_transf = DS.transform(X_source_new)
    **Use the module for a Pipeline/GridSearchCV**
    >>> # Import necessary libraries
    >>> import numpy as np
    >>> import pytest
    >>> import sklearn
    >>> from sklearn.cross_decomposition import PLSRegression
    >>> from sklearn.exceptions import NotFittedError
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.utils.estimator_checks import check_estimator
    >>> from sklearn.utils.metadata_routing import MetadataRouter
    >>> from chemotools.adaptation._direct_standardization import PiecewiseDirectStandardization
    >>> from chemotools.derivative import SavitzkyGolay
    >>> from chemotools.scatter import StandardNormalVariate
    >>>
    >>> # Generate sample data
    >>> rng = np.random.default_rng(17)
    >>> X_target = rng.normal(size=(100, 20))
    >>> X_source = X_target * 2 - rng.normal(size=(100, 20)) * 0.02
    >>> # Pipeline
    >>> pipe = Pipeline([
    >>>     ("scaler", StandardNormalVariate()),
    >>>     ("model", PiecewiseDirectStandardization(X_target = X_target)),
    >>> ])
    >>> pipe.fit(X_source)
    >>> X_transformed = pipe.transform(X_source)
    >>>
    >>> # Generate sample data
    >>> rng = np.random.default_rng(17)
    >>> X_target = rng.normal(size=(100, 20))
    >>> X_source = X_target * 2 - rng.normal(size=(100, 20)) * 0.02
    >>> # Pipeline + GridSearchCV
    >>> sklearn.set_config(enable_metadata_routing=True)
    >>> pipe = Pipeline([
    >>>     ("scaler", SavitzkyGolay()),
    >>>     ("model", PiecewiseDirectStandardization().set_fit_request(X_target=True)),
    >>>     ("pls", PLSRegression()),
    >>> ])
    >>> param_grid = {
    >>>     "scaler__window_length": [15, 25],
    >>>     "scaler__polyorder": [2, 3],
    >>>     "scaler__deriv": [1, 2],
    >>>     "model__window_length": [10, 15, 20],
    >>>     "model__n_components": [2, 3, 5],
    >>>     "pls__n_components": [2, 3],
    >>> }
    >>> grid = GridSearchCV(pipe, param_grid, cv=3, error_score="raise")
    >>> grid.fit(X_source, y_concentration, X_target=X_target)

    """

    _parameter_constraints: dict = {
        "X_target": ["no_validation"],
        "window_length": [Interval(Integral, 1, None, closed="left")],
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "scale": ["boolean"],
    }

    def __init__(
        self,
        X_target: np.ndarray | None = None,
        window_length: int = 25,
        n_components: int = 2,
        scale: bool = True,
    ):
        self.X_target = X_target
        self.window_length = window_length
        self.n_components = n_components
        self.scale = scale

    def fit(
        self, X: np.ndarray, y=None, X_target: np.ndarray | None = None
    ) -> "PiecewiseDirectStandardization":
        """
        Fit the PiecewiseDirectStandardization to the input data.
        If X_target is not provided, X is used as both source and target,
        resulting in an identity-like transformation.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The source data

        y : Ignored
            Present for API compatibility with scikit-learn.

        X_target : np.ndarray of shape (n_samples, n_features)
            The target data

        Returns
        -------
        self : PiecewiseDirectStandardization
            The PDS transformer.
        """

        # validate_data
        X = validate_data(self, X, ensure_2d=True, reset=True, dtype=np.float64)
        # Priority: explicit fit-time X_target > __init__ X_target > identity
        if X_target is not None:
            _X_target = np.asarray(X_target, dtype=np.float64)
        elif self.X_target is not None:
            _X_target = np.asarray(self.X_target, dtype=np.float64)
        else:
            _X_target = X.copy()

        if _X_target.shape != X.shape:
            raise ValueError(
                f"X and X_target must have the same shape, "
                f"got X={X.shape} and X_target={_X_target.shape}."
            )
        n, p = _X_target.shape
        self.n_samples_ = n
        self.n_features_ = p

        self.pls_ = []
        for i in range(p):
            # close to the edge avoid errors
            l_lim = max(0, i - self.window_length)
            r_lim = min(p, i + self.window_length + 1)
            model = PLSRegression(
                n_components=self.n_components,
                scale=self.scale,
            ).fit(X[:, l_lim:r_lim], _X_target[:, i])
            params = {
                "x_mean_": X[:, l_lim:r_lim].mean(axis=0),
                "coef_": model.coef_,
                "intercept_": model.intercept_,
            }
            self.pls_.append(params)
        return self

    def transform(self, X_new) -> np.ndarray:
        """
        Use the trained model to transform the source data

        Parameters
        ----------
        X_new : np.ndarray of shape (n_samples, n_features)
            The input data to transform

        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_features)
            The data transformed
        """
        # Check the data
        X_new = validate_data(
            self,
            X_new,
            ensure_2d=True,
            reset=False,
            dtype=np.float64,
        )
        # Verify that the model was trained
        check_is_fitted(self)
        X_transformed = np.zeros(X_new.shape)
        for i in range(self.n_features_):
            # close to the edge avoid errors
            l_lim = max(0, i - self.window_length)
            r_lim = min(self.n_features_, i + self.window_length + 1)

            X = X_new[:, l_lim:r_lim] - self.pls_[i]["x_mean_"]
            X_transformed[:, i] = (
                X @ self.pls_[i]["coef_"].T + self.pls_[i]["intercept_"]
            ).ravel()
        return X_transformed

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
        Number of features in the training data (X and X_target).

    x_mean_ : np.ndarray of shape (n_features, 2 * window_length + 1)
        Mean of the local X window for each feature.

    coef_ : np.ndarray of shape (n_features, 2 * window_length + 1)
        Regression coefficients for each local PLS model.

    intercept_ : np.ndarray of shape (n_features,)
        Intercept term for each local PLS model.

    Raises
    ------
    ValueError
        If X and X_target do not have the same shape.

    Examples
    --------
    **Basic usage**
    >>> import numpy as np
    >>> from chemotools.adaptation import PiecewiseDirectStandardization
    >>>
    >>> rng = np.random.default_rng(17)
    >>> X_target = rng.normal(size=(100, 20))
    >>> X_source = X_target * 2 - rng.normal(size=(100, 20)) * 0.02
    >>> X_source_new = rng.normal(size=(10, 20))
    >>>
    >>> PDS = PiecewiseDirectStandardization(X_target=X_target).fit(X_source)
    >>> X_transf = PDS.transform(X_source_new)

    **Pipeline**
    >>> import numpy as np
    >>> from sklearn.pipeline import Pipeline
    >>> from chemotools.adaptation import PiecewiseDirectStandardization
    >>> from chemotools.scatter import StandardNormalVariate
    >>>
    >>> rng = np.random.default_rng(17)
    >>> X_target = rng.normal(size=(100, 20))
    >>> X_source = X_target * 2 - rng.normal(size=(100, 20)) * 0.02
    >>>
    >>> pipe = Pipeline([
    >>>     ("scaler", StandardNormalVariate()),
    >>>     ("model", PiecewiseDirectStandardization(X_target=X_target)),
    >>> ])
    >>> pipe.fit(X_source)
    >>> X_transformed = pipe.transform(X_source)

    **Pipeline + GridSearchCV**
    >>> import numpy as np
    >>> import sklearn
    >>> from sklearn.cross_decomposition import PLSRegression
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.pipeline import Pipeline
    >>> from chemotools.adaptation import PiecewiseDirectStandardization
    >>> from chemotools.derivative import SavitzkyGolay
    >>>
    >>> rng = np.random.default_rng(17)
    >>> X_target = rng.normal(size=(100, 20))
    >>> X_source = X_target * 2 - rng.normal(size=(100, 20)) * 0.02
    >>> y_concentration = rng.normal(size=(100, 1))
    >>>
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
            Source data.

        y : Ignored
            Present for API compatibility with scikit-learn.

        X_target : np.ndarray of shape (n_samples, n_features), optional
            Target data. Overrides the value provided at initialization.

        Returns
        -------
        self : PiecewiseDirectStandardization
            PDS transformer.
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

        max_win = 2 * self.window_length + 1
        self.x_mean_ = np.zeros((p, max_win))
        self.coef_ = np.zeros((p, max_win))
        self.intercept_ = np.zeros(p)

        for i in range(p):
            l_lim = max(0, i - self.window_length)
            r_lim = min(p, i + self.window_length + 1)
            win_size = r_lim - l_lim

            model = PLSRegression(
                n_components=self.n_components,
                scale=self.scale,
            ).fit(X[:, l_lim:r_lim], _X_target[:, i])

            self.x_mean_[i, :win_size] = X[:, l_lim:r_lim].mean(axis=0)
            self.coef_[i, :win_size] = model.coef_.ravel()
            self.intercept_[i] = model.intercept_[0]
        return self

    def transform(self, X) -> np.ndarray:
        """
        Use the trained model to transform the source data

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to transform

        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_features)
            Data transformed
        """
        # Check the data
        X = validate_data(
            self,
            X,
            ensure_2d=True,
            reset=False,
            dtype=np.float64,
        )
        # Verify that the model was trained
        check_is_fitted(self)
        X_transformed = np.zeros(X.shape)
        for i in range(self.n_features_):
            l_lim = max(0, i - self.window_length)
            r_lim = min(self.n_features_, i + self.window_length + 1)
            win_size = r_lim - l_lim

            X_win = X[:, l_lim:r_lim] - self.x_mean_[i, :win_size]
            X_transformed[:, i] = X_win @ self.coef_[i, :win_size] + self.intercept_[i]
        return X_transformed

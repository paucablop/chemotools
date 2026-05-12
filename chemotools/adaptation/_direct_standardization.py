"""
The :mod:'chemotools.adaptation:DirectStandardization'
module implements a Direct Standardization transformer
"""

# Authors: Ruggero Guerrini
# License: MIT

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class DirectStandardization(TransformerMixin, BaseEstimator):
    """
    Direct Standardization (DS) transformer for calibration transfer
    (domain adaptation) applications.
    The transformer estimates a global linear mapping from the source
    space to the target space using least squares.
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
        
    Attributes
    ----------
    T_ : np.ndarray of shape (n_features, n_features)
        Linear transformation matrix mapping source space to target space.

    Reference
    ---------
    .. [1] Wang, Yongdong., Veltkamp,
        D. J., & Kowalski, B. R. (1991).
        Multivariate instrument standardization.
        Analytical Chemistry, 63(23), 2750–2756.

    Raises
    ------
    ValueError
        If X and X_target do not have the same shape.

    Examples
    --------
    **Basic usage**
    >>> import numpy as np
    >>> from chemotools.adaptation import DirectStandardization
    >>>
    >>> rng = np.random.default_rng(17)
    >>> X_target = rng.normal(size=(100, 20))
    >>> X_source = X_target * 2 - rng.normal(size=(100, 20)) * 0.02
    >>> X_source_new = rng.normal(size=(10, 20))
    >>>
    >>> DS = DirectStandardization(X_target=X_target).fit(X_source)
    >>> X_transf = DS.transform(X_source_new)

    **Pipeline**
    >>> import numpy as np
    >>> from sklearn.pipeline import Pipeline
    >>> from chemotools.adaptation import DirectStandardization
    >>> from chemotools.scatter import StandardNormalVariate
    >>>
    >>> rng = np.random.default_rng(17)
    >>> X_target = rng.normal(size=(100, 20))
    >>> X_source = X_target * 2 - rng.normal(size=(100, 20)) * 0.02
    >>>
    >>> pipe = Pipeline([
    >>>     ("scaler", StandardNormalVariate()),
    >>>     ("ds", DirectStandardization(X_target=X_target)),
    >>> ])
    >>> pipe.fit(X_source)
    >>> X_transformed = pipe.transform(X_source)

    **Pipeline + GridSearchCV**
    >>> import numpy as np
    >>> import sklearn
    >>> from sklearn.cross_decomposition import PLSRegression
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.pipeline import Pipeline
    >>> from chemotools.adaptation import DirectStandardization
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
    >>>     ("ds", DirectStandardization().set_fit_request(X_target=True)),
    >>>     ("pls", PLSRegression()),
    >>> ])
    >>> param_grid = {
    >>>     "scaler__window_length": [15, 25],
    >>>     "scaler__polyorder": [2, 3],
    >>>     "scaler__deriv": [1, 2],
    >>>     "pls__n_components": [2, 3],
    >>> }
    >>> grid = GridSearchCV(pipe, param_grid, cv=3, error_score="raise")
    >>> grid.fit(X_source, y_concentration, X_target=X_target)
    """

    _parameter_constraints: dict = {
        "X_target": ["no_validation"],
    }

    def __init__(self, X_target: np.ndarray | None = None):
        self.X_target = X_target

    def fit(
        self, X: np.ndarray, y=None, X_target: np.ndarray | None = None
    ) -> "DirectStandardization":
        """
        Fit the DirectStandardization model.

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
        self : DirectStandardization
            DS transformer.
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

        self.T_ = np.linalg.pinv(X) @ _X_target

        return self

    def transform(self, X) -> np.ndarray:
        """
        Transform the source data

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to transform

        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_features)
            Data transformed
        """
        # Validate input data
        check_is_fitted(self, ["T_"])
        X = validate_data(
            self,
            X,
            ensure_2d=True,
            reset=False,
            dtype=np.float64,
        )
        # Apply the transformation
        return X @ self.T_

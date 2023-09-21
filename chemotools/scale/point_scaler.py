import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted

from chemotools.utils.check_inputs import check_input


class PointScaler(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    """
    A transformer that scales the input data by the intensity value at a given point. 
    The point can be specified by an index or by a wavenumber.

    Parameters
    ----------
    point : int, 
        The point to scale the data by. It can be an index or a wavenumber.

    wavenumber : array-like, optional
        The wavenumbers of the input data. If not provided, the indices will be used
        instead. Default is None. If provided, the wavenumbers must be provided in
        ascending order.

    Attributes
    ----------
    point_index_ : int
        The index of the point to scale the data by. It is 0 if the wavenumbers are not provided.

    n_features_in_ : int
        The number of features in the input data.

    _is_fitted : bool
        Whether the transformer has been fitted to data.

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the input data.

    transform(X, y=0, copy=True)
        Transform the input data by scaling by the value at a given Point.
    """
    def __init__(self, point: int = 0, wavenumbers: np.ndarray = None):
        self.point = point
        self.wavenumbers = wavenumbers


    def fit(self, X: np.ndarray, y=None) -> "PointScaler":
        """
        Fit the transformer to the input data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to fit the transformer to.

        y : None
            Ignored.

        Returns
        -------
        self : PointScaler
            The fitted transformer.
        """
        # Check that X is a 2D array and has only finite values
        X = check_input(X)

        # Set the number of features
        self.n_features_in_ = X.shape[1]

        # Set the fitted attribute to True
        self._is_fitted = True

        # Set the point index
        if self.wavenumbers is None:
            self.point_index_ = self.point
        else:
            self.point_index_ = self._find_index(self.point)


        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Transform the input data by scaling by the value at a given Point.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to transform.

        y : None
            Ignored.

        Returns
        -------
        X_ : np.ndarray of shape (n_samples, n_features)
            The transformed data.
        """
        # Check that the estimator is fitted
        check_is_fitted(self, "_is_fitted")

        # Check that X is a 2D array and has only finite values
        X = check_input(X)
        X_ = X.copy()

        # Check that the number of features is the same as the fitted data
        if X_.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features but got {X_.shape[1]}")

        # Scale the data by Point
        for i, x in enumerate(X_):
            X_[i] = x / x[self.point_index_]
        
        return X_.reshape(-1, 1) if X_.ndim == 1 else X_
    
    def _find_index(self, target: float) -> int:
        wavenumbers = np.array(self.wavenumbers)
        return np.argmin(np.abs(wavenumbers - target))
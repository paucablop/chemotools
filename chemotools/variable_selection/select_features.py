import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted

from chemotools.utils.check_inputs import check_input


class SelectFeatures(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    """
    A transformer that Selects the spectral data to a specified array of features. This
    array can be continuous or discontinuous. The array of features is specified by:
    - by the indices of the wavenumbers to select,
    - by the wavenumbers to select, the wavenumbers must be provided to the transformer
        when it is initialised. If the wavenumbers are not provided, the indices will be
        used instead. The wavenumbers must be provided in ascending order.

    Parameters
    ----------
    features : narray-like, optional
        The index of the features to select. Default is None.

    wavenumbers : array-like, optional
        The wavenumbers of the input data. If not provided, the indices will be used
        instead. Default is None. If provided, the wavenumbers must be provided in
        ascending order.

    Attributes
    ----------
    features_index_ : int
        The index of the features to select.

    n_features_in_ : int
        The number of features in the input data.

    _is_fitted : bool
        Whether the transformer has been fitted to data.

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the input data.

    transform(X, y=0, copy=True)
        Transform the input data by cutting it to the specified range.
    """

    def __init__(
        self,
        features: np.ndarray = None,
        wavenumbers: np.ndarray = None,
    ):
        self.features = features
        self.wavenumbers = wavenumbers

    def fit(self, X: np.ndarray, y=None) -> "SelectFeatures":
        """
        Fit the transformer to the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to fit the transformer to.

        y : None
            Ignored.

        Returns
        -------
        self : SelectFeatures
            The fitted transformer.
        """
        # Check that X is a 2D array and has only finite values
        X = check_input(X)

        # Set the number of features
        self.n_features_in_ = X.shape[1]

        # Set the fitted attribute to True
        self._is_fitted = True

        # Set the start and end indices
        if self.features is None:
            self.features_index_ = self.features
            return self

        if self.wavenumbers is None:
            self.features_index_ = self.features
            return self

        self.features_index_ = self._find_indices()

        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Transform the input data by cutting it to the specified range.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
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
            raise ValueError(
                f"Expected {self.n_features_in_} features but got {X_.shape[1]}"
            )

        # Select the features
        if self.features is None:
            return X_

        return X_[:, self.features_index_]

    def _find_index(self, target: float) -> int:
        if self.wavenumbers is None:
            return target
        wavenumbers = np.array(self.wavenumbers)
        return np.argmin(np.abs(wavenumbers - target))

    def _find_indices(self) -> np.ndarray:
        return np.array([self._find_index(feature) for feature in self.features])

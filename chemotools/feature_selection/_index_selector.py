import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin

from sklearn.utils.validation import check_is_fitted

from chemotools.utils.check_inputs import check_input


class IndexSelector(BaseEstimator, SelectorMixin):
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

    def fit(self, X: np.ndarray, y=None) -> "IndexSelector":
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
        self : IndexSelector
            The fitted transformer.
        """
        # validate that X is a 2D array and has only finite values
        X = self._validate_data(X)

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

    def _get_support_mask(self):
        """
        Get the boolean mask indicating which features are selected.

        Returns
        -------
        mask : ndarray of shape (n_features_in_,)
            The mask indicating the selected features.
        """
        # Check that the estimator is fitted
        check_is_fitted(self)

        # Create the mask
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.features_index_] = True

        return mask

    def _find_index(self, target: float) -> int:
        if self.wavenumbers is None:
            return target
        wavenumbers = np.array(self.wavenumbers)
        return np.argmin(np.abs(wavenumbers - target))

    def _find_indices(self) -> np.ndarray:
        return np.array([self._find_index(feature) for feature in self.features])

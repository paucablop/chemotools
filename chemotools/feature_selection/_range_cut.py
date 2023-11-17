import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin
from sklearn.utils.validation import check_is_fitted


class RangeCut(BaseEstimator, SelectorMixin):
    """
    A selector that cuts the input data to a specified range. The range is specified:
    - by the indices of the start and end of the range,
    - by the wavenumbers of the start and end of the range. In this case, the wavenumbers
        must be provided to the transformer when it is initialised. If the wavenumbers
        are not provided, the indices will be used instead. The wavenumbers must be
        provided in ascending order.

    Parameters
    ----------
    start : int, optional
        The index or wavenumber of the start of the range. Default is 0.

    end : int, optional
        The index or wavenumber of the end of the range. Default is -1.

    wavenumbers : array-like, optional
        The wavenumbers of the input data. If not provided, the indices will be used
        instead. Default is None. If provided, the wavenumbers must be provided in
        ascending order.

    Attributes
    ----------
    start_index_ : int
        The index of the start of the range. It is 0 if the wavenumbers are not provided.

    end_index_ : int
        The index of the end of the range. It is -1 if the wavenumbers are not provided.


    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the input data.
    """

    def __init__(
        self,
        start: int = 0,
        end: int = -1,
        wavenumbers: np.ndarray = None,
    ):
        self.start = start
        self.end = end
        self.wavenumbers = wavenumbers

    def fit(self, X: np.ndarray, y=None) -> "RangeCut":
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
        self : RangeCut
            The fitted transformer.
        """
        # Check that X is a 2D array and has only finite values
        X = self._validate_data(X)

        # Set the start and end indices
        if self.wavenumbers is None:
            self.start_index_ = self.start
            self.end_index_ = self.end
        else:
            self.start_index_ = self._find_index(self.start)
            self.end_index_ = self._find_index(self.end)

        return self
    

    def _get_support_mask(self):
        """
        Get the boolean mask indicating which features are selected.

        Returns
        -------
        mask : np.ndarray of shape (n_features,)
            The boolean mask indicating which features are selected.
        """
        # Check that the estimator is fitted
        check_is_fitted(self, ["start_index_", "end_index_"])

        # Create the mask
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.start_index_ : self.end_index_] = True

        return mask

    def _find_index(self, target: float) -> int:
        wavenumbers = np.array(self.wavenumbers)
        return np.argmin(np.abs(wavenumbers - target))

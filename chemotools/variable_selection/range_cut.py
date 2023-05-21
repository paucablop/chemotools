import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted

from chemotools.utils.check_inputs import check_input


class RangeCut(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    """
    A transformer that cuts the input data to a specified range. The range is specified:
    - by the indices of the start and end of the range,
    - by the wavenumbers of the start and end of the range. In this case, the wavenumbers 
        must be provided to the transformer when it is initialised. If the wavenumbers
        are not provided, the indices will be used instead. The wavenumbers must be
        provided in ascending order.

    Parameters
    ----------
    wavenumbers : array-like, optional
        The wavenumbers of the input data. If not provided, the indices will be used
        instead. Default is None. If provided, the wavenumbers must be provided in
        ascending order.

    start : int, optional
        The index or wavenumber of the start of the range. Default is 0.

    end : int, optional
        The index or wavenumber of the end of the range. Default is -1.

    Attributes
    ----------
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
        wavenumbers: np.ndarray = None,
        start: int = 0,
        end: int = -1,
    ):
        self.wavenumbers = wavenumbers
        self.start = self._find_index(start)
        self.end = self._find_index(end)

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
        X = check_input(X)

        # Set the number of features
        self.n_features_in_ = X.shape[1]

        # Set the fitted attribute to True
        self._is_fitted = True

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

        # Range cut the spectra
        return X_[:, self.start : self.end]

    def _find_index(self, target: float) -> int:
        if self.wavenumbers is None:
            return target
        wavenumbers = np.array(self.wavenumbers)
        return np.argmin(np.abs(wavenumbers - target))

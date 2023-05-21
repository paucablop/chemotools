import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted

from chemotools.utils.check_inputs import check_input

class PolynomialCorrection(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    """
    A transformer that subtracts a polynomial baseline from the input data. The polynomial is 
    fitted to the points in the spectrum specified by the indices parameter.

    Parameters
    ----------
    order : int, optional
        The order of the polynomial to fit to the baseline. Defaults to 1.

    indices : list, optional
        The indices of the points in the spectrum to fit the polynomial to. Defaults to None,
        which fits the polynomial to all points in the spectrum (equivalent to detrend).

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
        Transform the input data by subtracting the polynomial baseline.

    _baseline_correct_spectrum(x)
        Subtract the polynomial baseline from a single spectrum.
    """
    def __init__(self, order: int = 1, indices: list = None) -> None:
        self.order = order
        self.indices = indices

    def fit(self, X: np.ndarray, y=None) -> "PolynomialCorrection":
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
        self : PolynomialCorrection
            The fitted transformer.
        """
        # Check that X is a 2D array and has only finite values
        X = check_input(X)

        # Set the number of features
        self.n_features_in_ = X.shape[1]

        # Set the fitted attribute to True
        self._is_fitted = True

        if self.indices is None:
            self.indices_ = range(0, len(X[0]))
        else:
            self.indices_ = self.indices

        return self
    
    def transform(self, X: np.ndarray, y:int=0, copy:bool=True) -> np.ndarray:
        """
        Transform the input data by subtracting the polynomial baseline.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to transform.

        y : int or float, optional
            Ignored.

        copy : bool, optional
            Whether to copy the input data before transforming. Defaults to True.

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

        # Calculate polynomial baseline correction
        for i, x in enumerate(X_):
            X_[i] = self._baseline_correct_spectrum(x)
        return X_.reshape(-1, 1) if X_.ndim == 1 else X_
    
    def _baseline_correct_spectrum(self, x: np.ndarray) -> np.ndarray:
        """
        Subtract the polynomial baseline from a single spectrum.

        Parameters
        ----------
        x : np.ndarray
            The spectrum to correct.

        Returns
        -------
        x : np.ndarray
            The corrected spectrum.
        """
        intensity = x[self.indices_]
        poly = np.polyfit(self.indices_, intensity, self.order)
        baseline = [np.polyval(poly, i) for i in range(0, len(x))]      
        return x - baseline
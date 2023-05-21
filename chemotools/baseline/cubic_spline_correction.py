import numpy as np
from scipy.interpolate import CubicSpline
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted

from chemotools.utils.check_inputs import check_input

class CubicSplineCorrection(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    """
    A transformer that corrects a baseline by subtracting a cubic spline through the 
    points defined by the indices.

    Parameters
    ----------
    indices : list, optional
        The indices of the features to use for the baseline correction.

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
        Transform the input data by subtracting the constant baseline value.

    """
    def __init__(self, indices: list = None) -> None:
        self.indices = indices

    def fit(self, X: np.ndarray, y=None) -> "CubicSplineCorrection":
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
        self : ConstantBaselineCorrection
            The fitted transformer.
        """
        # Check that X is a 2D array and has only finite values
        X = check_input(X)

        # Set the number of features
        self.n_features_in_ = X.shape[1]

        # Set the fitted attribute to True
        self._is_fitted = True

        if self.indices is None:
            self.indices_ = [0, len(X[0]) - 1]
        else:
            self.indices_ = self.indices

        return self

    def transform(self, X: np.ndarray, y=None, copy=True):
        """
        Transform the input data by subtracting the baseline.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to transform.

        y : None
            Ignored.

        copy : bool, optional
            Whether to copy the input data or not.

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

        # Calculate spline baseline correction
        for i, x in enumerate(X_):
            X_[i] = self._spline_baseline_correct(x)
        return X_.reshape(-1, 1) if X_.ndim == 1 else X_

    def _spline_baseline_correct(self, x: np.ndarray) -> np.ndarray:
        indices = self.indices_
        intensity = x[indices]  
        spl = CubicSpline(indices, intensity)
        baseline = spl(range(len(x)))      
        return x - baseline
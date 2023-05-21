import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted

from chemotools.utils.check_inputs import check_input


class LinearCorrection(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    """
    A transformer that corrects a baseline by subtracting a linear baseline through the
    initial and final points of the spectrum.

    Parameters
    ----------

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

    def _drift_correct_spectrum(self, x: np.ndarray) -> np.ndarray:

        # Can take any array and returns with a linear baseline correction
        # Find the x values at the edges of the spectrum
        y1: float = x[0]
        y2: float = x[-1]

        # Find the max and min wavenumebrs
        x1 = 0
        x2 = len(x)
        x_range = np.linspace(x1, x2, x2)

        # Calculate the straight line initial and end point
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        drift_correction = slope * x_range + intercept

        # Return the drift corrected spectrum
        return x - drift_correction

    def fit(self, X: np.ndarray, y=None) -> "LinearCorrection":
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

        return self

    def transform(self, X: np.ndarray, y=0, copy=True) -> np.ndarray:
        """
        Transform the input data by subtracting the constant baseline value.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to transform.

        y : int, float, optional
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

        # Calculate non-negative values
        for i, x in enumerate(X_):
            X_[i, :] = self._drift_correct_spectrum(x)
        return X_.reshape(-1, 1) if X_.ndim == 1 else X_
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted

from chemotools.utils.check_inputs import check_input


class SpectrumScale(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    """
    Scales the data by a value drawn from the uniform distribution centered
    around 1.0.

    Parameters
    ----------
    scale : float, default=0.0
        Range of the uniform distribution to draw the scaling factor from.

    random_state : int, default=None
        The random state to use for the random number generator.
    
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
        Transform the input data by scaling the spectrum.
    """


    def __init__(self, scale: int = 0.0, random_state: int = None):
        self.scale = scale
        self.random_state = random_state

    def fit(self, X: np.ndarray, y=None) -> "SpectrumScale":
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
        self : SpectrumScale
            The fitted transformer.
        """
        # Check that X is a 2D array and has only finite values
        X = check_input(X)

        # Set the number of features
        self.n_features_in_ = X.shape[1]

        # Set the fitted attribute to True
        self._is_fitted = True

        # Instantiate the random number generator
        self._rng = np.random.default_rng(self.random_state)

        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Transform the input data by scaling the spectrum.

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

        # Calculate the scaled spectrum
        for i, x in enumerate(X_):
            X_[i] = self._scale_spectrum(x)

        return X_.reshape(-1, 1) if X_.ndim == 1 else X_

    def _scale_spectrum(self, x) -> np.ndarray:
        scaling_factor = self._rng.uniform(low=1-self.scale, high=1+self.scale)
        return np.multiply(x, scaling_factor)
    
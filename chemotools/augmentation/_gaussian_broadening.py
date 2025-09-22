"""
The :mod:`chemotools.augmentation._gaussian_broadening` module implements the GaussianBroadening
transformer to broaden peaks in spectral data using Gaussian convolution.
"""

# Authors: Pau Cabaneros
# License: MIT

from typing import Literal, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils._param_validation import Interval, Real, StrOptions


class GaussianBroadening(TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """
    Transform spectral data by broadening peaks using Gaussian convolution.

    This transformer applies Gaussian smoothing to broaden peaks in spectral data.
    For each signal, a random sigma is chosen between 0 and the specified sigma value.

    Parameters
    ----------
    sigma : float, default=1.0
        Maximum standard deviation for the Gaussian kernel.
        The actual sigma used will be randomly chosen between 0 and this value.

    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, default='reflect'
        The mode parameter determines how the input array is extended when
        the filter overlaps a border. Default is 'reflect'.

    pad_value : float, default=0.0
        Value to fill past edges of input if mode is 'constant'.

    random_state : int, optional, default=None
        Random state for reproducible sigma selection.

    truncate : float, default=4.0
        Truncate the filter at this many standard deviations.
        Larger values increase computation time but improve accuracy.

    Examples
    --------
    >>> from chemotools.augmentation import GaussianBroadening
    >>> from chemotools.datasets import load_fermentation_train
    >>> # Load sample data
    >>> X, _ = load_fermentation_train()
    >>> # Instantiate the transformer
    >>> transformer = GaussianBroadening(sigma=2.0, mode="reflect")
    GaussianBroadening()
    >>> transformer.fit(X)
    >>> # Generate broadened data
    >>> X_broadened = transformer.transform(X)
    """

    _parameter_constraints: dict = {
        "sigma": [Interval(Real, 0, None, closed="both")],
        "mode": StrOptions({"reflect", "constant", "nearest", "mirror", "wrap"}),
        "pad_value": [Real],
        "random_state": [None, int, np.random.RandomState],
        "truncate": [Interval(Real, 0, None, closed="both")],
    }

    def __init__(
        self,
        sigma: float = 1.0,
        mode: Literal["reflect", "constant", "nearest", "mirror", "wrap"] = "reflect",
        pad_value: float = 0.0,
        random_state: Optional[int] = None,
        truncate: float = 4.0,
    ):
        self.sigma = sigma
        self.mode = mode
        self.pad_value = pad_value
        self.random_state = random_state
        self.truncate = truncate

    def fit(self, X: np.ndarray, y=None) -> "GaussianBroadening":
        """
        Fit the transformer to the data (in this case, only validates input).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to validate.

        y : None
            Ignored.

        Returns
        -------
        self : GaussianBroadening
            The fitted transformer.
        """
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        # Validate sigma parameter
        if not isinstance(self.sigma, (int, float)):
            raise ValueError("sigma must be a number")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")

        # Initialize random number generator
        self._rng = check_random_state(self.random_state)

        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Apply Gaussian broadening to the input data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The data to transform.

        y : None
            Ignored.

        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_features)
            The transformed data with broadened peaks.
        """
        check_is_fitted(self, "n_features_in_")
        X_ = validate_data(
            self,
            X,
            y="no_validation",
            ensure_2d=True,
            copy=True,
            reset=False,
            dtype=np.float64,
        )

        # Transform each sample
        for i, x in enumerate(X_):
            X_[i] = self._broaden_signal(x)

        return X_

    def _broaden_signal(self, x: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian broadening to a single signal.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            The input signal to broaden.

        Returns
        -------
        broadened_signal : ndarray of shape (n_features,)
            The broadened signal.
        """
        # Randomly choose sigma between 0 and max sigma
        sigma = self._rng.uniform(0, self.sigma)

        # Apply Gaussian filter
        return gaussian_filter1d(
            x, sigma=sigma, mode=self.mode, cval=self.pad_value, truncate=self.truncate
        )

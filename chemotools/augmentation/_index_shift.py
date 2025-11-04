"""
The :mod:`chemotools.augmentation._index_shift` module implements the IndexShift
transformer to randomly shift spectral data along the index axis.
"""

# Authors: Pau Cabaneros
# License: MIT

from typing import Literal, Optional

import numpy as np
from scipy.signal import convolve
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils._param_validation import Interval, Real, StrOptions


class IndexShift(TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """
    Shift the spectrum a given number of indices between -shift and +shift drawn
    from a discrete uniform distribution.

    Parameters
    ----------
    shift : int, default=0
        Maximum number of indices by which the data is randomly shifted.
        The actual shift is a random integer between -shift and shift (inclusive).

    padding_mode : {'zeros', 'constant', 'wrap', 'extend', 'mirror', 'linear'}, default='linear'
        Specifies how to handle padding when shifting the data:
            - 'zeros': Pads with zeros.
            - 'constant': Pads with a constant value defined by `pad_value`.
            - 'wrap': Circular shift (wraps around).
            - 'extend': Extends using edge values.
            - 'mirror': Mirrors the signal.
            - 'linear': Uses linear regression to extrapolate values.

    pad_value : float, default=0.0
        The value used for padding when `padding_mode='constant'`.

    random_state : int, optional, default=None
        The random seed for reproducibility.

    Attributes
    ----------
    n_features_in_ : int
        The number of features in the input data.

    Examples
    --------
    >>> from chemotools.augmentation import IndexShift
    >>> from chemotools.datasets import load_fermentation_train
    >>> # Load sample data
    >>> X, _ = load_fermentation_train()
    >>> # Instantiate the transformer
    >>> transformer = IndexShift(shift=2, padding_mode="constant",)
    IndexShift()
    >>> transformer.fit(X)
    >>> # Generate shifted data
    >>> X_shifted = transformer.transform(X)
    """

    _parameter_constraints: dict = {
        "shift": [Interval(Real, 0, None, closed="both")],
        "padding_mode": [
            StrOptions({"zeros", "constant", "wrap", "extend", "mirror", "linear"})
        ],
        "pad_value": [Real],
        "random_state": [None, int, np.random.RandomState],
    }

    def __init__(
        self,
        shift: int = 0,
        padding_mode: Literal[
            "zeros", "constant", "wrap", "extend", "mirror", "linear"
        ] = "linear",
        pad_value: float = 0.0,
        random_state: Optional[int] = None,
    ):
        self.shift = shift
        self.padding_mode = padding_mode
        self.pad_value = pad_value
        self.random_state = random_state

    def fit(self, X: np.ndarray, y=None) -> "IndexShift":
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
        self : IndexShift
            The fitted transformer.
        """
        # Check that X is a 2D array and has only finite values
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        # Instantiate the random number generator
        self._rng = check_random_state(self.random_state)

        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Transform the input data by shifting the spectrum.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to transform.

        y : None
            Ignored.

        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_features)
            The transformed data with the applied shifts.
        """
        # Check that the estimator is fitted
        check_is_fitted(self, "n_features_in_")

        # Check that X is a 2D array and has only finite values
        X_ = validate_data(
            self,
            X,
            y="no_validation",
            ensure_2d=True,
            copy=True,
            reset=False,
            dtype=np.float64,
        )

        # Calculate the standard normal variate
        for i, x in enumerate(X_):
            X_[i] = self._shift_signal(x)

        return X_.reshape(-1, 1) if X_.ndim == 1 else X_

    def _shift_signal(self, x: np.ndarray):
        """
        Shifts a discrete signal using convolution with a Dirac delta kernel.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
            The input signal to shift.

        Returns
        -------
        result : np.ndarray of shape (n_features,)
            The shifted signal.
        """
        shift = self._rng.randint(-self.shift, self.shift + 1)

        if self.padding_mode == "wrap":
            return np.roll(x, shift)

        # Create Dirac delta kernel with proper dimensions

        if shift >= 0:
            kernel = np.zeros(shift + 1)
            kernel[-1] = 1
        else:
            kernel = np.zeros(-shift + 1)
            kernel[0] = 1

        # Convolve signal with kernel
        shifted = convolve(x, kernel, mode="full")

        if shift >= 0:
            result = shifted[: len(x)] if x.ndim == 1 else shifted[: x.shape[0]]
            pad_length = shift
            pad_left = True
        else:
            result = shifted[-len(x) :] if x.ndim == 1 else shifted[-x.shape[0] :]
            pad_length = -shift
            pad_left = False

        if self.padding_mode == "zeros":
            return result

        elif self.padding_mode == "constant":
            mask = np.abs(result) < 1e-10
            result[mask] = self.pad_value
            return result

        elif self.padding_mode == "mirror":
            if pad_left:
                pad_values = x[pad_length - 1 :: -1]
                result[:pad_length] = pad_values[-pad_length:]
            else:
                pad_values = x[:-1][::-1]
                result[-pad_length:] = pad_values[:pad_length]

            return result

        elif self.padding_mode == "extend":
            if pad_left:
                result[:pad_length] = x[0]
            else:
                result[-pad_length:] = x[-1]
            return result

        elif self.padding_mode == "linear":
            # Get points for linear regression
            if pad_left:
                points = x[: pad_length + 1]  # Take first pad_length+1 points
                x_coords = np.arange(len(points))
                slope, intercept, *_ = stats.linregress(x_coords, points)

                # Generate new points using linear regression
                new_x = np.arange(-pad_length, 0)
                extrapolated = slope * new_x + intercept
                result[:pad_length] = extrapolated
            else:
                points = x[-pad_length - 1 :]  # Take last pad_length+1 points
                x_coords = np.arange(len(points))
                slope, intercept, *_ = stats.linregress(x_coords, points)

                # Generate new points using linear regression
                new_x = np.arange(len(points), len(points) + pad_length)
                extrapolated = slope * new_x + intercept
                result[-pad_length:] = extrapolated
            return result

        else:
            raise ValueError(
                f"Unknown padding mode: {self.padding_mode}. Please choose from 'zeros', 'constant', 'wrap', 'extend', 'mirror', or 'linear'."
            )

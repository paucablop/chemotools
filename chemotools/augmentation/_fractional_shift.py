from typing import Literal, Optional

import numpy as np
from scipy.interpolate import CubicSpline
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted, validate_data

class FractionalShift(TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """
    Shift the spectrum by a fractional amount, allowing shifts below one index.

    Parameters
    ----------
    shift : float, default=0.0
        Maximum amount by which the data is randomly shifted.
        The actual shift is a random float between -shift and shift.

    padding_mode : {'zeros', 'constant', 'wrap', 'extend', 'mirror', 'linear'}, default='linear'
        Specifies how to handle padding when shifting the data:
            - 'zeros': Pads with zeros.
            - 'constant': Pads with a constant value defined by `pad_value`.
            - 'wrap': Circular shift (wraps around).
            - 'extend': Extends using edge values.
            - 'mirror': Mirrors the signal.
            - 'linear': Uses linear regression on 5 points to extrapolate values.

    pad_value : float, default=0.0
        The value used for padding when `padding_mode='constant'`.

    random_state : int, optional, default=None
        The random seed for reproducibility.
    """

    def __init__(self, shift: float = 0.0, padding_mode: Literal[
                    "zeros", "constant", "extend", "mirror", "linear"] = "linear", 
                    pad_value: float = 0.0, random_state: Optional[int] = None):
        self.shift = shift
        self.padding_mode = padding_mode
        self.pad_value = pad_value
        self.random_state = random_state

    def fit(self, X: np.ndarray, y=None) -> "FractionalShift":
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
        self : FractionalShift
            The fitted transformer.
        """
        X = validate_data(self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64)
        self._rng = np.random.default_rng(self.random_state)
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
        X_ : np.ndarray of shape (n_samples, n_features)
            The transformed data with the applied shifts.
        """
        check_is_fitted(self, "n_features_in_")
        X_ = validate_data(self, X, y="no_validation", ensure_2d=True, copy=True, reset=False, dtype=np.float64)

        for i, x in enumerate(X_):
            X_[i] = self._shift_signal(x)
        
        return X_.reshape(-1, 1) if X_.ndim == 1 else X_

    def _shift_signal(self, x: np.ndarray) -> np.ndarray:
        """
        Shifts a signal by a fractional amount using cubic spline interpolation.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
            The input signal to shift.

        Returns
        -------
        shifted_signal : np.ndarray of shape (n_features,)
            The shifted signal.
        """
        shift = self._rng.uniform(-self.shift, self.shift)
        n = len(x)
        indices = np.arange(n)
        shifted_indices = indices + shift
        
        # Create cubic spline interpolator
        spline = CubicSpline(indices, x, bc_type='not-a-knot')
        shifted_signal = spline(shifted_indices)

        # Determine padding direction and length
        if shift >= 0:
            pad_length = len(shifted_indices[shifted_indices >= n-1])
            pad_left = False
        else:
            pad_length = len(shifted_indices[shifted_indices < 0])
            pad_left = True

        # Handle padding based on mode
        if self.padding_mode == "zeros":
            shifted_signal[shifted_indices < 0] = 0
            shifted_signal[shifted_indices >= n-1] = 0
        
        elif self.padding_mode == "constant":
            shifted_signal[shifted_indices < 0] = self.pad_value
            shifted_signal[shifted_indices >= n-1] = self.pad_value
        
        elif self.padding_mode == "mirror":
            if pad_left:
                pad_values = x[pad_length-1::-1]
                shifted_signal[shifted_indices < 0] = pad_values[:pad_length]
            else:
                pad_values = x[:-1][::-1]
                shifted_signal[shifted_indices >= n-1] = pad_values[:pad_length]
        
        elif self.padding_mode == "extend":
            if pad_left:
                shifted_signal[shifted_indices < 0] = x[0]
            else:
                shifted_signal[shifted_indices >= n-1] = x[-1]
        
        elif self.padding_mode == "linear":
            if pad_left:
                # Use first 5 points for regression
                if len(x) < 5:
                    points = x[:len(x)]  # Use all points if less than 5
                else:
                    points = x[:5]
                x_coords = np.arange(len(points))
                
                # Reshape arrays for linregress
                x_coords = x_coords.reshape(-1)
                points = points.reshape(-1)
                
                # Perform regression
                slope, intercept, _, _, _ = stats.linregress(x_coords, points)
                
                # Generate new points using linear regression
                new_x = np.arange(-pad_length, 0)
                extrapolated = slope * new_x + intercept
                shifted_signal[shifted_indices < 0] = extrapolated
            else:
                # Use last 5 points for regression
                if len(x) < 5:
                    points = x[-len(x):]  # Use all points if less than 5
                else:
                    points = x[-5:]
                x_coords = np.arange(len(points))
                
                # Reshape arrays for linregress
                x_coords = x_coords.reshape(-1)
                points = points.reshape(-1)
                
                # Perform regression
                slope, intercept, _, _, _ = stats.linregress(x_coords, points)
                
                # Generate new points using linear regression
                new_x = np.arange(len(points), len(points) + pad_length)
                extrapolated = slope * new_x + intercept
                shifted_signal[shifted_indices >= n] = extrapolated
        
        else:
            raise ValueError(f"Unknown padding mode: {self.padding_mode}")

        return shifted_signal
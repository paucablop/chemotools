"""
The :mod:`chemotools.smooth._savitzky_golay_filter` module implements the Savitzky-Golay Filter (SGF) transformation.
"""

# Authors: Nusret Emirhan Salli <nusret.emirhan.salli@gmail.com>, Pau Cabaneros
# License: MIT

from typing import Literal, Optional
from numbers import Integral

import numpy as np
from sklearn.utils._param_validation import Interval, StrOptions
from ._base import _BaseFIRFilter


class SavitzkyGolayFilter(_BaseFIRFilter):
    """
    A transformer that calculates the Savitzky-Golay filter of the input data.

    Parameters
    ----------
    window_size : int, optional
        The size of the window to use for the Savitzky-Golay filter. Must be odd. Default
        is 3.

    polynomial_order : int, optional
        The order of the polynomial to use for the Savitzky-Golay filter. Must be less
        than window_size. Default is 1.

    mode : str, optional
        The mode to use for the Savitzky-Golay filter. Can be "nearest", "constant",
        "reflect", "wrap", "mirror" or "interp". Default is "nearest".

    Attributes
    ----------
    n_features_in_ : int
        The number of features in the training data.

    Examples
    --------
    >>> from chemotools.datasets import load_fermentation_train
    >>> from chemotools.smooth import SavitzkyGolayFilter
    >>> # Load sample data
    >>> X, _ = load_fermentation_train()
    >>> # Initialize SavitzkyGolayFilter
    >>> sgf = SavitzkyGolayFilter()
    SavitzkyGolayFilter()
    >>> # Fit and transform the data
    >>> X_smoothed = sgf.fit_transform(X)

    """

    _parameter_constraints: dict = {
        "window_size": [Interval(Integral, 1, None, closed="left")],
        "polynomial_order": [Interval(Integral, 0, None, closed="left")],
        "mode": [StrOptions({"mirror", "constant", "nearest", "wrap", "interp"})],
        "axis": [Interval(Integral, 0, None, closed="left")],
    }

    def __init__(
        self,
        window_size: int = 3,
        polynomial_order: int = 1,
        mode: Literal["mirror", "constant", "nearest", "wrap", "interp"] = "nearest",
        axis: int = 1,
    ) -> None:
        super().__init__(window_size=window_size, mode=mode, axis=axis)
        self.polynomial_order = polynomial_order

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "SavitzkyGolayFilter":
        """
        Fit the Savitzky-Golay filter to the data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to fit the transformer.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : SavitzkyGolayFilter
            The fitted transformer.
        """
        return super().fit(X, y)

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Transform the input data by applying the Savitzky-Golay filter.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to transform.

        y : None
            Ignored to align with API.

        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_features)
            The transformed data.
        """
        return super().transform(X, y)

    def _compute_kernel(self) -> np.ndarray:
        if self.polynomial_order >= self.window_size:
            raise ValueError("polynomial_order must be < window_size.")
        # Prefer SciPy's reference coefficients in convolution form
        try:
            from scipy.signal import savgol_coeffs

            k = np.asarray(
                savgol_coeffs(
                    self.window_size, self.polynomial_order, deriv=0, use="conv"
                ),
                dtype=np.float64,
            )
        except Exception:
            # Robust LS fallback (intercept row of (A^T A)^{-1} A^T)
            m = (self.window_size - 1) // 2
            i = np.arange(-m, m + 1, dtype=np.float64)
            A = np.vander(i, N=self.polynomial_order + 1, increasing=True)
            ATA_inv = np.linalg.pinv(A.T @ A)
            k = (ATA_inv @ A.T)[0, :]
            k = 0.5 * (k + k[::-1])
        k /= k.sum()
        return k

    def _get_interp_order(self) -> int:
        """
        Return polynomial order for Savitzky-Golay extrapolation.

        This enables scipy-compatible polynomial extrapolation when mode='interp'.

        Returns
        -------
        int
            The polynomial order used in the Savitzky-Golay filter.
        """
        return self.polynomial_order

    def _apply_filter_1d(self, x: np.ndarray) -> np.ndarray:
        """
        Override to use scipy-compatible edge handling for mode='interp'.

        For Savitzky-Golay with mode='interp', scipy uses a special algorithm:
        - Fits polynomial to edge windows
        - Evaluates polynomial values directly at edge points (no convolution!)
        - Uses standard convolution for middle section
        """
        if self.mode == "interp":
            # Use scipy's exact algorithm for interp mode
            return self._apply_sg_interp(x)
        else:
            # Use base class pad-then-convolve for other modes
            return super()._apply_filter_1d(x)

    def _apply_sg_interp(self, x: np.ndarray) -> np.ndarray:
        """
        Apply Savitzky-Golay filter with scipy-compatible interp mode.

        Replicates scipy's _fit_edges_polyfit behavior:
        1. Fit polynomial to first window_size points, evaluate at first half_window positions
        2. Apply standard convolution to ALL positions (with padding)
        3. Replace first half_window and last half_window positions with polynomial values
        """
        m = self._half_
        n = len(x)
        window_len = self.window_size

        if n < window_len:
            # Signal too short for interp mode, fall back to edge padding
            x_padded = np.pad(x, (m, m), mode="edge")
            return np.convolve(x_padded, self.kernel_, mode="valid")

        # Apply standard convolution to entire signal (with edge padding for the convolution itself)
        x_padded = np.pad(x, (m, m), mode="edge")
        result = np.convolve(x_padded, self.kernel_, mode="valid")

        # Now replace edges with polynomial-fitted values

        # Left edge: fit polynomial to first window and evaluate at edge positions
        x_left_window = x[:window_len]
        poly_left = np.polyfit(
            np.arange(window_len, dtype=np.float64),
            x_left_window,
            self.polynomial_order,
        )
        indices_left = np.arange(m, dtype=np.float64)
        result[:m] = np.polyval(poly_left, indices_left)

        # Right edge: fit polynomial to last window and evaluate at edge positions
        x_right_window = x[-window_len:]
        poly_right = np.polyfit(
            np.arange(window_len, dtype=np.float64),
            x_right_window,
            self.polynomial_order,
        )
        # Evaluate at the last m positions: [window_len-m, ..., window_len-1]
        indices_right = np.arange(window_len - m, window_len, dtype=np.float64)
        result[-m:] = np.polyval(poly_right, indices_right)

        return result

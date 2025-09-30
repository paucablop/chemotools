from typing import Literal

import numpy as np
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

    def __init__(
        self,
        window_size: int = 3,
        polynomial_order: int = 1,
        mode: Literal["mirror", "constant", "nearest", "wrap", "interp"] = "nearest",
        axis: int = 1,
    ) -> None:
        super().__init__(window_size=window_size, mode=mode, axis=axis)
        self.polynomial_order = polynomial_order

    def _compute_kernel(self) -> np.ndarray:
        if self.polynomial_order >= self.window_size:
            raise ValueError("polynomial_order must be < window_size.")
        # Prefer SciPy’s reference coefficients in convolution form
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

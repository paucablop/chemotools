from typing import Literal

import numpy as np
from ._base import _BaseFIRFilter


class SavitzkyGolayFilter(_BaseFIRFilter):
    """
    Savitzky–Golay smoother via FIR coefficients so it shares the FIR base.

    Parameters
    ----------
    window_size : int, odd >= 3
    polynomial_order : int, < window_size
    mode, axis : inherited
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

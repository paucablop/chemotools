
from __future__ import annotations
from typing import Literal, Optional
import numbers
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils._param_validation import Interval, StrOptions





class ModifiedSincFilter(TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """
    A transformer that smooths each row by convolving with a Modified-Sinc
    (windowed-sinc) kernel, with optional passband flattening.

    This follows Schmid et al.'s recommendation to replace SG smoothing with
    a modified-sinc kernel and to handle boundaries via extrapolation + convolution. [1]

    Parameters
    ----------
    window_size : int, optional (default=21)
        Odd number of taps in the FIR kernel (>= 3). Larger => stronger smoothing.

    n : int, optional (default=6)
        Even "order" that sets zeros of the sinc core so that the kernel vanishes at x=±1.

    alpha : float, optional (default=3.0)
        Gaussian window strength; larger -> stronger taper (more stopband suppression).

    mode : {"mirror", "constant", "nearest", "wrap", "interp"}, optional (default="interp")
        Boundary handling. "interp" performs **linear extrapolation** at both ends before
        convolution (recommended in [1]). Others map to NumPy-like padding.

    flatten_passband : bool, optional (default=True)
        If True, apply a small correction to reduce the first few even moments of the kernel,
        which flattens low-frequency gain (helps preserve peak heights).

    n_corrections : int, optional (default=2)
        Number of even moments to target (μ2, μ4, ...). Up to 3 is reasonable.

    Attributes
    ----------
    kernel_ : ndarray of shape (window_size,)
        The symmetric, DC-preserving FIR kernel.

    References
    ----------
    [1] Schmid, M.; Rath, D.; Diebold, U. "Why and How Savitzky–Golay Filters Should Be
        Replaced", ACS Meas. Sci. Au (2022): advocates a modified-sinc smoother and linear
        extrapolation at the boundaries before convolution. Also see PubMed/PMC. 
    """

    # Scikit-learn param validation (optional but nice)
    _parameter_constraints: dict = {
        "window_size": [Interval(numbers.Integral, 3, None, closed="left")],
        "n": [Interval(numbers.Integral, 4, None, closed="left")],
        "alpha": [Interval(numbers.Real, 0, None, closed="left")],
        "mode": [StrOptions({"mirror", "constant", "nearest", "wrap", "interp"})],
        "flatten_passband": [bool],
        "n_corrections": [Interval(numbers.Integral, 0, 10, closed="both")],
    }

    def __init__(
        self,
        window_size: int = 21,
        n: int = 6,
        alpha: float = 3.0,
        mode: Literal["mirror", "constant", "nearest", "wrap", "interp"] = "interp",
        flatten_passband: bool = True,
        n_corrections: int = 2,
    ) -> None:
        self.window_size = window_size
        self.n = n
        self.alpha = alpha
        self.mode = mode
        self.flatten_passband = flatten_passband
        self.n_corrections = n_corrections

    # ------------------------------------------------------------------ #
    # sklearn API
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "ModifiedSincFilter":
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )
        self.kernel_ = self._build_kernel()
        self._half_ = (self.kernel_.size - 1) // 2
        return self

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        check_is_fitted(self, "kernel_")
        X_ = validate_data(
            self,
            X,
            y="no_validation",
            ensure_2d=True,
            copy=True,
            reset=False,
            dtype=np.float64,
        )

        for i, row in enumerate(X_):
            X_[i] = self._apply_filter_1d(row)

        return X_.reshape(-1, 1) if X_.ndim == 1 else X_

    # ------------------------------------------------------------------ #
    # Kernel construction (paper-aligned)
    # ------------------------------------------------------------------ #
    def _build_kernel(self) -> np.ndarray:
        if self.window_size % 2 == 0:
            raise ValueError("window_size must be odd.")
        if self.n % 2 != 0:
            raise ValueError("n must be even (so the sinc has zeros at the window edges).")
        if self.alpha <= 0:
            raise ValueError("alpha must be > 0.")

        m = (self.window_size - 1) // 2
        i = np.arange(-m, m + 1, dtype=np.float64)
        x = i / max(m, 1)  # normalized to [-1, 1]

        # Base: windowed-sinc — Gaussian-tapered sinc((n/2)*x).  [Sinc/window background: 2,3]
        core = np.sinc(0.5 * self.n * x)
        window = np.exp(-self.alpha * x * x)
        h = core * window

        # Optional: passband flattening via low-order even-moment correction.
        # We add small, symmetric basis functions that vanish at the edges:
        #   b_j(x) = window(x) * x * sin((2j+1) * pi * x)   (even overall)
        # Choose κ to reduce μ2, μ4, ... (moments of i^(2r)), which flattens |H(ω)| near ω=0.
        if self.flatten_passband and self.n_corrections > 0:
            B = []
            for j in range(self.n_corrections):
                bj = window * x * np.sin((2 * j + 1) * np.pi * x)
                # enforce exact symmetry numerically
                bj = 0.5 * (bj + bj[::-1])
                B.append(bj)
            B = np.vstack(B) if B else np.zeros((0, h.size))

            # Build linear system on even moments μ2, μ4, ...
            # μ_{2r}(v) = sum_k v[k] * i[k]^(2r).
            targets = []
            A = []
            for r in range(1, self.n_corrections + 1):
                pow_vec = (i.astype(np.float64) ** (2 * r))
                mu_base = np.sum(h * pow_vec)
                targets.append(-mu_base)
                A.append(np.sum(B * pow_vec, axis=1))
            if len(A) > 0:
                A = np.vstack(A)
                targets = np.asarray(targets, dtype=np.float64)
                # Solve A @ kappa = targets (least squares is fine; values are tiny).
                kappa, *_ = np.linalg.lstsq(A, targets, rcond=None)
                h = h + kappa @ B

        # Final symmetry + DC normalization (preserve constants).
        h = 0.5 * (h + h[::-1])
        s = np.sum(h)
        if not np.isfinite(s) or abs(s) < 1e-15:
            raise FloatingPointError("Kernel normalization failed; try different parameters.")
        h = h / s

        return h

    # ------------------------------------------------------------------ #
    # Convolution + boundary handling
    # ------------------------------------------------------------------ #
    def _apply_filter_1d(self, x: np.ndarray) -> np.ndarray:
        k = self.kernel_
        m = self._half_
        xp = self._pad_1d(x, m)
        # Valid on padded array gives "same" length as original.
        y = np.convolve(xp, k, mode="valid")
        return y

    def _pad_1d(self, x: np.ndarray, m: int) -> np.ndarray:
        if m == 0:
            return x.copy()
        mode = self.mode
        if mode == "interp":
            # Linear extrapolation with slope from boundary neighbor samples. [1]
            if x.size < 2:
                left = np.repeat(x[0], m)
                right = np.repeat(x[-1], m)
            else:
                ls = x[1] - x[0]
                rs = x[-1] - x[-2]
                left = x[0] - ls * np.arange(m, 0, -1, dtype=np.float64)
                right = x[-1] + rs * np.arange(1, m + 1, dtype=np.float64)
            return np.concatenate([left, x, right], axis=0)
        elif mode == "nearest":
            return np.pad(x, (m, m), mode="edge")
        elif mode == "mirror":
            # NumPy "reflect" mirrors without repeating edge value (close to SciPy's SG "mirror")
            return np.pad(x, (m, m), mode="reflect")
        elif mode == "wrap":
            return np.pad(x, (m, m), mode="wrap")
        elif mode == "constant":
            return np.pad(x, (m, m), mode="constant", constant_values=(x[0], x[-1]))
        else:
            raise ValueError(f"Unknown mode='{mode}'")
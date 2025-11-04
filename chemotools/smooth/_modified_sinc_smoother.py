"""
The :mod:chemotools.smooth._modified_sinc_smoother module implements the Modified Sinc Filter (MSF) transformation.
"""

# Authors: Nusret Emirhan Salli <nusret.emirhan.salli@gmail.com>
# License: MIT

from __future__ import annotations
from typing import Literal, Optional
import numpy as np
from sklearn.utils._param_validation import Integral, Interval, StrOptions, Real

from ._base import _BaseFIRFilter


class ModifiedSincFilter(_BaseFIRFilter):
    """
    Modified Sinc smoothing (MS) for denoising while preserving peak positions based on the paper "Why and How Savitzky–Golay Filters Should Be Replaced."

    The Modified Sinc smoother is a linear-phase FIR filter: the signal is
    convolved with a fixed, symmetric kernel. The kernel is built from:
        1) a sinc core with argument ((n + 4) / 2) * pi * x (Eq. 3, p. 187),
        2) a special Gaussian-like window w(x) whose value and slope vanish at
            the window ends (Eq. 4, p. 187), and
        3) small optional correction terms that flatten the passband so low
            frequencies are almost unattenuated (Eqs. 7–8 and Table 1, pp. 187–188).

    Parameters
    ----------
    window_size : int, default=21
        Odd number of taps (2*m + 1). Larger values give stronger smoothing.

    n : int, default=6
        Even integer >= 2. Controls how many inner zeros the sinc has within
        the window. The paper discusses n = 2, 4, 6, 8, 10 (Fig. 2).

    alpha : float, default=4.0
        Positive window width parameter. Larger alpha reduces side lobes more
        aggressively (steeper roll-off).

    use_corrections : bool, default=True
        If True and n in {6, 8, 10} and the window is large enough, add the
        small passband-flattening terms from Eqs. 7–8 (coefficients from Table 1).

    mode : {"mirror", "constant", "nearest", "wrap", "interp"}, default="interp"
        Boundary strategy, passed to the base FIR class. "interp" performs
        linear extrapolation (recommended in the paper).

    axis : int, default=1
        Axis along which to smooth for 2D inputs (rows x features). Use 1 to
        smooth within each row.

    Methods
    -------
    fit(X, y=None)
        Inherited from the base class. Validates input and builds the kernel.

    transform(X, y=None)
        Inherited from the base class. Pads, convolves, and returns the smoothed data.

    References
    ----------
    [1] Schmid, M.; Rath, D.; Diebold, U. "Why and How Savitzky–Golay Filters Should Be Replaced."
    ACS measurement science Au 2022, 2 (2), 185-196.

    Examples
    --------
    >>> from chemotools.datasets import load_fermentation_train
    >>> from chemotools.smooth import ModifiedSincFilter
    >>> # Load sample data
    >>> X, _ = load_fermentation_train()
    >>> # Initialize ModifiedSincFilter
    >>> msf = ModifiedSincFilter()
    ModifiedSincFilter()
    >>> # Fit and transform the data
    >>> X_smoothed = msf.fit_transform(X)
    """

    _parameter_constraints: dict = {
        "window_size": [Interval(Integral, 1, None, closed="left")],
        "n": [Interval(Integral, 2, None, closed="left")],
        "alpha": [Interval(Real, 0, None, closed="neither")],
        "use_corrections": ["boolean"],
        "mode": [StrOptions({"mirror", "constant", "nearest", "wrap", "interp"})],
        "axis": [Interval(Integral, 0, None, closed="left")],
    }

    def __init__(
        self,
        window_size: int = 21,
        n: int = 6,
        alpha: float = 4.0,
        use_corrections: bool = True,
        mode: Literal["mirror", "constant", "nearest", "wrap", "interp"] = "interp",
        axis: int = 1,
    ) -> None:
        super().__init__(window_size=window_size, mode=mode, axis=axis)
        self.n = n
        self.alpha = alpha
        self.use_corrections = use_corrections

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "ModifiedSincFilter":
        """
        Fit the Modified Sinc Filter to the data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to fit the transformer.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : ModifiedSincFilter
            The fitted transformer.
        """
        return super().fit(X, y)

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Transform the input data by applying the Modified Sinc Filter.

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
        """
        Build the Modified Sinc kernel h[i] for i = -m ... m.

        Implementation map to the paper:
          1) Map index to x = i / (m + 1)  (Eq. 5).
          2) Base sinc: np.sinc(0.5 * (n + 4) * x)  (Eq. 3).
          3) Solve a 3x3 system for the window coefficients so that
             w(0)=1, w(1)=0, w'(1)=0; then we form w(x)  (Eq. 4).
          4) h = sinc * w.
          5) Optional corrections from based on Eqs. 7–8 (Table 1).
          6) Symmetrize and normalize so sum(h)=1  (Eq. 6).
        """

        # ---- parameter checks ----
        if self.n % 2 != 0 or self.n < 2:
            raise ValueError("n must be an even integer >= 2.")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive.")

        # ---- 1) Eq. 5: index -> normalized coordinate in [-1, 1] ----
        m = (self.window_size - 1) // 2
        i = np.arange(-m, m + 1, dtype=np.float64)
        x = i / (m + 1) if m >= 0 else np.array([0.0])

        # ---- 2) Eq. 3: modified sinc core (note that numpy's sinc uses sin(pi*u)/(pi*u)) ----
        core = np.sinc(0.5 * (self.n + 4) * x)

        # ---- 3) Eq. 4: window with w(0)=1, w(1)=0, w'(1)=0 ----
        # Precompute the exponentials appearing in those constraints.
        E1 = np.exp(
            -self.alpha * 1.0
        )  # exp(-alpha * 1^2)   at x = 1 (central Gaussian)
        Ep = np.exp(-self.alpha * 1.0)  # exp(-alpha * (1-2)^2) = exp(-alpha) for (x-2)
        Em = np.exp(
            -self.alpha * 9.0
        )  # exp(-alpha * (1+2)^2) = exp(-9*alpha) for (x+2)
        e4 = np.exp(-self.alpha * 4.0)  # exp(-alpha * (±2)^2) at x = 0

        # Linear system rows correspond to: w(0)=1, w(1)=0, w'(1)=0
        M = np.array(
            [
                [1.0, 2.0 * e4, 1.0],
                [E1, (Ep + Em), 1.0],
                [-2 * self.alpha * E1, 2 * self.alpha * (Ep - 3 * Em), 0.0],
            ],
            dtype=np.float64,
        )
        # solve the system using linear algebra.
        rhs = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        Acoef, Bcoef, Ccoef = np.linalg.solve(M, rhs)

        # Window values calculated
        window = (
            Acoef * np.exp(-self.alpha * x**2)
            + Bcoef
            * (
                np.exp(-self.alpha * (x - 2.0) ** 2)
                + np.exp(-self.alpha * (x + 2.0) ** 2)
            )
            + Ccoef
        )

        # ---- 4) base kernel = windowed sinc ----
        h = core * window

        # ---- 5) Eqs. 7–8 + Table 1: optional passband-flattening corrections ----
        if (
            self.use_corrections
            and self._has_kappa_table(self.n)
            and (m >= self.n // 2 + 2)
        ):
            # nu = 1 for n/2 odd (n=6,10); nu = 2 for n=8 (Eq. 7)
            nu = 1 if ((self.n // 2) % 2 == 1) else 2
            kappas = self._kappa_coeffs(
                self.n, m
            )  # kappa = a + b / (c - m)^3  (Eq. 8; Table 1)
            corr = 0.0
            # add the correction term according to Eq. 7
            for j, kappa in enumerate(kappas):
                corr += kappa * window * x * np.sin((2 * j + nu) * np.pi * x)
            h = h + corr

        # ---- 6) Eq. 6: Enforce symmetry and Direct Current = 1 ----
        h = 0.5 * (h + h[::-1])  # make it symmetric
        s = h.sum()
        if not np.isfinite(s) or abs(s) < 1e-15:
            raise FloatingPointError(
                "Kernel normalization failed; try different parameters."
            )
        h = h / s  # Normalize so sum = 1
        return h

    # ====== Table 1 (p. 188): kappa fit coefficients for Eq. 8 ======
    @staticmethod
    def _has_kappa_table(n: int) -> bool:
        # The paper provides corrections for n = 6, 8, 10
        return n in (6, 8, 10)

    @staticmethod
    def _kappa_coeffs(n: int, m: int) -> np.ndarray:
        """
        Return [kappa_0] for n=6; [kappa_0, kappa_1] for n=8 or 10.
        Formula: kappa_j^(n)(m) = a_j^(n) + b_j^(n) / (c_j^(n) - m)^3
        (Eq. 8, p. 188; coefficients from Table 1).
        """
        if n == 6:
            ABC = [(0.00172, 0.02437, 1.64375)]
        elif n == 8:
            ABC = [(0.00440, 0.08821, 2.35938), (0.00615, 0.02472, 3.63594)]
        elif n == 10:
            ABC = [(0.00118, 0.04219, 2.74688), (0.00367, 0.12780, 2.77031)]
        else:
            return np.zeros(0, dtype=np.float64)
        return np.asarray([a + b / ((c - m) ** 3) for a, b, c in ABC], dtype=np.float64)

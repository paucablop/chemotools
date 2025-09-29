# Authors: Nusret Emirhan Salli <nusret.emirhan.salli@gmail.com>
# License: MIT


from __future__ import annotations
from typing import Literal
import numpy as np

from ._base import _BaseFIRFilter


class ModifiedSincFilter(_BaseFIRFilter):
    """
    Modified-sinc (MS) smoother.

    Kernel on normalized x = i/(m+1) (so the first point *outside* the support is x=±1):
        h(x) = A · w(x) · sinc(((n+4)/2) · x) + Σ κ_j^(n) · w(x) · x · sin((2j+ν)πx)
    - w(x) is a Gaussian-based window constructed so that w(0)=1, w(1)=0, and w'(1)=0,
      i.e., amplitude and slope vanish at the ends as described in the paper. :contentReference[oaicite:3]{index=3}
    - κ_j^(n) follow κ = a + b/(c - m)^3 (Table 1) for n ∈ {6,8,10}. :contentReference[oaicite:4]{index=4}
    - Final kernel is symmetrized and normalized (DC = 1).

    Parameters
    ----------
    window_size : int, odd >= 3
    n : int, even >= 4
        Controls number of sinc “wiggles” in support (use 6, 8, or 10 for paper’s κ).
    alpha : float > 0
        Gaussian width parameter for the window construction.
    use_corrections : bool, default=True
        Apply passband-flattening corrections for n ∈ {6,8,10} when valid.
    mode, axis : inherited
    """

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

    def _compute_kernel(self) -> np.ndarray:
        """
        Compute the Modified Sinc kernel based on parameters.

        Returns:
            np.ndarray: Symmetric kernel with sum=1.0 (DC preserving)
        """
        # Parameter validation
        if self.n % 2 != 0 or self.n < 2:
            raise ValueError("n must be an even integer ≥ 2.")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive.")

        # Calculate kernel points and normalize x to [-1, 1] range
        m = (self.window_size - 1) // 2
        i = np.arange(-m, m + 1, dtype=np.float64)
        x = i / (m + 1) if m >= 0 else np.array([0.0])

        # Core sinc function (with zeros at specific points for even n)
        core = np.sinc(0.5 * (self.n + 4) * x)  # np.sinc(u) := sin(pi*u)/(pi*u)

        # Create window function with properties: w(0)=1, w(1)=0, w'(1)=0
        # Window form: w(x) = A*exp(-α x^2) + B*(exp(-α(x-2)^2)+exp(-α(x+2)^2)) + C
        E1 = np.exp(-self.alpha * 1.0)  # e^{-α}
        Ep = np.exp(-self.alpha * 1.0)  # (x-2)^2 at x=1
        Em = np.exp(-self.alpha * 9.0)  # (x+2)^2 at x=1
        e4 = np.exp(-self.alpha * 4.0)  # (±2)^2 at x=0

        M = np.array(
            [
                [1.0, 2.0 * e4, 1.0],  # w(0) = 1
                [E1, (Ep + Em), 1.0],  # w(1) = 0
                [
                    -2 * self.alpha * E1,
                    2 * self.alpha * (Ep - 3 * Em),
                    0.0,
                ],  # w'(1) = 0
            ],
            dtype=np.float64,
        )

        rhs = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        # Solve for window coefficients
        Acoef, Bcoef, Ccoef = np.linalg.solve(M, rhs)

        # Apply window function to all points
        window = (
            Acoef * np.exp(-self.alpha * x**2)
            + Bcoef
            * (
                np.exp(-self.alpha * (x - 2.0) ** 2)
                + np.exp(-self.alpha * (x + 2.0) ** 2)
            )
            + Ccoef
        )

        # Initial kernel: sinc core * window
        h = core * window

        # Apply optional passband-flattening corrections from paper
        if (
            self.use_corrections
            and self._has_kappa_table(self.n)
            and (m >= self.n // 2 + 2)
        ):
            # ν = 1 for n=6,10; ν = 2 for n=8
            nu = 1 if ((self.n // 2) % 2 == 1) else 2

            # Get correction coefficients from paper's table
            coeffs = self._kappa_coeffs(self.n, m)

            # Apply correction terms
            B = []
            for j, kappa in enumerate(coeffs):
                bj = window * x * np.sin((2 * j + nu) * np.pi * x)
                B.append(kappa * bj)
            if B:
                h = h + np.sum(np.stack(B, axis=0), axis=0)

        # Ensure perfect symmetry and normalize to sum=1
        h = 0.5 * (h + h[::-1])
        s = h.sum()

        if not np.isfinite(s) or abs(s) < 1e-15:
            raise FloatingPointError(
                "Kernel normalization failed; try different parameters."
            )

        # Return DC-preserving kernel (sum = 1.0)
        h = h / s
        return h

    # ====== κ(a,b,c) table per paper’s Table 1 (eq. 8) ======
    @staticmethod
    def _has_kappa_table(n: int) -> bool:
        return n in (6, 8, 10)

    @staticmethod
    def _kappa_coeffs(n: int, m: int) -> np.ndarray:
        """
        Returns [κ_0] for n=6; [κ_0, κ_1] for n=8 or 10; using κ = a + b/(c - m)^3.
        Coefficients (a,b,c) taken from Table 1 of the paper.
        """
        # (a, b, c) tuples in the order of j
        if n == 6:
            ABC = [(0.00172, 0.02437, 1.64375)]
        elif n == 8:
            ABC = [(0.00440, 0.08821, 2.35938), (0.00615, 0.02472, 3.63594)]
        elif n == 10:
            ABC = [(0.00118, 0.04219, 2.74688), (0.00367, 0.12780, 2.77031)]
        else:
            return np.zeros(0, dtype=np.float64)

        ks = []
        for a, b, c in ABC:
            ks.append(a + b / ((c - m) ** 3))
        return np.asarray(ks, dtype=np.float64)

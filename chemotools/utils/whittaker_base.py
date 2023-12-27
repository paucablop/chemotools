import numpy as np
from scipy.linalg import cho_solve_banded, cholesky_banded
from scipy.sparse import csr_matrix, dia_matrix

from chemotools.utils.banded_linalg import conv_to_lu_banded_storage
from chemotools.utils.finite_differences import (
    forward_finite_diff_conv_matrix,
    posdef_mod_squared_fw_fin_diff_conv_matrix,
)
from chemotools.utils.models import _PENTAPY_AVAILABLE, BandedSolveDecompositions

if _PENTAPY_AVAILABLE:
    import pentapy as pp
# else nothing

_CholeskyDecomposition = tuple[np.ndarray, bool]
_PentapyDecomposition = None
_Decomposition = _PentapyDecomposition | _CholeskyDecomposition


class WhittakerLikeSolver:
    """This class can be used to solve linear systems of equations that involve banded
    matrices as they occur in applications like the Whittaker-Henderson-smoothing or
    derived methods like Asymmetric Least Squares (ALS) baseline correction.
    It support weights and tries to use the most efficient method available.

    Attributes
    ----------
    series_size_ : int
        The size of the series to smooth. It is equivalent to `n_features_in_`, but it
        was renamed to be allow for definition after the initialisation.
    lam_ : int or float
        The lambda parameter to use for the Whittaker smooth.
    differences_ : int
        The number of differences to use for the Whittaker smooth. If the aim is to
        obtain a smooth estimate of the ``m``-th order derivative, this should be set to
        at least ``m + 2``.
    l_and_u_ : tuple[int, int]
        The number of sub- (first) and superdiagonals (second element). Both will equal
        ``differences_``.
    fw_fin_diff_mat_ : dia_matrix
        The finite difference matrix, which serves as a precursor for the penalty matrix
        in its sparse representation as DIA-matrix.
    base_squ_fw_fin_diff_mat_ : csr_matrix
        The squared finite difference matrix, which serves as the penalty matrix in its
        sparse representation as CSR-matrix. It is already made positive definite by
        adding a multiple of the identity matrix to the main diagonal, but otherwise it
        is its original form. It can be used directly for baseline correction algorithms
        that do not require sophisticated modifications of the penalty matrix.
    base_squ_fw_fin_diff_mat_lub_ : np.ndarray
        The banded storage version of ``base_squ_fw_fin_diff_mat_`` for LAPACK LU
        decomposition. It is stored this way because it is efficient for Pentapy while
        being slightly inefficient for the Cholesky decomposition. Since the conversion
        for the latter only required row access in a C-order array, this should not be
        a major time sink though.
    __dtype : type, default=np.float64
        The data type to which the series to be smoothed will be converted to. To avoid
        numerical issues, all series are converted to double precision.
    __machine_min_tol_mult : int, default=10
        The multiple of the machine epsilon that is used to make the penalty matrix
        positive definite. It is only relevant if it exceeds ``series_size``.
        Positive definiteness is assured by lifting the main diagonal by a small
        multiple of the identity matrix whose scale depends on the machine precision
        as well as the maximum eigenvalue of the squared forward finite difference
        matrix. Please refer to the documentation of
        ``posdef_mod_squared_fw_fin_diff_conv_matrix`` for more details.
    __allow_pentapy : bool, default=True
        Whether to enable the Pentapy solver if available. This is only used for
        debugging and testing purposes.

    """  # noqa: E501

    __dtype: type = np.float64
    __machine_min_tol_mult: int = 10
    __allow_pentapy: bool = True

    def __init__(
        self,
    ) -> None:
        pass

    def _setup_for_fit(
        self,
        series_size: int,
        lam: int | float,
        differences: int,
    ) -> None:
        """Pre-computes everything that can be computed for the smoothing in general as
        well as for fitting the lambda parameter itself.
        """

        # the input arguments are stored
        self.series_size_: int = series_size
        self.lam_: int | float = lam
        self.differences_: int = differences
        self.max_eigval_mult_: float = (  # type: ignore
            np.finfo(self.__dtype).eps * max(self.__machine_min_tol_mult, series_size)
        )

        # the forward finite difference matrix is computed ...
        self.l_and_u_: tuple[int, int] = (self.differences_, self.differences_)
        self.fw_fin_diff_mat_: dia_matrix = forward_finite_diff_conv_matrix(
            differences=self.differences_, series_size=self.series_size_
        )
        # ... followed by the squared forward finite difference matrix
        self.base_squ_fw_fin_diff_mat_: csr_matrix = (
            posdef_mod_squared_fw_fin_diff_conv_matrix(
                fw_fin_diff_mat=self.fw_fin_diff_mat_,
                differences=self.differences_,
                dia_mod_matrix=None,
                max_eigval_mult=self.max_eigval_mult_,
                dtype=self.__dtype,
            )
        )
        self.base_squ_fw_fin_diff_mat_lub_: np.ndarray = conv_to_lu_banded_storage(
            a=self.base_squ_fw_fin_diff_mat_,
            l_and_u=self.l_and_u_,
        )

        # finally, Pentapy is enabled if available, the number of differences is 2,
        # and the lambda parameter is not fitted automatically
        self._pentapy_enabled: bool = (
            _PENTAPY_AVAILABLE and self.differences_ == 2 and self.__allow_pentapy
        )

    def _pentapy_solve(self, ab: np.ndarray, bw: np.ndarray) -> np.ndarray:
        """Solves the linear system of equations ``(W + lam * D^T @ D) @ x = W @ b``
        with the Pentapy package. This is written as the system ``A @ x = b`` where
        ``A = W + lam * D^T @ D`` and ``b = W @ b``.

        Notes
        -----
        Pentapy does not (maybe yet) allow for 2D right-hand side matrices, so the
        solution is computed for each column of ``bw`` separately.

        """

        # for 1-dimensional right-hand side vectors, the solution is computed directly
        if bw.ndim == 1:
            return pp.solve(
                mat=ab,
                rhs=bw,
                is_flat=True,
                index_row_wise=False,
                solver=1,
            )

        # for 2-dimensional right-hand side matrices, the solution is computed for each
        # column separately
        else:
            solution = np.empty(shape=(bw.shape[1], bw.shape[0]))
            for iter_j in range(0, bw.shape[1]):
                solution[iter_j, ::] = pp.solve(
                    mat=ab,
                    rhs=bw[::, iter_j],
                    is_flat=True,
                    index_row_wise=False,
                    solver=1,
                )

            return solution.transpose()

    def _cholesky_solve(
        self, ab: np.ndarray, bw: np.ndarray
    ) -> tuple[np.ndarray, tuple[np.ndarray, bool]]:
        """Solves the linear system of equations ``(W + lam * D^T @ D) @ x = W @ b``
        with the Cholesky decomposition. This is written as the system ``A @ x = b``
        where ``A = W + lam * D^T @ D`` and ``b = W @ b``.

        Even though it is mathematically guaranteed that ``A`` is positive definite,
        numerical errors can lead to a non-positive definite matrix. In this case, the
        Cholesky decomposition fails and a ``LinAlgError`` is raised.

        """

        lower = True
        cb = cholesky_banded(ab, lower=lower, check_finite=False)
        decomposition = (cb, lower)
        return (
            cho_solve_banded(cb_and_lower=decomposition, b=bw, check_finite=False),
            decomposition,
        )

    def _solve(
        self,
        bw: np.ndarray,
        log_lam: float,
        w: np.ndarray | None,
        mod_squ_fin_diff_mat_lub: np.ndarray,
    ) -> tuple[np.ndarray, _Decomposition, BandedSolveDecompositions]:
        """Solves the linear system of equations ``(W + lam * D^T @ D) @ x = W @ b``
        where ``W`` is a diagonal matrix with the weights ``w`` on the main diagonal and
        ``D`` is the finite difference matrix of order ``differences``.

        Parameters
        ----------
        bw : np.ndarray of shape (n,) or (n, m)
            The weighted right-hand side vector or matrix of the linear system of
            equations.
        log_lam : float
            The logarithm of the lambda parameter to use for the Whittaker-like smooth.
        w : np.ndarray of shape (n,)
            The weights to use for the linear system of equations. It must be a vector
            even if ``bw`` is a matrix because having ``bw`` as a matrix is only
            possible if lambda is fixed and the same weights are applied to all series.
        mod_squ_fin_diff_mat_lub : np.ndarray of shape (n, n)
            The positive definite (modified) squared forward finite difference matrix
            stored in the banded storage for LAPACK LU decomposition.

        Returns
        -------
        x : np.ndarray of shape (n,)
            The solution vector of the linear system of equations.
        decomposition : tuple
            The decomposition used to solve the linear system of equations.
            For the Cholesky decomposition, this is a tuple ``(cb, lower)`` where ``cb``
            is the banded storage of the Cholesky decomposition and ``lower`` is a
            boolean flag indicating whether the lower or upper triangular matrix is
            stored.
            For the Pentapy solver this is ``None``.
        decomposition_type : BandedSolveDecompositions
            The type of decomposition used to solve the linear system of equations.

        """

        # the banded storage for a LAPACK LU decomposition is computed by updating the
        # diagonal of the squared forward finite difference matrix D^T @ D with the
        # weights
        ab = np.exp(log_lam) * mod_squ_fin_diff_mat_lub
        if w is not None:
            ab[self.differences_, ::] += w
        else:
            ab[self.differences_, ::] += 1.0

        # the linear system of equations is solved with the most efficient method with
        # Cholesky decomposition as fallback
        # Case 1: Pentapy can be used
        if self._pentapy_enabled:
            x = self._pentapy_solve(ab=ab, bw=bw)
            if np.all(np.isfinite(x)):
                return (
                    x,
                    None,
                    BandedSolveDecompositions.PENTAPY,
                )

            else:
                # if Pentapy fails, the Cholesky decomposition is used as fallback
                x, decomposition = self._cholesky_solve(
                    ab=ab[self.differences_ : :, ::], bw=bw
                )
                return x, decomposition, BandedSolveDecompositions.CHOLESKY

        # Case 2: Pentapy cannot be used, but the matrix is NUMERICALLY positive
        # definite
        else:
            x, decomposition = self._cholesky_solve(
                ab=ab[self.differences_ : :, ::], bw=bw
            )
            return x, decomposition, BandedSolveDecompositions.CHOLESKY

    def _solve_single_x_fixed_lam(
        self,
        x_weighted: np.ndarray,
        w: np.ndarray | None,
        mod_squ_fin_diff_mat_lub: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Fits the Whittaker-like smooth with a fixed lambda parameter.

        For the parameters, please refer to the documentation of ``_solve``. Instead of
        a 2D-Array, a 1D-Array is expected for ``x`` and ``w``. Besides, it expects
        the product ``x * w`` to be passed as ``x_weighted`` since this is more
        efficient than computing it inside the solver.

        """

        # the solution of the linear system of equations is computed
        x_smooth, _, _ = self._solve(
            bw=x_weighted,
            log_lam=np.log(self.lam_),  # type: ignore
            w=w,
            mod_squ_fin_diff_mat_lub=mod_squ_fin_diff_mat_lub,
        )

        # finally, the solution is returned together with the lambda parameter
        return x_smooth, self.lam_  # type: ignore

    def _solve_single_x(
        self,
        x: np.ndarray,
        w: np.ndarray | None,
        mod_squ_fin_diff_mat_lub: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Fits the Whittaker-like smooth to a single series for a fixed or fitted
        lambda parameter.

        For the parameters, please refer to the documentation of ``solve``. Instead of
        2D-Arrays, 1D-Arrays are expected for ``x`` and ``w``.

        """

        # first, the weights need to be ensured to be invertible by using the relative
        # condition number and then the weighted series is computed
        # NOTE: this numerical trick ensures that the smoothing also works in the
        #       limiting case that ``lam`` is vanishing. Since the diagonal matrix W has
        #       eigenvalues that correspond to the main diagonal entries, this problem
        #       is readily solved by bounding the minimum weight to ``rcond * w.max()``
        #       which works since a maximum weight of zero has already been excluded
        if w is not None:
            w_lifted = np.maximum(w, self.max_eigval_mult_ * w.max())
            x_wavg = np.average(x, weights=w_lifted)
            x_weighted = w_lifted * (x - x_wavg)
        else:
            w_lifted = None
            x_wavg = np.average(x)
            x_weighted = x - x_wavg

        # then, the solution of the linear system of equations is computed
        # NOTE: this is a placeholder where an if-else-statement needs to be inserted
        #       for then the lambda parameter needs to be evaluated automatically
        x_smooth, lam = self._solve_single_x_fixed_lam(
            x_weighted=x_weighted,
            w=w_lifted,
            mod_squ_fin_diff_mat_lub=mod_squ_fin_diff_mat_lub,
        )
        return x_smooth + x_wavg, lam

    def _solve_multiple_x(
        self,
        X: np.ndarray,
        w: np.ndarray | None,
        mod_squ_fin_diff_mat_lub: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fits the Whittaker-like smooth to multiple series when the lambda parameter
        is fixed and the same weights are applied to all series.
        It leverages the ability of LAPACK (not pentapy) to solve multiple linear
        systems of equations at once from the same inversion.

        For the parameters, please refer to the documentation of ``_solve``.

        """

        # in this special case, the solution of the linear system of equations can be
        # computed with a single matrix inversion
        # first, the weights need to be ensured to be invertible by using the relative
        # condition number and then the weighted series is computed
        # NOTE: this numerical trick ensures that the smoothing also works in the
        #       limiting case that ``lam`` is vanishing. Since the diagonal matrix W has
        #       eigenvalues that correspond to the main diagonal entries, this problem
        #       is readily solved by bounding the minimum weight to ``rcond * w.max()``
        #       which works since a maximum weight of zero has already been excluded
        if w is not None:
            w_lifted = np.maximum(w, self.max_eigval_mult_ * w.max()).ravel()
            x_wavg = np.average(X, weights=w_lifted, axis=1)
            x_weighted = np.transpose(
                w_lifted[np.newaxis, ::] * (X - x_wavg[::, np.newaxis])
            )
        else:
            x_wavg = np.average(X, axis=1)
            x_weighted = np.transpose(X) - x_wavg[np.newaxis, ::]
            w_lifted = None

        # then, the solution of the linear system of equations is computed
        X_smooth, _, _ = self._solve(
            bw=x_weighted,
            log_lam=np.log(self.lam_),  # type: ignore
            w=w_lifted,
            mod_squ_fin_diff_mat_lub=mod_squ_fin_diff_mat_lub,
        )

        return (
            np.transpose(X_smooth + x_wavg[np.newaxis, ::]),
            np.full(shape=(X.shape[0],), fill_value=self.lam_),  # type: ignore
        )

    def _whittaker_solve(
        self,
        X: np.ndarray,
        *,
        w: np.ndarray | None = None,
        use_same_w_for_all: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solves the linear equations for Whittaker-Henderson smoothing for Arrays that
        are stored in 2D format, i.e., each series is stored as a row.
        Internally it chooses the most appropriate method and solver depending on the
        data dimensionality, the weights, and the system's available packages (pentapy).

        Parameters
        ----------
        X : np.ndarray of shape (n, m)
            The series to be smoothed stored as individual rows.
        w : np.ndarray of shape(1, m), shape(n, m), or None
            The weights to be applied for smoothing. If only a single row is provided
            and ``use_same_w_for_all``, the same weights can be applied for all series
            in `X`, which enhances the smoothing a lot for fixed smoothing parameters
            `lam`.
            If ``None``, no weights are applied and each datapoint is assumed to have
            equal importance. This allows for ``use_same_w_for_all`` to be ``True``
            as well.
        use_same_w_for_all
            Whether to use the same weights for all series in `X`. This is only possible
            if `w` is a single row or ``None``.

        Returns
        -------
        X_smooth : np.ndarray of shape(n, m)
            The smoothed series stored as individual rows.
        lam : np.ndarray of shape(n,)
            The lambda parameter used for the smoothing of each series. If `lam` was
            fixed, this is a vector of length `n` with the same value for each series.

        """  # noqa: E501

        # a nested function is defined for updating the weights
        # TODO: add zero-weight protection (eigenvalues are weights themselves)
        def update_to_next_weights(iter_i: int) -> None:
            nonlocal w_curr
            if iter_i > 0:
                if w is None:
                    w_curr = None
                    return
                elif not use_same_w_for_all:
                    w_curr = w[iter_i, ::].copy()
                else:
                    return

            else:
                if w is None:
                    w_curr = None
                else:
                    w_curr = w[iter_i, ::].copy()

        assert (
            X.dtype == self.__dtype
        ), f"Internal error: Promotion to {self.__dtype} failed."

        # if multiple x with the same weights are to be solved for fixed lambda, this
        # can be done more efficiently by leveraging Pentapy's and LAPACK'S ability to
        # perform multiple solves from the same inversion at once
        if use_same_w_for_all:
            return self._solve_multiple_x(
                X=X, w=w, mod_squ_fin_diff_mat_lub=self.base_squ_fw_fin_diff_mat_lub_
            )
        # else nothing

        # otherwise, the solution of the linear system of equations is computed for
        # each series
        X_smooth = np.empty_like(X)
        lam = np.empty(shape=(X.shape[0],))
        w_curr = None
        for iter_i, x in enumerate(X):
            update_to_next_weights(iter_i=iter_i)
            X_smooth[iter_i], lam[iter_i] = self._solve_single_x(
                x=x,
                w=w_curr,
                mod_squ_fin_diff_mat_lub=self.base_squ_fw_fin_diff_mat_lub_,
            )

        return X_smooth, lam

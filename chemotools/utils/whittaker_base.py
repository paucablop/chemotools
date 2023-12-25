import numpy as np
from scipy.linalg import cho_solve_banded, cholesky_banded
from scipy.optimize import minimize_scalar

from chemotools.utils.banded_linalg import (
    conv_symm_sparse_banded_sposdef_to_posdef,
    conv_to_lu_banded_storage,
    lu_banded,
    lu_solve_banded,
    slodget_cho_banded,
)
from chemotools.utils.finite_differences import forward_finite_diff_conv_matrix
from chemotools.utils.models import _PENTAPY_AVAILABLE, BandedSolveDecompositions

if _PENTAPY_AVAILABLE:
    import pentapy as pp
# else nothing

_LUDecomposition = tuple[np.ndarray, np.ndarray, tuple[int, int]]
_CholeskyDecomposition = tuple[np.ndarray, bool]
_PentapyDecomposition = None
_Decomposition = _PentapyDecomposition | _CholeskyDecomposition | _LUDecomposition

LN_OF_TWO_PI = np.log(2.0 * np.pi)


class WhittakerLikeSolver:
    """This class can be used to solve linear systems of equations that involve banded
    matrices as they occur in applications like the Whittaker-Henderson-smoothing or
    derived methods like Asymmetric Least Squares (ALS) baseline correction.
    It support weights and tries to use the most efficient method available.
    Besides, it also offers the possibility to fit the roughness penalty itself.

    Attributes
    ----------
    _lam : int or float or None, default=1e2
        The lambda parameter to use for the Whittaker smooth.
        If ``None``, the transformer will fit the smoothness parameter itself by
        maximising the marginal likelihood, which can be computationally expensive, but
        more accurate than using (Generalized) Cross-Validation (see Notes).

    _differences : int, default=1
        The number of differences to use for the Whittaker smooth. If the aim is to
        obtain a smooth estimate of the `m`-th order derivative, this should be set to
        at least ``m + 2``.

    _rcond : float, default=1e-15
        The relative condition number which is used to keep all matrices involved
        positive definite. This is only used if ``lam`` is ``None``.
        It works in the same way as the ``rcond`` parameter of SciPy's ``linalg.pinvh``.

    _allow_pentapy : bool, default=True
        Whether to enable the Pentapy solver if available. This is only used for
        debugging and testing purposes.

    Notes
    -----
    If ``lam`` is ``None``, the pentapy solver cannot be used even if available.
    Besides, the computational load increases since - especially for large series and
    high differences - the pseudo-determinant of the penalty matrix ``P = D^T @ D``
    needs to be computed, which is computationally expensive and also subject to
    numerical inaccuracies. The latter cause some eigenvalues to be numerically
    negative, even though their true value is positive and of order ``<< 1e-16``, which
    makes their accurate computation numerically impossible with double precision.
    Therefore, the eigenvalues are lifted by adding a small value to the diagonal of
    the penalty matrix before computing the pseudo-determinant to make the smallest
    eigenvalue numerically positive when compared to the largest eigenvalue. From a
    smoothing point of view, this turns the Whittaker-Smoothing with derivative penalty
    into a blend of Whittaker Smoothing and Tikhonov Regularisation.
    So, in contrast to ``P = D^T @ D``, the penalty matrix ``P = D^T @ D + c * I`` is
    used where ``c`` is a very small numerical value, so in first approximation, the
    combined smoother is still mostly a Whittaker smoother.

    """

    __log_lam_bounds: tuple[float, float] = (
        -34.5,  # 1e-15
        115.13,  # 1e50
    )
    __allow_pentapy: bool = True

    def __init__(
        self,
    ) -> None:
        pass

    def _setup_for_fit(
        self,
        series_size: int,
        lam: int | float | None,
        differences: int,
        rcond: float = 1e-15,
    ) -> None:
        """Pre-computes everything that can be computed for the smoothing in general as
        well as for fitting the lambda parameter itself.
        """

        # the input arguments are stored
        self._lam: int | float | None = lam
        self._differences: int = differences
        self._rcond: float = rcond

        # the banded storage for a LAPACK LU decomposition is computed for the squared
        # forward finite difference matrix D^T @ D which is the penalty matrix P
        self.auto_lam_: bool = self._lam is None
        self.l_and_u_: tuple[int, int] = (self._differences, self._differences)
        self.series_size_: int = series_size
        self.squ_fw_fin_diff_mat_ = forward_finite_diff_conv_matrix(
            differences=self._differences,
            accuracy=1,
            series_size=series_size,
        )
        self.squ_fw_fin_diff_mat_ = (
            self.squ_fw_fin_diff_mat_.T @ self.squ_fw_fin_diff_mat_
        )

        # if the lambda parameter is to be fitted automatically, the penalty matrix is
        # converted to a positive definite matrix and its log-determinant is computed
        if self.auto_lam_:
            self.squ_fw_fin_diff_mat_ = conv_symm_sparse_banded_sposdef_to_posdef(
                a=self.squ_fw_fin_diff_mat_, l_and_u=self.l_and_u_, rcond=self._rcond
            )
        # else nothing

        # finally, the matrix is converted to a banded storage
        self.fw_fin_diff_mat_lu_banded_: np.ndarray = conv_to_lu_banded_storage(
            a=self.squ_fw_fin_diff_mat_,
            l_and_u=self.l_and_u_,
        )

        # if the lambda parameter is to be fitted automatically, the log-determinant of
        # the penalty matrix is computed, which reduces to summing up the logarithms of
        # of the squared main diagonal elements of its banded Cholesky decomposition
        if self._lam is None:
            lower = False
            penalty_chol = cholesky_banded(
                ab=self.fw_fin_diff_mat_lu_banded_[0 : self._differences + 1, ::],
                lower=lower,
                check_finite=False,
            )

            self.penalty_log_det_: float
            det_sign, self.penalty_log_det_ = slodget_cho_banded(
                decomposition=(penalty_chol, lower)
            )
            assert det_sign > 0.0, "The penalty matrix is still not positive definite."

        else:
            self.penalty_log_det_: float = float("nan")

        # finally, Pentapy is enabled if available, the number of differences is 2,
        # and the lambda parameter is not fitted automatically
        self._pentapy_enabled: bool = (
            _PENTAPY_AVAILABLE
            and self._differences == 2
            and not self.auto_lam_
            and self.__allow_pentapy
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

        lower = False
        cb = cholesky_banded(ab, lower=lower, check_finite=False)
        decomposition = (cb, lower)
        return (
            cho_solve_banded(cb_and_lower=decomposition, b=bw, check_finite=False),
            decomposition,
        )

    def _lu_solve(
        self, ab: np.ndarray, bw: np.ndarray
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, tuple[int, int]]]:
        """Solves the linear system of equations ``(W + lam * D^T @ D) @ x = W @ b``
        with the LU decomposition. This is written as the system ``A @ x = b`` where
        ``A = W + lam * D^T @ D`` and ``b = W @ b``.

        """

        # the LU decomposition is computed, but if the matrix cannot properly be
        # decomposed and at least one diagonal element of U is zero, a LinAlgError is
        # raised
        try:
            lub, ipiv = lu_banded(
                l_and_u=self.l_and_u_,
                ab=ab,
                check_finite=False,
            )
            decomposition = (lub, ipiv, self.l_and_u_)

        except RuntimeWarning:
            raise np.linalg.LinAlgError()

        # the linear system is solved
        return (
            lu_solve_banded(
                decomposition=decomposition,
                b=bw,
                check_finite=False,
            ),
            decomposition,
        )

    def _solve(
        self,
        bw: np.ndarray,
        log_lam: float,
        w: np.ndarray | None,
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
            even if ``wb`` is a matrix because having ``wb`` as a matrix is only
            possible if lambda is fixed and the same weights are applied to all series.

        Returns
        -------
        x : np.ndarray of shape (n,)
            The solution vector of the linear system of equations.

        decomposition : tuple
            The decomposition used to solve the linear system of equations.
            For the LU decomposition, this is a tuple ``(lub, ipiv, l_and_u)`` where
            ``lub`` is the banded storage of the LU decomposition, ``ipiv`` is the pivot
            vector, and ``l_and_u`` is the tuple ``(l, u)`` with the lower and upper
            bandwidth of ``lub``.
            For the Cholesky decomposition, this is a tuple ``(cb, lower)`` where ``cb``
            is the banded storage of the Cholesky decomposition and ``lower`` is a
            boolean flag indicating whether the lower or upper triangular matrix is
            stored.
            For the Pentapy solver, this is ``None``.

        decomposition_type : BandedSolveDecompositions
            The type of decomposition used to solve the linear system of equations.

        Notes
        -----
        This methods has the following fallback strategy in case of failures (->):

            - with pentapy: Pentapy -> LU -> weighted polynomial fit (``np.polyfit``)
            - without pentapy: Cholesky -> LU -> weightedd polynomial fit
                (``np.polyfit``)

        Why ``np.polyfit``? If the LU-decomposition fails, the lambda parameter is so
        large that the penalty matrix is numerically singular. But on the other hand
        this also means that the ``differences``-th order derivative of the series
        should be as small as possible and the data fidelity term has no influence on
        the solution. Fortunately, the penalty can be reduced to zero by fitting the
        data with a weighted polynomial of order ``differences - 1`` because its
        ``differences``-th order derivative is zero. It is however still closer to the
        data than smoother solutions, i.e., even lower order polynomials whose
        derivatives would also be zero.

        """

        # the banded storage for a LAPACK LU decomposition is computed by updating the
        # diagonal of the squared forward finite difference matrix D^T @ D with the
        # weights
        # NOTE: using the inverse of lambda is more efficient than using lambda directly
        #       since then it needs to be applied to the weights only rather than a
        #       possible large matrix
        ab = np.exp(log_lam) * self.fw_fin_diff_mat_lu_banded_
        if w is not None:
            ab[self._differences, ::] += w
        else:
            ab[self._differences, ::] += 1.0

        # the linear system of equations is solved with the most efficient method with
        # LU decomposition as the fallback
        try:
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
                    raise np.linalg.LinAlgError()

            # Case 2: Pentapy cannot be used, but the matrix is NUMERICALLY positive
            # definite
            else:
                x, decomposition = self._cholesky_solve(
                    ab=ab[0 : self._differences + 1], bw=bw
                )
                return x, decomposition, BandedSolveDecompositions.CHOLESKY

        # Case 3: Pentapy cannot be used and the matrix is NOT NUMERICALLY positive
        # definite, so the fallback is to use the LU decomposition
        except np.linalg.LinAlgError:
            try:
                x, decomposition = self._lu_solve(ab=ab, bw=bw)
                if np.all(np.isfinite(x)):
                    return x, decomposition, BandedSolveDecompositions.LU

                else:
                    raise np.linalg.LinAlgError()

            # Case 4: the LU decomposition also fails, so the fallback is to fit a
            # polynomial
            except np.linalg.LinAlgError:
                idx_vect = np.arange(
                    start=0,
                    stop=self.series_size_,
                    step=1,
                    dtype=np.int64,
                )
                poly = np.poly1d(
                    np.polyfit(x=idx_vect, y=bw, deg=self._differences - 1, w=w)
                )
                return poly(idx_vect), None, BandedSolveDecompositions.POLYFIT

    # FIXME: this method is not yet used and needs to be tested
    def _calc_neg_marginal_likelihood(
        self,
        x_orig: np.ndarray,
        x_smooth: np.ndarray,
        decomposition: _Decomposition,
        solver: BandedSolveDecompositions,
        log_lam: float,
        w: np.ndarray | None,
        w_logdet: float,
        lml_sample_size_corr: float,
    ) -> float:
        """Computes the negative marginal likelihood of the Whittaker-like smooth."""

        # running this method is only possible if the lambda parameter is fitted and
        # the decomposition originates from a Cholesky decomposition
        # TODO: maybe also allow LU decompositions
        assert self.auto_lam_ and solver == BandedSolveDecompositions.CHOLESKY, (
            "The solution of the linear system could not be computed with a Cholesky "
            "decomposition."
        )

        # now, the weighted sum of squared residuals (wRSS) is computed
        if w is not None:
            wrss = np.sum(w * np.square(x_orig - x_smooth))
        else:
            wrss = np.sum(np.square(x_orig - x_smooth))

        # then, the sum of squared penalties (SSP) is computed using the positive
        # definite penalty matrix
        ssp = np.exp(log_lam) * (x_smooth @ self.squ_fw_fin_diff_mat_ @ x_smooth)

        # besides, the log-determinant of the matrix 1/lambda * W + P which is simple
        # because its Cholesky decomposition is already available
        ww_plus_p_det_sign, ww_plus_p_log_det = slodget_cho_banded(
            decomposition=decomposition  # type: ignore
        )
        assert (
            ww_plus_p_det_sign > 0.0
        ), "The matrix to invert was not positive definite."

        # finally, the reduced marginal likelihood is computed
        return 0.5 * (
            wrss
            + ssp
            - w_logdet
            - (self.series_size_ * log_lam + self.penalty_log_det_)
            + ww_plus_p_log_det
            + lml_sample_size_corr
        )

    # FIXME: this method is not yet used and needs to be tested
    def opt_target_auto_lam(
        self,
        log_lam: float,
        x: np.ndarray,
        x_weighted: np.ndarray,
        w: np.ndarray | None,
        w_logdet: float,
        lml_sample_size_corr: float,
    ) -> float:
        """Computes the target function to be minimised when fitting the lambda
        parameter itself.
        """

        # first, the solution of the linear system of equations is computed
        x_smooth, decomposition, solver = self._solve(
            bw=x_weighted, log_lam=log_lam, w=w
        )

        # then, the reduced marginal likelihood is determined and returned
        return self._calc_neg_marginal_likelihood(
            x_orig=x,
            x_smooth=x_smooth,
            decomposition=decomposition,
            solver=solver,
            log_lam=log_lam,
            w=w,
            w_logdet=w_logdet,
            lml_sample_size_corr=lml_sample_size_corr,
        )

    # FIXME: this method is not yet used and needs to be tested
    def _solve_single_x_auto_lam(
        self,
        x: np.ndarray,
        x_weighted: np.ndarray,
        w: np.ndarray | None,
        w_logdet: float,
        num_nonzero_w: int,
    ) -> tuple[np.ndarray, float]:
        """Fits the lambda parameter itself by maximising the reduced marginal
        likelihood. "Reduced" refers to the fact that only the terms that depend on the
        smoothing parameter `lam` are considered.

        For the parameters, please refer to the documentation of ``solve``. Instead of
        a 2D-Array, a 1D-Array is expected for ``x`` and ``w``. Besides, it expects
        the product ``x * w`` to be passed as ``x_weighted`` since this is more
        efficient than computing it inside the solver.

        Notes
        -----
        The logarithm of the marginal likelihood is defined as
        ``-0.5 * (wRSS + SSP - log(pdet(W)) - log(pdet(P)) + log(det(W + P)) +
        (n_obs - diff) * log(2 * pi))`` where

        - `W` as the diagonal matrix of weights
        - `P` is the penalty matrix
        - `wRSS` is the weighted Sum of Squares Residuals between the original and the
            smoothed series `x` and `x_smoothed`
            (``(x - x_smoothed).T @ W @ (x - x_smoothed)``)
        - `SSP` is the sum of squared penalties (``x_smoothed.T @ P @ x_smoothed``)
        - `pdet` is the pseudo-determinant of a matrix (product of its non-zero
            eigenvalues)
        - `det` is the determinant of a matrix (product of its eigenvalues)
        - `n_obs` is the number of observations with non-zero weights
        -"""

        # the sample size correction summand for the marginal likelihood is computed
        lml_sample_size_corr = (num_nonzero_w - self._differences) * LN_OF_TWO_PI
        # the target function is minimised using the bounded Brent method
        opt_res = minimize_scalar(
            fun=self.opt_target_auto_lam,
            bounds=self.__log_lam_bounds,
            method="bounded",
            args=(
                x,
                x_weighted,
                w,
                w_logdet,
                lml_sample_size_corr,
            ),
        )
        assert opt_res.success, "The optimisation did not converge."

        # the solution of the linear system of equations is computed
        x_smooth, _, _ = self._solve(bw=x_weighted, log_lam=opt_res.x, w=w)

        # finally, the solution and the lambda parameter are returned
        return x_smooth, np.exp(opt_res.x)

    def _solve_single_x_fixed_lam(
        self,
        x: np.ndarray,
        x_weighted: np.ndarray,
        w: np.ndarray | None,
    ) -> tuple[np.ndarray, float]:
        """Fits the Whittaker-like smooth with a fixed lambda parameter.

        For the parameters, please refer to the documentation of ``solve``. Instead of
        a 2D-Array, a 1D-Array is expected for ``x`` and ``w``. Besides, it expects
        the product ``x * w`` to be passed as ``x_weighted`` since this is more
        efficient than computing it inside the solver.

        """

        # the solution of the linear system of equations is computed
        x_smooth, _, _ = self._solve(
            bw=x_weighted,
            log_lam=np.log(self._lam),  # type: ignore
            w=w,
        )

        # finally, the solution is returned together with the lambda parameter
        return x_smooth, self._lam  # type: ignore

    def _solve_single_x(
        self,
        x: np.ndarray,
        w: np.ndarray | None,
        w_logdet: float,
        num_nonzero_w: int,
    ) -> tuple[np.ndarray, float]:
        """Fits the Whittaker-like smooth to a single series for a fixed or fitted
        lambda parameter.

        For the parameters, please refer to the documentation of ``solve``. Instead of
        a 2D-Array, a 1D-Array is expected for ``x`` and ``w``.

        """

        # first, the weighted series is computed
        if w is not None:
            x_weighted = w * x
        else:
            x_weighted = x

        # then, the solution of the linear system of equations is computed
        if self.auto_lam_:
            return self._solve_single_x_auto_lam(
                x=x,
                x_weighted=x_weighted,
                w=w,
                w_logdet=w_logdet,
                num_nonzero_w=num_nonzero_w,
            )
        else:
            return self._solve_single_x_fixed_lam(x=x, x_weighted=x_weighted, w=w)

    def _solve_multiple_x(
        self,
        X: np.ndarray,
        w: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fits the Whittaker-like smooth to multiple series when the lambda parameter
        is fixed and the same weights are applied to all series.
        It leverages the ability of Pentapy and LAPACK to solve multiple linear systems
        of equations at once from the same inversion.

        For the parameters, please refer to the documentation of ``solve``.

        """

        # in this special case, the solution of the linear system of equations can be
        # computed with a single matrix inversion
        if w is not None:
            x_weighted = np.transpose(w * X)
            w_inter = w.ravel()
        else:
            x_weighted = np.transpose(X)
            w_inter = w

        # then, the solution of the linear system of equations is computed
        X_smooth, _, _ = self._solve(
            bw=x_weighted,
            log_lam=np.log(self._lam),  # type: ignore
            w=w_inter,
        )

        return (
            np.transpose(X_smooth),
            np.full(shape=(X.shape[0],), fill_value=self._lam),  # type: ignore
        )

    def _whittaker_solve(
        self,
        X: np.ndarray,
        *,
        w: np.ndarray | None = None,
        use_same_w_for_all: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solves the linear equations for Whittaker-Henderson smoothing. Internally it
        chooses the most appropriate method and solver depending on the data
        dimensionality, the weights, and the system's available packages (pentapy).

        Parameters
        ----------
        X : np.ndarray of shape(n, m)
            The series to be smoothed stored as individual rows.
        w : np.ndarray of shape(1, m), shape(n, m), or None
            The weights to be applied for smoothing. If only a single row is provided
            and ``use_same_w_for_all``, the same weights can be applied for all series
            in `X`, which enhances the smoothing a lot for fixed smoothing parameters
            `lam`.
            If ``None``, no weights are applied and each datapoint is assumed to have
            equal importance, This allows for ``use_same_w_for_all`` to be ``True``
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

        """

        # a nested function is defined for updating the weights
        # TODO: add zero-weight protection (eigenvalues are weights themselves)
        def update_to_next_weights(iter_i: int) -> None:
            nonlocal w_curr, w_logdet_curr, num_nonzero_w_curr
            if iter_i > 0:
                if w is None:
                    w_curr = None
                    w_logdet_curr = 0.0  # for identity matrix
                    num_nonzero_w_curr = self.series_size_
                    return
                elif not use_same_w_for_all:
                    w_curr = w[iter_i, ::]
                    non_zero_idxs = (
                        w_curr
                        >= np.finfo(w_curr.dtype).eps * w_curr.max() * w_curr.size
                    )
                    w_logdet_curr = np.sum(np.log(w_curr[non_zero_idxs]))
                    num_nonzero_w_curr = np.sum(non_zero_idxs)
                else:
                    return

            else:
                if w is None:
                    w_curr = None
                    w_logdet_curr = 1.0
                    num_nonzero_w_curr = self.series_size_
                else:
                    w_curr = w[iter_i, ::]
                    non_zero_idxs = (
                        w_curr
                        >= np.finfo(w_curr.dtype).eps * w_curr.max() * w_curr.size
                    )
                    w_logdet_curr = np.sum(np.log(w_curr[non_zero_idxs]))

        # if multiple x with the same weights are to be solved for fixed lambda, this
        # can be done more efficiently by leveraging Pentapy's and LAPACK'S ability to
        # perform multiple solves from the same inversion at once
        if not self.auto_lam_ and use_same_w_for_all:
            return self._solve_multiple_x(X=X, w=w)
        # else nothing

        # otherwise, the solution of the linear system of equations is computed for
        # each series
        X_smooth = np.empty_like(X)
        lam = np.empty(shape=(X.shape[0],))
        w_curr = None
        w_logdet_curr = float("nan")
        num_nonzero_w_curr = -1
        for iter_i, x in enumerate(X):
            update_to_next_weights(iter_i=iter_i)
            X_smooth[iter_i], lam[iter_i] = self._solve_single_x(
                x=x,
                w=w_curr,
                w_logdet=w_logdet_curr,
                num_nonzero_w=num_nonzero_w_curr,
            )

        return X_smooth, lam

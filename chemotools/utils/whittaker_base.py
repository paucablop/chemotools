"""
This utility submodule provides the base class for the Whittaker-like smoothing
algorithm. It is used to solve linear systems of equations that involve banded
matrices as they occur in applications like the Whittaker-Henderson-smoothing or
derived methods like Asymmetric Least Squares (ALS) baseline correction.

"""

### Imports ###

from math import ceil, exp
from typing import Generator, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize_scalar

from chemotools.utils.banded_linalg import (
    LAndUBandCounts,
    conv_upper_chol_banded_to_lu_banded_storage,
    lu_banded,
    lu_solve_banded,
    slogdet_lu_banded,
)
from chemotools.utils.finite_differences import (
    calc_forward_diff_kernel,
    gen_squ_fw_fin_diff_mat_cho_banded,
)
from chemotools.utils.models import (
    _PENTAPY_AVAILABLE,
    BandedLUFactorization,
    BandedPentapyFactorization,
    BandedSolvers,
    WhittakerSmoothLambda,
    WhittakerSmoothMethods,
)

if _PENTAPY_AVAILABLE:
    import pentapy as pp

### Type Aliases ###

_Factorization = Union[BandedLUFactorization, BandedPentapyFactorization]
_FactorizationForLogMarginalLikelihood = BandedLUFactorization
_WhittakerSmoothLambdaPlain = Tuple[
    Union[int, float], Union[int, float], WhittakerSmoothMethods
]


### Class Implementation ###


class WhittakerLikeSolver:
    """
    This class can be used to solve linear systems of equations that involve banded
    matrices as they occur in applications like the Whittaker-Henderson-smoothing or
    derived methods like Asymmetric Least Squares (ALS) baseline correction.
    It support weights and tries to use the most efficient method available.

    Attributes
    ----------
    n_data_ : int
        The number of data points within the series to smooth. It is equivalent to
        ``n_features_in_``, but it was renamed to be allow for definition after the
        initialisation.
    differences_ : int
        The number of differences to use for the smoothing. If the aim is to obtain a
        smooth estimate of the ``m``-th order derivative, this should be set to
        at least ``m + 2``.
        For higher orders, the systems to solve tend to get numerically instable,
        especially when ``n_data_`` grows large and high values for ``lam_`` are used.
        Values below 1 are not allowed.
    _lam_inter_ : WhittakerSmoothLambda
        The internal representation of the lambda parameter to use for the smoothing,
        a.k.a. the penalty weight or smoothing parameter.
        It is internally stored as an instance of the dataclass :class:`WhittakerSmoothLambda`.
    _l_and_u_ : (int, int)
        The number of sub- (first) and superdiagonals (second element) of the final
        matrix to solve for smoothing. Both elements will equal ``differences_``.
    _diff_kernel_flipped_ : ndarray of shape (0, ) or (differences + 1,)
        The flipped kernel to use for the forward finite differences. It is only
        required for the automatic fitting of the lambda parameter by maximizing the log
        marginal likelihood, i.e., when ``lam_ == WhittakerSmoothMethods.LOG_MARGINAL_LIKELIHOOD``.
        Flipping is required due to NumPy's definition of convolution.
    _penalty_matb_ : ndarray of shape (n_data - differences + 1, n_data - differences + 1)
        The squared forward finite differences matrix ``D.T @ D`` stored in the banded
        storage format used for LAPACK's banded LU decomposition.
    _penalty_mat_log_pseudo_det_ : float
        The natural logarithm of the pseudo-determinant of the squared forward finite
        differences matrix ``D.T @ D`` which is used for the automatic fitting of the
        lambda parameter by maximizing the log marginal likelihood, i.e., when
        ``lam_ == WhittakerSmoothMethods.LOG_MARGINAL_LIKELIHOOD``.
        If ``lam_`` is fixed, this is a NaN-value.
    _pentapy_enabled_ : bool
        Whether the Pentapy solver is enabled for the smoothing (``True``) or not
        (``False``).
        It can only be used if the number of differences is 2 and the lambda parameter
        is fixed (and of course if ``pentapy`` is available).
    __dtype : type, default=np.float64
        The data type to which the series to be smoothed will be converted to. To avoid
        numerical issues, all series are converted to double precision.
    __allow_pentapy : bool, default=True
        Whether to enable the Pentapy solver if available. This is only used for
        debugging and testing purposes.
    __zero_weight_tol : float, default=1e-10
        If any of the weights drops below ``weights.max() * __zero_weight_tol``, the
        weight is considered zero for the evaluation of the log marginal likelihood.

    """  # noqa: E501

    __LN_TWO_PI: float = 1.8378770664093453
    __LN_TEN: float = 2.302585092994046
    __dtype: type = np.float64
    __allow_pentapy: bool = True
    __zero_weight_tol: float = 1e-10

    def __init__(
        self,
    ) -> None:
        pass

    ### Initialization and Setup Methods ###

    def _calc_penalty_log_pseudo_det(self) -> float:
        """
        Computes the natural logarithm of the pseudo-determinant of the squared forward
        finite differences matrix ``D.T @ D`` which is necessary for the calculation of
        the log marginal likelihood for the automatic fitting of the penalty weight.

        Returns
        -------
        log_pseudo_det : float
            The natural logarithm of the pseudo-determinant of the penalty matrix.

        Raises
        ------
        RuntimeError
            If the pseudo-determinant of the penalty matrix is negative, thereby
            indicating that the system is extremely ill-conditioned and the automatic
            fitting of the penalty weight is not possible.

        Notes
        -----
        Basically, this could be solved by evaluation of the eigenvalues of ``D.T @ D``
        with a banded eigensolver, but this is computationally expensive and not
        necessary.
        The pseudo-determinant of ``D.T @ D`` is the determinant of ``D @ D.T`` because
        ``D.T @ D`` is rank-deficient with ``differences`` zero eigenvalues while
        ``D @ D.T`` has full rank.
        Since both matrices share the same non-zero eigenvalues, the pseudo-determinant
        is easily computed as the determinant of ``D @ D.T`` via a partially pivoted
        LU decomposition.

        Throughout this method, the matrix ``D.T @ D`` is referred to as the "flipped
        penalty matrix" even though it is not actually flipped.

        """

        # the flipped penalty matrix D @ D.T is computed
        # NOTE: the matrix is returned with integer entries because integer computations
        #       can be carried out at maximum precision; this has to be converted to
        #       double precision for the LU decomposition
        flipped_penalty_matb = gen_squ_fw_fin_diff_mat_cho_banded(
            n_data=self.n_data_,
            differences=self.differences_,
            orig_first=True,
        ).astype(self.__dtype)

        # the pseudo-determinant is computed from the partially pivoted LU decomposition
        # of the flipped penalty matrix
        flipped_l_and_u, flipped_penalty_matb = (
            conv_upper_chol_banded_to_lu_banded_storage(ab=flipped_penalty_matb)
        )
        log_pseudo_det_sign, log_pseudo_det = slogdet_lu_banded(
            lub_factorization=lu_banded(
                l_and_u=flipped_l_and_u,
                ab=flipped_penalty_matb,
                check_finite=False,
            ),
        )

        # if the sign of the pseudo-determinant is positive, the log pseudo-determinant
        # is returned
        if log_pseudo_det_sign > 0.0:
            return log_pseudo_det

        # otherwise, if is negative, the penalty matrix is extremely ill-conditioned and
        # the automatic fitting of the penalty weight is not possible
        raise RuntimeError(
            f"\nThe pseudo-determinant of the penalty D.T @ D matrix is negative, "
            f"indicating that the system is extremely ill-conditioned.\n"
            f"Automatic fitting for {self.n_data_} data points and difference order "
            f"{self.differences_} is not possible.\n"
            f"Please consider reducing the number of data points to smooth by, e.g., "
            f"binning or lowering the difference order."
        )

    def _setup_for_fit(
        self,
        n_data: int,
        differences: int,
        lam: Union[int, float, _WhittakerSmoothLambdaPlain, WhittakerSmoothLambda],
    ) -> None:
        """
        Pre-computes everything that can be computed for the smoothing in general as
        well as for fitting the lambda parameter itself.

        For the parameters, please refer to the documentation of the class.

        """

        # the input arguments are stored and validated
        self.n_data_: int = n_data
        self.differences_: int = differences

        self._lam_inter_: WhittakerSmoothLambda
        if isinstance(lam, (int, float)):
            self._lam_inter_ = WhittakerSmoothLambda(
                bounds=lam,
                method=WhittakerSmoothMethods.FIXED,
            )
        elif isinstance(lam, WhittakerSmoothLambda):
            self._lam_inter_ = lam
        elif isinstance(lam, tuple):
            if len(lam) != 3:
                raise ValueError(
                    f"\nThe lambda parameter must be a tuple of three elements (lower "
                    f"bound, upper bound, method), but it has {len(lam)} elements "
                    f"instead."
                )

            self._lam_inter_ = WhittakerSmoothLambda(
                bounds=(lam[0], lam[1]),
                method=lam[2],
            )
        else:
            raise TypeError(
                f"\nThe lambda parameter must be an integer, a float, a tuple of "
                f"(lower bound, upper bound, method), or an instance of "
                f"WhittakerSmoothLambda, but it is {type(lam)} instead."
            )

        # the squared forward finite difference matrix D.T @ D is computed ...
        # NOTE: the matrix is returned with integer entries because integer computations
        #       can be carried out at maximum precision; this has to be converted to
        #       double precision for the LU decomposition
        self._l_and_u_: LAndUBandCounts
        self._penalty_matb_: np.ndarray = gen_squ_fw_fin_diff_mat_cho_banded(
            n_data=self.n_data_,
            differences=self.differences_,
            orig_first=False,
        ).astype(self.__dtype)

        # ... and cast to the banded storage format for LAPACK's LU decomposition
        self._l_and_u_, self._penalty_matb_ = (
            conv_upper_chol_banded_to_lu_banded_storage(ab=self._penalty_matb_)
        )

        # if the penalty weight is fitted automatically by maximization of the
        # log marginal likelihood, the natural logarithm of the pseudo-determinant of
        # D.T @ D is pre-computed together with the forward finite difference kernel
        self._diff_kernel_flipped_: np.ndarray = np.ndarray([], dtype=self.__dtype)
        self._penalty_mat_log_pseudo_det_: float = float("nan")
        if self._lam_inter_.fit_auto and self._lam_inter_.method_used in {
            WhittakerSmoothMethods.LOGML,
        }:
            # NOTE: the kernel is also returned with integer entries because integer
            #       computations can be carried out at maximum precision
            self._diff_kernel_flipped_ = np.flip(
                calc_forward_diff_kernel(differences=self.differences_)
            ).astype(self.__dtype)
            self._penalty_mat_log_pseudo_det_ = self._calc_penalty_log_pseudo_det()

        # finally, Pentapy is enabled if available, the number of differences is 2,
        # and the lambda parameter is not fitted automatically
        self._pentapy_enabled_: bool = (
            _PENTAPY_AVAILABLE
            and self.differences_ == 2
            and self.__allow_pentapy
            and not self._lam_inter_.fit_auto
        )

    ### Solver Methods ###

    def _solve_pentapy(self, ab: np.ndarray, b_weighted: np.ndarray) -> np.ndarray:
        """
        Solves the linear system of equations ``(W + lam * D.T @ D) @ x = W @ b``
        with the ``pentapy`` package. This is the same as solving the linear system
        ``A @ x = b`` where ``A = W + lam * D.T @ D`` and ``b = W @ b``.

        Notes
        -----
        Pentapy does not (maybe yet) allow for 2D right-hand side matrices, so the
        solution is computed for each column of ``bw`` separately.

        """  # noqa: E501

        # for 1-dimensional right-hand side vectors, the solution is computed directly
        if b_weighted.ndim == 1:
            return pp.solve(
                mat=ab,
                rhs=b_weighted,
                is_flat=True,
                index_row_wise=False,
                solver=1,
            )

        # for 2-dimensional right-hand side matrices, the solution is computed for each
        # column separately
        else:
            # NOTE: the solutions are first written into the rows of the solution matrix
            #       because row-access is more efficient for C-contiguous arrays;
            #       afterwards, the solution matrix is transposed
            solution = np.empty(shape=(b_weighted.shape[1], b_weighted.shape[0]))
            for iter_j in range(0, b_weighted.shape[1]):
                solution[iter_j, ::] = pp.solve(
                    mat=ab,
                    rhs=b_weighted[::, iter_j],
                    is_flat=True,
                    index_row_wise=False,
                    solver=1,
                )

            return solution.transpose()

    def _solve_pivoted_lu(
        self,
        ab: np.ndarray,
        b_weighted: np.ndarray,
    ) -> tuple[np.ndarray, BandedLUFactorization]:
        """
        Solves the linear system of equations ``(W + lam * D.T @ D) @ x = W @ b``
        with the LU decomposition. This is the same as solving the linear system
        ``A @ x = b`` where ``A = W + lam * D.T @ D`` and ``b = W @ b``.

        If the LU decomposition fails, a ``LinAlgError`` is raised which is fatal since
        the next level of escalation would be using a QR-decomposition which is not
        implemented (yet).

        """  # noqa: E501

        lub_factorization = lu_banded(
            l_and_u=self._l_and_u_,
            ab=ab,
            check_finite=False,
        )
        return (
            lu_solve_banded(
                lub_factorization=lub_factorization,
                b=b_weighted,
                check_finite=False,
                overwrite_b=True,
            ),
            lub_factorization,
        )

    def _solve(
        self,
        lam: float,
        b_weighted: np.ndarray,
        w: Union[float, np.ndarray],
    ) -> tuple[np.ndarray, BandedSolvers, _Factorization]:
        """
        Solves the linear system of equations ``(W + lam * D.T @ D) @ x = W @ b``
        where ``W`` is a diagonal matrix with the weights ``w`` on the main diagonal and
        ``D`` is the finite difference matrix of order ``differences``. ``lam``
        represents the penalty weight for the smoothing.
        For details on why the system is not formulated in a more efficient way, please
        refer to the Notes section.

        Parameters
        ----------
        lam : float
            The penalty weight lambda to use for the smoothing.
        b_weighted : ndarray of shape (m,) or (m, n)
            The weighted right-hand side vector or matrix of the linear system of
            equations given by ``W @ b``.
        w : float or ndarray of shape (m,)
            The weights to use for the linear system of equations given in terms of the
            main diagonal of the weight matrix ``W``.
            It can either be a vector of weights for each data point or a single
            scalar - namely ``1.0`` - if no weights are provided.

        Returns
        -------
        x : np.ndarray of shape (m,)
            The solution vector of the linear system of equations.
        decomposition_type : BandedSolveDecompositions
            The type of decomposition used to solve the linear system of equations.
        decomposition : BandedLUFactorization or BandedPentapyFactorization
            The decomposition used to solve the linear system of equations which is
            stored as a class instance specifying everything required to solve the
            system with the ``decomposition_type`` used.

        Raises
        ------
        RuntimeError
            If all available solvers failed to solve the linear system of equations
            which indicates a highly ill-conditioned system.

        Notes
        -----
        It might seem more efficient to solve the linear system ``((1.0 / lam) * W + D.T @ D) @ x = (1.0 / lam) * W @ b``
        because this only requires a multiplication of ``m`` weights with the reciprocal
        of the penalty weight whereas the multiplication with ``D.T @ D`` requires
        roughly ``m * (1 + 2 * differences)`` multiplications with ``m`` as the number
        of data points and ``differences`` as the difference order. On top of that,
        ``m * differences`` multiplications - so roughly 50% - would be redundant given
        that the penalty ``D.T @ D`` matrix is symmetric.
        However, NumPy's scalar multiplication is so highly optimized that the
        multiplication with ``D.T @ D`` without considering symmetry is almost as fast
        as the multiplication with the diagonal matrix ``W``, especially when compared
        to the computational load of the banded solvers.

        """  # noqa: E501

        # the banded storage format for the LAPACK LU decomposition is computed by
        # scaling the penalty matrix with the penalty weight lambda and then adding the
        # diagonal matrix with the weights
        ab = lam * self._penalty_matb_
        ab[self.differences_, ::] += w

        # the linear system of equations is solved with the most efficient method
        # Case 1: Pentapy can be used
        if self._pentapy_enabled_:
            x = self._solve_pentapy(ab=ab, b_weighted=b_weighted)
            if np.isfinite(x).all():
                return (
                    x,
                    BandedSolvers.PENTAPY,
                    BandedPentapyFactorization(),
                )

        # Case 2: LU decomposition (final fallback for pentapy)
        try:
            x, lub_factorization = self._solve_pivoted_lu(ab=ab, b_weighted=b_weighted)
            return x, BandedSolvers.PIVOTED_LU, lub_factorization

        except np.linalg.LinAlgError:
            available_solvers = f"{BandedSolvers.PIVOTED_LU}"
            if self._pentapy_enabled_:
                available_solvers = f"{BandedSolvers.PENTAPY}, {available_solvers}"

            raise RuntimeError(
                f"\nAll available solvers ({available_solvers}) failed to solve the "
                f"linear system of equations which indicates a highly ill-conditioned "
                f"system.\n"
                f"Please consider reducing the number of data points to smooth by, "
                f"e.g., binning or lowering the difference order."
            )

    ### Auxiliary Methods to prepare the data for the solver ###

    def calc_wrss(
        self, b: np.ndarray, b_smooth: np.ndarray, w: Union[float, np.ndarray]
    ) -> float:
        """
        Computes the (weighted) Sum of Squared Residuals (w)RSS between the original and
        the smoothed series.

        """

        # Case 1: no weights are provided
        if isinstance(w, float):
            return np.square(b - b_smooth).sum()

        # Case 2: weights are provided
        return (w * np.square(b - b_smooth)).sum()

    def _calc_log_marginal_likelihood(
        self,
        factorization: _FactorizationForLogMarginalLikelihood,
        log_lam: float,
        lam: float,
        b: np.ndarray,
        b_smooth: np.ndarray,
        w: Union[float, np.ndarray],
        w_plus_penalty_plus_n_samples_term: float,
    ) -> float:
        """
        Computes the log marginal likelihood for the automatic fitting of the penalty
        weight lambda. For the definitions used (and manipulated here), please refer to
        the Notes section.

        Parameters
        ----------
        factorization : BandedLUFactorization
            The factorization of the matrix to solve the linear system of equations,
            i.e., ``W + lambda * D.T @ D`` from the description above.
            Currently, only partially pivoted banded LU decompositions can be used to
            compute the log marginal likelihood.
        log_lam : float
            The natural logarithm of the penalty weight lambda used for the smoothing.
        lam : float
            The penalty weight lambda used for the smoothing, i.e., ``exp(log_lam)``.
        b, b_smooth : ndarray of shape (m,)
            The original series and its smoothed counterpart.
        w : float or ndarray of shape (m,)
            The weights to use for the smoothing.
        w_plus_penalty_plus_n_samples_term : float
            The last term of the log marginal likelihood that is constant since it
            involves the weights, the penalty matrix, and the number of data points
            which are all constant themselves (see the Notes for details).

        Notes
        -----
        The log marginal likelihood is given by:

        ``-0.5 * [wRSS + lambda * PSS - ln(pseudo_det(W)) - ln(pseudo_det(lambda * D.T @ D)) + ln(det(W + lambda * D.T @ D)) + (n^ - d) * ln(2 * pi)]``

        or better

        ``-0.5 * [wRSS + lambda * PSS - ln(pseudo_det(W)) - (n - d) * ln(lambda) - ln(det(D @ D.T)) + ln(det(W + lambda * D.T @ D)) + (n^ - d) * ln(2 * pi)]``

        where:

        - ``wRSS`` is the weighted Sum of Squared Residuals between the original and the
            smoothed series,
        - ``PSS`` is the Penalty Sum of Squares which is given by the sum of the squared
            elements of the ``d``-th order forward finite differences of the smoothed
            series,
        - ``d`` is the difference order used for the smoothing.
        - ``ln`` as the natural logarithm,
        - ``pseudo_det(A)`` is the pseudo-determinant of the matrix ``A``, i.e., the
            product of its non-zero eigenvalues,
        - ``det(A)`` is the determinant of the matrix ``A``, i.e., the product of its
            eigenvalues,
        - ``W`` is the diagonal matrix with the weights on the main diagonal,
        - ``D.T @ D`` is the squared forward finite differences matrix, and
        - ``n`` is the number of data points in the series to smooth,
        - ``n^`` is the number of data points with non-zero weights in the series to
            smooth.

        It should be noted that ``pseudo_det(D.T @ D)`` is replaced by ``det(D @ D.T)``
        here because the latter is not rank-deficient.

        """  # noqa: E501

        # first, the weighted Sum of Squared Residuals is computed ...
        wrss = self.calc_wrss(b=b, b_smooth=b_smooth, w=w)
        # ... followed by the Penalty Sum of Squares which requires the squared forward
        # finite differences of the smoothed series
        # NOTE: ``np.convolve`` is used to compute the forward finite differences and
        #       since it flips the provided kernel, an already flipped kernel is used
        pss = (
            lam
            * np.square(
                np.convolve(b_smooth, self._diff_kernel_flipped_, mode="valid")
            ).sum()
        )

        # besides the determinant of the combined left hand side matrix has to be
        # computed from its decomposition
        lhs_logdet_sign, lhs_logabsdet = slogdet_lu_banded(
            lub_factorization=factorization,
        )

        # if the sign of the determinant is positive, the log marginal likelihood is
        # computed and returned
        if lhs_logdet_sign > 0.0:
            return -0.5 * (
                wrss
                + pss
                - (b.size - self.differences_) * log_lam
                + lhs_logabsdet
                + w_plus_penalty_plus_n_samples_term
            )

        # otherwise, if the determinant is negative, the system is extremely
        # ill-conditioned and the log marginal likelihood cannot be computed
        raise RuntimeError(
            "\nThe determinant of the combined left hand side matrix "
            "W + lambda * D.T @ D is negative, indicating that the system is extremely "
            "ill-conditioned.\n"
            "The log marginal likelihood cannot be computed.\n"
            "Please consider reducing the number of data points to smooth by, e.g., "
            "binning or lowering the difference order."
        )

    def _marginal_likelihood_objective(
        self,
        log_lam: float,
        b: np.ndarray,
        w: Union[float, np.ndarray],
        w_plus_penalty_plus_n_samples_term: float,
    ) -> float:
        """
        The objective function to minimize for the automatic fitting of the penalty
        weight lambda by maximizing the log marginal likelihood.
        For the definition of the log marginal likelihood, please refer to the
        description of the method :meth:`_calc_log_marginal_likelihood`.

        """

        # first, the linear system of equations is solved with the given penalty weight
        # lambda
        lam = exp(log_lam)

        # Case 1: no weights are provided
        if isinstance(w, float):
            b_smooth, _, factorization = self._solve(
                lam=lam,
                b_weighted=b,
                w=w,
            )

        # Case 2: weights are provided
        else:
            b_smooth, _, factorization = self._solve(
                lam=lam,
                b_weighted=b * w,
                w=w,
            )

        # finally, the log marginal likelihood is computed and returned (negative since
        # the objective function is minimized, but the log marginal likelihood is
        # to be maximized)
        return (-1.0) * self._calc_log_marginal_likelihood(
            factorization=factorization,  # type: ignore
            log_lam=log_lam,
            lam=lam,
            b=b,
            b_smooth=b_smooth,
            w=w,
            w_plus_penalty_plus_n_samples_term=w_plus_penalty_plus_n_samples_term,
        )

    ### Solver management methods ###

    def _solve_single_b_fixed_lam(
        self,
        b: np.ndarray,
        w: Union[float, np.ndarray],
        lam: Optional[float] = None,
    ) -> tuple[np.ndarray, float]:
        """
        Solves for the Whittaker-like smoother solution for a single series with a fixed
        penalty weight lambda.

        """

        # if no value was provided for the penalty weight lambda, the respective class
        # attribute is used instead
        lam = self._lam_inter_.fixed_lambda if lam is None else lam

        # the weights and the weighted series are computed depending on whether weights
        # are provided or not
        # Case 1: no weights are provided
        if isinstance(w, float):
            return (
                self._solve(
                    lam=lam,
                    b_weighted=b.copy(),
                    w=w,
                )[0],
                lam,
            )

        # Case 2: weights are provided
        return (
            self._solve(
                lam=lam,
                b_weighted=b * w,
                w=w,
            )[0],
            lam,
        )

    def _solve_single_b_auto_lam_lml(
        self,
        b: np.ndarray,
        w: Union[float, np.ndarray],
    ) -> tuple[np.ndarray, float]:
        """
        Solves for the Whittaker-like smoother solution for a single series with an
        automatically fitted penalty weight lambda by maximizing the log marginal
        likelihood.

        """

        # first, the constant terms of the log marginal likelihood are computed starting
        # from the log pseudo-determinant of the weight matrix, i.e., the product of the
        # non-zero elements of the weight vector
        nnz_w = self.n_data_
        log_pseudo_det_w = 0.0  # ln(1**nnz_w) = 0.0
        if isinstance(w, np.ndarray):
            nonzero_w_idxs = np.where(w > w.max() * self.__zero_weight_tol)[0]
            nnz_w = nonzero_w_idxs.size
            log_pseudo_det_w = np.log(w[nonzero_w_idxs]).sum()

        # the constant term of the log marginal likelihood is computed
        w_plus_n_samples_term = (
            (nnz_w - self.differences_) * self.__LN_TWO_PI
            - log_pseudo_det_w
            - self._penalty_mat_log_pseudo_det_
        )

        # unless the search space spans less than 1 decade, i.e., ln(10) ~= 2.3, a grid
        # search is carried out to shrink the search space for the final optimization;
        # the grid is spanned with an integer number of steps of half a decade
        log_low_bound, log_upp_bound = self._lam_inter_.log_auto_bounds
        bound_log_diff = log_upp_bound - log_low_bound
        if bound_log_diff > self.__LN_TEN:
            half_decade = 0.5 * self.__LN_TEN
            target_best = float("inf")
            n_steps = 1 + ceil(bound_log_diff / half_decade)  #
            # NOTE: the following ensures that the upper bound is not exceeded
            step_size = bound_log_diff / (n_steps - 1)

            # all the trial values are evaluated and the best one is stored
            for trial in range(0, n_steps):
                log_lam_curr = log_low_bound + trial * step_size
                target_curr = self._marginal_likelihood_objective(
                    log_lam=log_lam_curr,
                    b=b,
                    w=w,
                    w_plus_penalty_plus_n_samples_term=w_plus_n_samples_term,
                )

                if target_curr < target_best:
                    log_lam_best = log_lam_curr
                    target_best = target_curr

            # then, the bounds for the final optimization are shrunk to plus/minus half
            # a decade around the best trial value
            # NOTE: the following ensures that the bounds are not violated
            log_low_bound = max(log_lam_best - half_decade, log_low_bound)
            log_upp_bound = min(log_lam_best + half_decade, log_upp_bound)

        # the optimization of the log marginal likelihood is carried out
        opt_res = minimize_scalar(
            fun=self._marginal_likelihood_objective,
            bounds=(log_low_bound, log_upp_bound),
            args=(b, w, w_plus_n_samples_term),
            method="bounded",
            options={"xatol": 0.05},
        )

        # the optimal penalty weight lambda is returned together with the smoothed
        # series
        return self._solve_single_b_fixed_lam(b=b, w=w, lam=exp(opt_res.x))

    def _solve_multiple_b(
        self,
        X: np.ndarray,
        w: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Solves for the Whittaker-like smoother solution for multiple series when the
        lambda parameter is fixed and the same weights are applied to all series.
        It leverages the ability of LAPACK (not ``pentapy``) to solve multiple linear
        systems of equations at once from the same factorization.

        For the parameters, please refer to the documentation of ``_solve``.

        """

        # then, the solution of the linear system of equations is computed for the
        # transposed series matrix (expected right-hand side format for the solvers)
        # Case 1: no weights are provided
        if w is None:
            X_smooth, _, _ = self._solve(
                lam=self._lam_inter_.fixed_lambda,
                b_weighted=X.transpose(),
                w=1.0,
            )

        # Case 2: weights are provided
        else:
            X_smooth, _, _ = self._solve(
                lam=self._lam_inter_.fixed_lambda,
                b_weighted=(X * w).transpose(),
                w=w[0, ::],
            )

        return (
            X_smooth.transpose(),
            np.full(shape=(X.shape[0],), fill_value=self._lam_inter_.fixed_lambda),
        )

    def _get_weight_generator(
        self, w: Optional[np.ndarray], n_series: int
    ) -> Generator[Union[float, np.ndarray], None, None]:
        """
        Generates a generator that yields the weights for each series in a series matrix
        ``X``.

        """

        # Case 1: No weights
        if w is None:
            for _ in range(n_series):
                yield 1.0

        # Case 2: 1D weights
        elif w.ndim == 1:
            for _ in range(n_series):
                yield w

        # Case 3: 2D weights
        elif w.ndim == 2:
            for w_vect in w:
                yield w_vect

    ### Main Solver Entry Point ###

    def _whittaker_solve(
        self,
        X: np.ndarray,
        *,
        w: Optional[np.ndarray] = None,
        use_same_w_for_all: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Solves for the Whittaker-like smoother solution for Arrays that are stored in
        2D format, i.e., each series is stored as a row.
        Internally it chooses the most appropriate method and solver depending on the
        data dimensionality, the weights, and the system's available packages
        (``pentapy``).

        Parameters
        ----------
        X : ndarray of shape (m, n)
            The series to be smoothed stored as individual rows.
        w : ndarray of shape(1, n) or shape(m, n) or None
            The weights to be applied for smoothing. If only a single row is provided
            and ``use_same_w_for_all`` is ``True``, the same weights can be applied
            for all series in ``X``, which enhances the smoothing a lot for fixed
            smoothing parameters ``lam``.
            If ``None``, no weights are applied and each datapoint is assumed to have
            equal importance. This allows for ``use_same_w_for_all`` to be ``True``
            as well.
        use_same_w_for_all
            Whether to use the same weights for all series in ``X``. This is only
            possible if ``w`` is a single row or ``None``.

        Returns
        -------
        X_smooth : ndarray of shape(m, n)
            The smoothed series stored as individual rows.
        lam : np.ndarray of shape(m, )
            The lambda parameter used for the smoothing of each series. If ``lam`` was
            fixed, this is a vector of length ``m`` with the same value for each series.

        """  # noqa: E501

        # if multiple x with the same weights are to be solved for fixed lambda, this
        # can be done more efficiently by leveraging LAPACK'S (not pentapy's) ability to
        # perform multiple solves from the same inversion at once
        if use_same_w_for_all and not self._lam_inter_.fit_auto:
            return self._solve_multiple_b(X=X, w=w)

        # otherwise, the solution of the linear system of equations is computed for
        # each series
        # first, the smoothing method is specified depending on whether the penalty
        # weight lambda is fitted automatically or not
        smooth_method_assignment = {
            WhittakerSmoothMethods.FIXED: self._solve_single_b_fixed_lam,
            WhittakerSmoothMethods.LOGML: self._solve_single_b_auto_lam_lml,
        }
        smooth_method = smooth_method_assignment[self._lam_inter_.method_used]

        # then, the solution is computed for each series by means of a loop
        X_smooth = np.empty_like(X)
        lam = np.empty(shape=(X.shape[0],))
        w_gen = self._get_weight_generator(w=w, n_series=X.shape[0])
        for iter_i, (x_vect, w_vect) in enumerate(zip(X, w_gen)):
            X_smooth[iter_i], lam[iter_i] = smooth_method(b=x_vect, w=w_vect)

        return X_smooth, lam

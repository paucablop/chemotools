"""
This utility submodule provides the base class for the Whittaker-like smoothing
algorithm. It is used to solve linear systems of equations that involve banded
matrices as they occur in applications like the Whittaker-Henderson-smoothing or
derived methods like Asymmetric Least Squares (ALS) baseline correction.

"""

### Imports ###

from math import exp
from typing import Optional, Union

import numpy as np

from chemotools._runtime import PENTAPY_AVAILABLE
from chemotools.utils import models
from chemotools.utils.banded_linalg import LAndUBandCounts, slogdet_lu_banded
from chemotools.utils.whittaker_base import auto_lambda as auto
from chemotools.utils.whittaker_base import initialisation as init
from chemotools.utils.whittaker_base import solvers
from chemotools.utils.whittaker_base.misc import get_weight_generator

### Type Aliases ###

_Factorization = Union[models.BandedLUFactorization, models.BandedPentapyFactorization]
_FactorizationForLogMarginalLikelihood = models.BandedLUFactorization


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
    _penalty_mat_banded_ : ndarray of shape (n_data - differences + 1, n_data - differences + 1)
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

    def __init__(self) -> None:  # pragma: no cover
        pass

    ### Initialization and Setup Methods ###

    def _setup_for_fit(
        self,
        n_data: int,
        differences: int,
        lam: init._LambdaSpecs,
    ) -> None:
        """
        Pre-computes everything that can be computed for the smoothing in general as
        well as for fitting the lambda parameter itself.

        For the parameters, please refer to the documentation of the class.

        """

        # the input arguments are stored and validated
        self.n_data_: int = n_data
        self.differences_: int = differences
        self._lam_inter_: models.WhittakerSmoothLambda = init.get_checked_lambda(
            lam=lam
        )

        # the squared forward finite difference matrix D.T @ D is computed in band
        # storage format for LAPACK's banded LU decomposition
        self._l_and_u_: LAndUBandCounts
        self._penalty_mat_banded_: np.ndarray
        self._l_and_u_, self._penalty_mat_banded_ = init.get_squ_fw_diff_mat_banded(
            n_data=self.n_data_,
            differences=self.differences_,
            orig_first=False,
            dtype=self.__dtype,
        )

        # if the penalty weight is fitted automatically by maximization of the
        # log marginal likelihood, the natural logarithm of the pseudo-determinant of
        # D.T @ D is pre-computed together with the forward finite difference kernel
        self._diff_kernel_flipped_: np.ndarray = np.ndarray([], dtype=self.__dtype)
        self._penalty_mat_log_pseudo_det_: float = float("nan")
        if self._lam_inter_.fit_auto and self._lam_inter_.method_used in {
            models.WhittakerSmoothMethods.LOGML,
        }:
            # NOTE: the kernel is also returned with integer entries because integer
            #       computations can be carried out at maximum precision
            self._diff_kernel_flipped_ = init.get_flipped_fw_diff_kernel(
                differences=self.differences_,
                dtype=self.__dtype,
            )
            self._penalty_mat_log_pseudo_det_ = init.get_penalty_log_pseudo_det(
                n_data=self.n_data_,
                differences=self.differences_,
                dtype=self.__dtype,
            )

        # finally, Pentapy is enabled if available, the number of differences is 2,
        # and the lambda parameter is not fitted automatically
        self._pentapy_enabled_: bool = (
            PENTAPY_AVAILABLE
            and self.differences_ == 2
            and self.__allow_pentapy
            and not self._lam_inter_.fit_auto
        )

    ### Solver Methods ###

    # TODO: implement solver that does not rely on normal equations
    def _solve(
        self,
        lam: float,
        b_weighted: np.ndarray,
        w: Union[float, np.ndarray],
    ) -> tuple[np.ndarray, models.BandedSolvers, _Factorization]:
        """
        Internal wrapper for the solver methods to solve the linear system of equations
        for the Whittaker-like smoother.
        It will first attempt to solve the system via the normal equations via either
        a direct pentadiagonal solve or an LU decomposition of the banded normal
        equations matrix. This is less numerically stable because the condition number
        of the normal equations matrix is the square of the condition number of the
        original system, but on the other hand, it can be way faster.
        If this fails, it will fall back to the more numerically stable QR
        decomposition (to be implemented).

        """  # noqa: E501

        return solvers.solve_normal_equations(
            lam=lam,
            differences=self.differences_,
            l_and_u=self._l_and_u_,
            penalty_mat_banded=self._penalty_mat_banded_,
            b_weighted=b_weighted,
            w=w,
            pentapy_enabled=self._pentapy_enabled_,
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
                    b_weighted=b,
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

        # if the weights are not provided, the log marginal likelihood cannot be
        # computed - at least not in a meaningful way
        if isinstance(w, (float, int)):
            raise ValueError(
                "\nAutomatic fitting of the penalty weight lambda by maximizing the "
                "log marginal likelihood is only possible if weights are provided.\n"
                "Please provide weights for the series to smooth."
            )

        # first, the constant terms of the log marginal likelihood are computed starting
        # from the log pseudo-determinant of the weight matrix, i.e., the product of the
        # non-zero elements of the weight vector
        nonzero_w_idxs = np.where(w > w.max() * self.__zero_weight_tol)[0]
        nnz_w = nonzero_w_idxs.size
        log_pseudo_det_w = np.log(w[nonzero_w_idxs]).sum()

        # the constant term of the log marginal likelihood is computed
        w_plus_n_samples_term = (
            (nnz_w - self.differences_) * self.__LN_TWO_PI
            - log_pseudo_det_w
            - self._penalty_mat_log_pseudo_det_
        )

        # the optimization of the log marginal likelihood is carried out
        opt_lambda = auto.get_optimized_lambda(
            fun=self._marginal_likelihood_objective,
            lam=self._lam_inter_,
            args=(b, w, w_plus_n_samples_term),
        )

        # the optimal penalty weight lambda is returned together with the smoothed
        # series
        return self._solve_single_b_fixed_lam(b=b, w=w, lam=opt_lambda)

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
            models.WhittakerSmoothMethods.FIXED: self._solve_single_b_fixed_lam,
            models.WhittakerSmoothMethods.LOGML: self._solve_single_b_auto_lam_lml,
        }
        smooth_method = smooth_method_assignment[self._lam_inter_.method_used]

        # then, the solution is computed for each series by means of a loop
        X_smooth = np.empty_like(X)
        lam = np.empty(shape=(X.shape[0],))
        w_gen = get_weight_generator(w=w, n_series=X.shape[0])
        for iter_i, (x_vect, w_vect) in enumerate(zip(X, w_gen)):
            X_smooth[iter_i], lam[iter_i] = smooth_method(b=x_vect, w=w_vect)

        return X_smooth, lam

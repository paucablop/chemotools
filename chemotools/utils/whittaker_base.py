"""
This utility submodule provides the base class for the Whittaker-like smoothing
algorithm. It is used to solve linear systems of equations that involve banded
matrices as they occur in applications like the Whittaker-Henderson-smoothing or
derived methods like Asymmetric Least Squares (ALS) baseline correction.

"""

### Imports ###

from typing import Generator, Optional, Union, overload

import numpy as np

from chemotools.utils.banded_linalg import (
    LAndUBandCounts,
    conv_upper_chol_banded_to_lu_banded_storage,
    lu_banded,
    lu_solve_banded,
    slogdet_lu_banded,
)
from chemotools.utils.finite_differences import gen_squ_fw_fin_diff_mat_cho_banded
from chemotools.utils.models import (
    _PENTAPY_AVAILABLE,
    AutoSmoothMethods,
    BandedLUFactorization,
    BandedPentapyFactorization,
    BandedSolvers,
)

if _PENTAPY_AVAILABLE:
    import pentapy as pp

### Type Aliases ###

_Decomposition = Union[BandedLUFactorization, BandedPentapyFactorization]


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
    lam_ : int or float or AutoSmoothMethods
        The lambda parameter to use for the smoothing, a.k.a. the penalty weight or
        smoothing parameter.
        If a member of :class:`AutoSmoothMethods` is provided, the lambda parameter is
        fitted automatically, but then the pre-computations in meth:`_setup_for_fit`
        and/or :meth:`_whittaker_solve` might take significantly longer because more
        pre-computations are required and multiple penalty weights are tested.
    differences_ : int
        The number of differences to use for the smoothing. If the aim is to obtain a
        smooth estimate of the ``m``-th order derivative, this should be set to
        at least ``m + 2``.
        For higher orders, the systems to solve tend to get numerically instable,
        especially when ``n_data_`` grows large and high values for ``lam_`` are used.
        Values below 1 are not allowed.
    _auto_fit_lam_ : bool
        Whether the lambda parameter is fitted automatically (``True``) or fixed
        (``False``).
    _l_and_u_ : (int, int)
        The number of sub- (first) and superdiagonals (second element) of the final
        matrix to solve for smoothing. Both elements will equal ``differences_``.
    _penalty_matb_ : ndarray of shape (n_data - differences + 1, n_data - differences + 1)
        The squared forward finite differences matrix ``D.T @ D`` stored in the banded
        storage format used for LAPACK's banded LU decomposition.
    _penalty_mat_log_pseudo_det_ : float
        The natural logarithm of the pseudo-determinant of the squared forward finite
        differences matrix ``D.T @ D`` which is used for the automatic fitting of the
        lambda parameter by maximizing the log marginal likelihood, i.e., when
        ``lam_ == AutoSmoothMethods.LOG_MARGINAL_LIKELIHOOD``.
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

    """  # noqa: E501

    __dtype: type = np.float64
    __allow_pentapy: bool = True

    def __init__(
        self,
    ) -> None:
        pass

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
        ).astype(np.float64)

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
            f"\nThe pseudo-determinant of the penalty matrix is negative, indicating "
            f"that the system is extremely ill-conditioned.\n"
            f"Automatic fitting for {self.n_data_} data points and difference order "
            f"{self.differences_} is not possible.\n"
            f"Please consider reducing the number of data points to smooth by, e.g., "
            f"binning or lowering the difference order."
        )

    def _setup_for_fit(
        self,
        n_data: int,
        lam: Union[int, float, AutoSmoothMethods],
        differences: int,
    ) -> None:
        """
        Pre-computes everything that can be computed for the smoothing in general as
        well as for fitting the lambda parameter itself.

        For the parameters, please refer to the documentation of the class.

        """

        # the input arguments are stored
        self.n_data_: int = n_data
        self.lam_: Union[int, float, AutoSmoothMethods] = lam
        self.differences_: int = differences

        # the squared forward finite difference matrix D.T @ D is computed ...
        # NOTE: the matrix is returned with integer entries because integer computations
        #       can be carried out at maximum precision; this has to be converted to
        #       double precision for the LU decomposition
        self._l_and_u_: LAndUBandCounts
        self._penalty_matb_: np.ndarray = gen_squ_fw_fin_diff_mat_cho_banded(
            n_data=self.n_data_,
            differences=self.differences_,
            orig_first=False,
        ).astype(np.float64)

        # ... and cast to the banded storage format for LAPACK's LU decomposition
        self._l_and_u_, self._penalty_matb_ = (
            conv_upper_chol_banded_to_lu_banded_storage(ab=self._penalty_matb_)
        )

        # if the penalty weight is fitted automatically by maximization of the
        # log marginal likelihood, the natural logarithm of the pseudo-determinant of
        # D.T @ D is pre-computed
        self._auto_fit_lam_: bool = isinstance(self.lam_, AutoSmoothMethods)
        self._penalty_mat_log_pseudo_det_: float = float("nan")
        if (
            self._auto_fit_lam_
            and self.lam_ == AutoSmoothMethods.LOG_MARGINAL_LIKELIHOOD
        ):
            self._penalty_mat_log_pseudo_det_: float = (
                self._calc_penalty_log_pseudo_det()
            )

        # finally, Pentapy is enabled if available, the number of differences is 2,
        # and the lambda parameter is not fitted automatically
        self._pentapy_enabled_: bool = (
            _PENTAPY_AVAILABLE
            and self.differences_ == 2
            and self.__allow_pentapy
            and not self._auto_fit_lam_
        )

    def _solve_pentapy(self, ab: np.ndarray, b_pen_weighted: np.ndarray) -> np.ndarray:
        """
        Solves the linear system of equations ``((1.0 / lam) * W + D.T @ D) @ x = (1.0 / lam) * W @ b``
        with the ``pentapy`` package. This is the same as solving the linear system
        ``A @ x = b`` where ``A = (1.0 / lam) * W + D.T @ D`` and ``b = (1.0 / lam) * W @ b``.

        Notes
        -----
        Pentapy does not (maybe yet) allow for 2D right-hand side matrices, so the
        solution is computed for each column of ``bw`` separately.

        """  # noqa: E501

        # for 1-dimensional right-hand side vectors, the solution is computed directly
        if b_pen_weighted.ndim == 1:
            return pp.solve(
                mat=ab,
                rhs=b_pen_weighted,
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
            solution = np.empty(
                shape=(b_pen_weighted.shape[1], b_pen_weighted.shape[0])
            )
            for iter_j in range(0, b_pen_weighted.shape[1]):
                solution[iter_j, ::] = pp.solve(
                    mat=ab,
                    rhs=b_pen_weighted[::, iter_j],
                    is_flat=True,
                    index_row_wise=False,
                    solver=1,
                )

            return solution.transpose()

    def _solve_pivoted_lu(
        self,
        ab: np.ndarray,
        b_pen_weighted: np.ndarray,
    ) -> tuple[np.ndarray, BandedLUFactorization]:
        """
        Solves the linear system of equations ``((1.0 / lam) * W + D.T @ D) @ x = (1.0 / lam) * W @ b``
        with the LU decomposition. This is the same as solving the linear system
        ``A @ x = b`` where ``A = (1.0 / lam) * W + D.T @ D`` and ``b = (1.0 / lam) * W @ b``.

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
                b=b_pen_weighted,
                check_finite=False,
                overwrite_b=True,
            ),
            lub_factorization,
        )

    def _solve(
        self,
        b_pen_weighted: np.ndarray,
        w_pen: np.ndarray,
    ) -> tuple[np.ndarray, BandedSolvers, _Decomposition]:
        """
        Solves the linear system of equations ``((1.0 / lam) * W + D^T @ D) @ x = (1.0 / lam) * W @ b``
        where ``W`` is a diagonal matrix with the weights ``w`` on the main diagonal and
        ``D`` is the finite difference matrix of order ``differences``.
        For details on why the system was formulated like this and not as usually done
        in the literature, please refer to the Notes section.

        Parameters
        ----------
        b_pen_weighted : ndarray of shape (m,) or (m, n)
            The penalized-weighted right-hand side vector or matrix of the linear system
            of equations given by ``(1.0 / lam) * W @ b``.
        log_lam : float
            The logarithm of the penalty weight lambda to use for the smoothing.
        w_pen : ndarray of shape (m,)
            The penalized weights to use for the linear system of equations given by
            ``(1.0 / lam) * W``.
            It must be a vector even if ``bw`` is a matrix because having ``bw`` as a
            matrix is only possible if lambda is fixed and the same weight vector has
            to be applied to all series

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
        Using the multiplication of the weight matrix ``W`` with the reciprocal of the
        penalty weight lambda ``1.0 / lam`` is way more efficient because ``W`` only
        possesses a single diagonal of non-zero elements while ``D.T @ D`` is a banded
        matrix with at least 3 diagonals for ``differences >= 1``. ``D.T @ D`` is even
        symmetric, so roughly 50% of the multiplications with ``D.T @ D`` would be
        redundant.
        Given a pre-computed ``(1.0) / lam * W``, the weighted right-hand side vector
        ``(1.0 / lam) * W @ b`` is computed by element-wise multiplication.
        So, instead of at least 3 * ``n_data`` only 2 * ``n_data`` multiplications are
        required. If the number of bands in ``D.T @ D`` is 5 (``differences == 2``), the
        number of multiplications is reduced by 60% already.

        """  # noqa: E501

        # the banded storage format for the LAPACK LU decomposition is computed by
        # updating the main diagonal of the penalty matrix with the penalized weights
        ab = self._penalty_matb_.copy()
        ab[self.differences_, ::] += w_pen

        # the linear system of equations is solved with the most efficient method
        # Case 1: Pentapy can be used
        if self._pentapy_enabled_:
            x = self._solve_pentapy(ab=ab, b_pen_weighted=b_pen_weighted)
            if np.isfinite(x).all():
                return (
                    x,
                    BandedSolvers.PENTAPY,
                    BandedPentapyFactorization(),
                )

        # Case 2: LU decomposition (final fallback for pentapy)
        try:
            x, lub_factorization = self._solve_pivoted_lu(
                ab=ab, b_pen_weighted=b_pen_weighted
            )
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

    @overload
    def _get_penalized_weights(self, w: None) -> float: ...

    @overload
    def _get_penalized_weights(self, w: np.ndarray) -> np.ndarray: ...

    def _get_penalized_weights(
        self, w: Optional[np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Computes the penalized weights to be used for the linear system of equations,
        i.e., ``(1.0 / lam) * W`` where ``W`` is a diagonal matrix with the weights
        ``w`` on the main diagonal.

        """

        # if no weights are provided, the penalized weights are simply the reciprocal of
        # the penalty weight lambda
        if w is None:
            return 1.0 / self.lam_  # type: ignore

        # otherwise, the penalized weights are the product of the reciprocal of the
        # penalty weight lambda and the weights
        # NOTE: instead of using divisions, the weights are multiplied with the
        #       reciprocal of the penalty weight lambda which is less numerically
        #       accurate but way faster
        return w * (1.0 / self.lam_)  # type: ignore

    def _solve_single_b_fixed_lam(
        self,
        b: np.ndarray,
        w: Optional[np.ndarray],
    ) -> tuple[np.ndarray, float]:
        """
        Solves for the Whittaker-like smoother solution for a single series with a fixed
        penalty weight lambda.

        For the parameters, please refer to the documentation of ``_solve``. Instead of
        a 2D-Array, a 1D-Array is expected for ``b`` and ``w``.

        """

        # the penalized weights are computed
        w_pen = self._get_penalized_weights(w=w)

        # finally, the solution is returned together with the lambda parameter
        return self._solve(b_pen_weighted=b * w_pen, w_pen=w_pen)[0], self.lam_  # type: ignore

    def _solve_single_b(
        self,
        b: np.ndarray,
        w: Optional[np.ndarray],
    ) -> tuple[np.ndarray, float]:
        """
        Solves for the Whittaker-like smoother solution for a single series with a fixed
        or fitted lambda parameter.

        For the parameters, please refer to the documentation of ``solve``. Instead of
        2D-Arrays, 1D-Arrays are expected for ``x`` and ``w``.

        """
        # then, the solution of the linear system of equations is computed
        # NOTE: this is a placeholder where an if-else-statement needs to be inserted
        #       for then the lambda parameter needs to be evaluated automatically
        if not self._auto_fit_lam_:
            return self._solve_single_b_fixed_lam(b=b, w=w)

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

        # the penalized weights are computed
        w_pen = self._get_penalized_weights(w=w)
        if isinstance(w_pen, float):
            w_pen = np.array([w_pen], dtype=self.__dtype)

        # then, the solution of the linear system of equations is computed for the
        # transposed series matrix (expected right-hand side format for the solvers)
        # FIXME: ``w_pen`` somehow becomes an integer for the type checker
        X_smooth, _, _ = self._solve(
            b_pen_weighted=(X * w_pen[np.newaxis, ::]).transpose(),  # type: ignore
            w_pen=w_pen,  # type: ignore
        )

        return (
            X_smooth.transpose(),
            np.full(shape=(X.shape[0],), fill_value=self.lam_),  # type: ignore
        )

    def _get_weight_generator(
        self, w: Optional[np.ndarray], n_series: int
    ) -> Generator[Optional[np.ndarray], None, None]:
        """
        Generates a generator that yields the weights for each series in a series matrix
        ``X``.

        """

        # Case 1: No weights
        if w is None:
            for _ in range(n_series):
                yield None

        # Case 2: 1D weights
        elif w.ndim == 1:
            for _ in range(n_series):
                yield w

        # Case 3: 2D weights
        elif w.ndim == 2:
            for w_vect in w:
                yield w_vect

    def _whittaker_solve(
        self,
        X: np.ndarray,
        *,
        w_vect: np.ndarray | None = None,
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
        if use_same_w_for_all:
            return self._solve_multiple_b(X=X, w=w_vect)

        # otherwise, the solution of the linear system of equations is computed for
        # each series
        X_smooth = np.empty_like(X)
        lam = np.empty(shape=(X.shape[0],))
        w_gen = self._get_weight_generator(w=w_vect, n_series=X.shape[0])
        for iter_i, (x_vect, w_vect) in enumerate(zip(X, w_gen)):
            X_smooth[iter_i], lam[iter_i] = self._solve_single_b(b=x_vect, w=w_vect)

        return X_smooth, lam


if __name__ == "__main__":

    import time

    from matplotlib import pyplot as plt

    NOISE_STDDEV = 0.05
    N_DATA = 1000
    N_NOISE_REALIZATIONS = 10

    x = np.linspace(0, 2 * np.pi, N_DATA)
    np.random.seed(42)
    y_singles = np.empty(shape=(N_NOISE_REALIZATIONS, N_DATA))
    noise_level = NOISE_STDDEV * (1 + 2 * np.abs(x - np.pi))
    for iter_i in range(N_NOISE_REALIZATIONS):
        y_singles[iter_i, ::] = np.cos(x) + np.random.normal(scale=noise_level)

    y_stddev = y_singles.std(axis=0, ddof=1)
    y = np.tile(y_singles.mean(axis=0)[np.newaxis, ::], reps=(2, 1))
    y += np.array([0.0, 1.0])[::, np.newaxis]

    start = time.time()
    tt = WhittakerLikeSolver()
    tt._setup_for_fit(n_data=x.size, lam=1e3, differences=1)
    weights = 1.0 / np.square(y_stddev)
    y_smooth, lam = tt._whittaker_solve(
        X=y,
        w_vect=np.array([weights, np.concatenate((weights[500:], weights[:500]))]),
        use_same_w_for_all=False,
    )
    print(f"Time: {(time.time() - start):.3f} seconds")

    fig, ax = plt.subplots()

    ax.plot(x, y.T, label="Original")
    for idx in range(0, y.shape[0]):
        ax.fill_between(
            x,
            y_smooth[idx, ::] - 2 * y_stddev,
            y_smooth[idx, ::] + 2 * y_stddev,
            alpha=0.5,
            label="Confidence Interval",
        )
    ax.plot(x, y_smooth.T, label="Smoothed")

    plt.show()

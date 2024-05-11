"""
This submodule contains the functions used for the automated fitting of the penalty
weight lambda within the ``WhittakerLikeSolver`` class based on the log marginal
likelihood that would have cluttered the class implementation.

"""

### Imports ###

from typing import Union

import numpy as np

from chemotools.utils import banded_linalg as bla
from chemotools.utils import models
from chemotools.utils.whittaker_base.auto_lambda.shared import get_smooth_wrss

### Constants ###

_LN_TWO_PI: float = 1.8378770664093453  # ln(2 * pi)

### Type Aliases ###

# TODO: add QR factorization
_FactorizationForLogMarginalLikelihood = models.BandedLUFactorization

### Functions ###


def get_log_marginal_likelihood_constant_term(
    differences: int,
    penalty_mat_log_pseudo_det: float,
    w: np.ndarray,
    zero_weight_tol: float,
) -> float:
    """
    Computes the constant term of the log marginal likelihood for the automatic fitting
    of the penalty weight lambda, i.e.,

    ``(n^ - d) * ln(2 * pi) - ln(pseudo_det(W)) - ln(pseudo_det(D.T @ D))``

    or better

    ``(n^ - d) * ln(2 * pi) - ln(pseudo_det(W)) - ln(det(D @ D.T))``

    with:

    - ``ln`` as the natural logarithm,
    - ``pseudo_det(A)`` as the pseudo-determinant of the matrix ``A``, i.e., the product
        of its non-zero eigenvalues,
    - ``det(A)`` as the determinant of the matrix ``A``, i.e., the product of its
        eigenvalues,
    - ``W`` as the diagonal matrix with the weights on the main diagonal,
    - ``D.T @ D`` as the squared forward finite differences matrix,
    - ``D @ D.T`` as the flipped squared forward finite differences matrix,
    - ``d`` as the difference order used for the smoothing, and
    - ``n^`` as the number of data points with non-zero weights in the series to smooth.

    It should be noted that ``pseudo_det(D.T @ D)`` is replaced by ``det(D @ D.T)`` here
    because the latter is not rank-deficient.

    """

    # first, the constant terms of the log marginal likelihood are computed starting
    # from the log pseudo-determinant of the weight matrix, i.e., the product of the
    # non-zero elements of the weight vector
    nonzero_w_idxs = np.where(w > w.max() * zero_weight_tol)[0]
    nnz_w = nonzero_w_idxs.size
    log_pseudo_det_w = np.log(w[nonzero_w_idxs]).sum()

    # the constant term of the log marginal likelihood is computed
    return (
        (nnz_w - differences) * _LN_TWO_PI
        - log_pseudo_det_w
        - penalty_mat_log_pseudo_det
    )


def get_log_marginal_likelihood(
    factorization: _FactorizationForLogMarginalLikelihood,
    log_lam: float,
    lam: float,
    differences: int,
    diff_kernel_flipped: np.ndarray,
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
    differences : int
        The order of the finite differences to use for the smoothing.
    diff_kernel_flipped : ndarray of shape (differences + 1,)
        The flipped forward finite differences kernel used for the smoothing.
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

    with:

    - ``wRSS`` as the weighted Sum of Squared Residuals between the original and the
        smoothed series,
    - ``PSS`` as the Penalty Sum of Squares which is given by the sum of the squared
        elements of the ``d``-th order forward finite differences of the smoothed
        series,
    - ``lambda`` as the penalty weight used for the smoothing,
    - ``d`` as the difference order used for the smoothing,
    - ``ln`` as the natural logarithm,
    - ``pseudo_det(A)`` as the pseudo-determinant of the matrix ``A``, i.e., the
        product of its non-zero eigenvalues,
    - ``det(A)`` as the determinant of the matrix ``A``, i.e., the product of its
        eigenvalues,
    - ``W`` as the diagonal matrix with the weights on the main diagonal,
    - ``D.T @ D`` as the squared forward finite differences matrix,
    - ``D @ D.T`` as the flipped squared forward finite differences matrix,
    - ``n`` is the number of data points in the series to smooth, and
    - ``n^`` is the number of data points with non-zero weights in the series to
        smooth.

    It should be noted that ``pseudo_det(D.T @ D)`` is replaced by ``det(D @ D.T)``
    here because the latter is not rank-deficient.

    """  # noqa: E501

    # first, the weighted Sum of Squared Residuals is computed ...
    wrss = get_smooth_wrss(b=b, b_smooth=b_smooth, w=w)
    # ... followed by the Penalty Sum of Squares which requires the squared forward
    # finite differences of the smoothed series
    # NOTE: ``np.convolve`` is used to compute the forward finite differences and
    #       since it flips the provided kernel, an already flipped kernel is used
    pss = (
        lam * np.square(np.convolve(b_smooth, diff_kernel_flipped, mode="valid")).sum()
    )

    # besides the determinant of the combined left hand side matrix has to be
    # computed from its decomposition
    lhs_logdet_sign, lhs_logabsdet = bla.slogdet_lu_banded(
        lub_factorization=factorization,
    )

    # if the sign of the determinant is positive, the log marginal likelihood is
    # computed and returned
    if lhs_logdet_sign > 0.0:
        return -0.5 * (
            wrss
            + pss
            - (b.size - differences) * log_lam
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

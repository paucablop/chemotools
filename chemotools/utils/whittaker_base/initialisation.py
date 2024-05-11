"""
This submodule contains the utility functions used at the initialisation of the
``WhittakerLikeSolver`` class that would have cluttered the class implementation.

"""

### Imports ###

from typing import Any, Tuple, Type, Union

import numpy as np

from chemotools.utils import banded_linalg as bla
from chemotools.utils import finite_differences as fdiff
from chemotools.utils import models

### Type Aliases ###

_RealNumeric = Union[int, float]
_WhittakerSmoothLambdaPlain = Tuple[
    _RealNumeric,
    _RealNumeric,
    models.WhittakerSmoothMethods,
]
_LambdaSpecs = Union[
    _RealNumeric,
    _WhittakerSmoothLambdaPlain,
    models.WhittakerSmoothLambda,
]

### Constants ###

_RealNumericTypes = (int, float)

### Functions ###


def get_checked_lambda(lam: Any) -> models.WhittakerSmoothLambda:
    """
    Checks the penalty weights lambda and casts it to the respective dataclass used
    inside the ``WhittakerLikeSolver`` class.

    """

    # if lambda is already the correct dataclass, it can be returned directly since all
    # the checks have already been performed
    if isinstance(lam, models.WhittakerSmoothLambda):
        return lam

    # now, there are other cases to check
    # Case 1: lambda is a single number
    if isinstance(lam, _RealNumericTypes):
        return models.WhittakerSmoothLambda(
            bounds=lam, method=models.WhittakerSmoothMethods.FIXED
        )

    # Case 2: lambda is a tuple
    if isinstance(lam, tuple):
        # if the tuple has the wrong length, an error is raised
        if len(lam) != 3:
            raise ValueError(
                f"\nThe lambda parameter must be a tuple of three elements (lower "
                f"bound, upper bound, method), but it has {len(lam)} elements "
                f"instead."
            )

        # otherwise, the tuple is unpacked and the dataclass is created
        return models.WhittakerSmoothLambda(
            bounds=(lam[0], lam[1]),
            method=lam[2],
        )

    # Case 3: lambda is not a valid type
    raise TypeError(
        f"\nThe lambda parameter must be an integer, a float, a tuple of (lower bound, "
        f"upper bound, method), or an instance of WhittakerSmoothLambda, but it is "
        f"{type(lam)} instead."
    )


def get_squ_fw_diff_mat_banded(
    n_data: int,
    differences: int,
    orig_first: bool,
    dtype: Type,
) -> Tuple[bla.LAndUBandCounts, np.ndarray]:
    """
    Returns the squared forward finite difference penalty matrix ``D.T @ D`` or its
    "flipped" counterpart ``D @ D.T`` in the banded storage format used for LAPACK's
    banded LU decomposition.

    """

    # the squared forward finite difference matrix D.T @ D or D @ D.T is generated ...
    # NOTE: the matrix is returned with integer entries because integer computations
    #       can be carried out at maximum precision; this has to be converted to
    #       double precision for the LU decomposition
    penalty_mat_banded = fdiff.gen_squ_fw_fin_diff_mat_cho_banded(
        n_data=n_data,
        differences=differences,
        orig_first=orig_first,
    ).astype(dtype)

    # ... and cast to the banded storage format for LAPACK's LU decomposition
    return bla.conv_upper_chol_banded_to_lu_banded_storage(ab=penalty_mat_banded)


def get_flipped_fw_diff_kernel(differences: int, dtype: Type) -> np.ndarray:
    """
    Returns the flipped forward finite difference kernel for the specified difference
    order.

    """

    return np.flip(fdiff.calc_forward_diff_kernel(differences=differences)).astype(
        dtype
    )


def get_penalty_log_pseudo_det(n_data: int, differences: int, dtype: Type) -> float:
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
        If the pseudo-determinant of the penalty matrix is negative, thereby indicating
        that the system is extremely ill-conditioned and the automatic fitting of the
        penalty weight is not possible.

    Notes
    -----
    Basically, this could be solved by evaluation of the eigenvalues of ``D.T @ D`` with
    a banded eigensolver, but this is computationally expensive and not necessary (the
    function is tested against this though).
    The pseudo-determinant of ``D.T @ D`` is the determinant of ``D @ D.T`` because
    ``D.T @ D`` is rank-deficient with ``differences`` zero eigenvalues while
    ``D @ D.T`` has full rank.
    Since both matrices share the same non-zero eigenvalues, the pseudo-determinant is
    easily computed as the determinant of ``D @ D.T`` via a partially pivoted LU
    decomposition.

    Throughout this function, the matrix ``D.T @ D`` is referred to as the "flipped
    penalty matrix" even though it is not actually flipped.

    """

    # the flipped penalty matrix D @ D.T is computed
    flipped_l_and_u, flipped_penalty_matb = get_squ_fw_diff_mat_banded(
        n_data=n_data,
        differences=differences,
        orig_first=True,
        dtype=dtype,
    )

    # the pseudo-determinant is computed from the partially pivoted LU decomposition
    # of the flipped penalty matrix
    log_pseudo_det_sign, log_pseudo_det = bla.slogdet_lu_banded(
        lub_factorization=bla.lu_banded(
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
        f"Automatic fitting for {n_data} data points and difference order "
        f"{differences} is not possible.\n"
        f"Please consider reducing the number of data points to smooth by, e.g., "
        f"binning or lowering the difference order."
    )

"""
This script implements utility functions required for testing the
:mod:`chemotools.utils` module.

It contains doctests itself that are executed when running the script, but they are
automatically tested when running the whole test suite as well. This ensures that the
test utilities are working as expected as well.

"""

### Imports ###

from math import exp, isnan
from typing import Tuple, Union

import numpy as np
from scipy.linalg import eigvals_banded
from scipy.optimize import brute, minimize_scalar
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse import diags as sp_diags
from scipy.sparse import linalg as spla

from chemotools.utils.finite_differences import calc_forward_diff_kernel
from chemotools.utils._whittaker_base import WhittakerLikeSolver

### Utility Functions ###


def float_is_bit_equal(value: float, reference: float) -> bool:
    """
    Checks if two floating-point numbers are equal up to the last bit and handles the
    case of NaN values as well.

    Doctests
    --------
    >>> # Imports
    >>> from tests.test_for_utils.utils_funcs import float_is_bit_equal

    >>> # Test 1
    >>> float_is_bit_equal(value=1.0, reference=1.0)
    True

    >>> # Test 2
    >>> float_is_bit_equal(value=1.0, reference=10.0)
    False

    >>> # Test 3
    >>> float_is_bit_equal(value=1.0, reference=float("nan"))
    False

    >>> # Test 4
    >>> float_is_bit_equal(value=float("nan"), reference=float("nan"))
    True

    >>> # Test 5
    >>> float_is_bit_equal(value=float("nan"), reference=1.0)
    False

    """

    if isnan(reference):
        return isnan(value)

    return value == reference


def conv_upper_cho_banded_storage_to_sparse(ab: np.ndarray) -> csr_matrix:
    """
    Converts a banded matrix stored in the upper banded storage used for LAPACK's banded
    Cholesky decomposition to a sparse ``CSR`` matrix.
    For more information on the banded storage, please see the documentation of
    :func:`chemotools.utils.banded_linalg.conv_upper_chol_banded_to_lu_banded_storage`.

    Doctests
    --------
    >>> # Imports
    >>> import numpy as np
    >>> from numpy import nan
    >>> from tests.test_for_utils.utils_funcs import (
    ...     conv_upper_cho_banded_storage_to_sparse,
    ... )

    >>> # Generating a set of test matrices
    >>> # Matrix 1
    >>> ab = np.array(
    ...     [
    ...         [nan, nan,  1.,  2.,  3.],
    ...         [nan,  4.,  5.,  6.,  7.],
    ...         [ 8.,  9., 10., 11., 12.],
    ...     ]
    ... )
    >>> conv_upper_cho_banded_storage_to_sparse(ab=ab).toarray()
    array([[ 8.,  4.,  1.,  0.,  0.],
           [ 4.,  9.,  5.,  2.,  0.],
           [ 1.,  5., 10.,  6.,  3.],
           [ 0.,  2.,  6., 11.,  7.],
           [ 0.,  0.,  3.,  7., 12.]])

    >>> # Matrix 2
    >>> ab = np.array(
    ...     [
    ...         [nan, nan, nan,  1.],
    ...         [nan, nan,  2.,  3.],
    ...         [nan,  4.,  5.,  6.],
    ...         [ 7.,  8.,  9., 10.],
    ...     ]
    ... )
    >>> conv_upper_cho_banded_storage_to_sparse(ab=ab).toarray()
    array([[ 7.,  4.,  2.,  1.],
           [ 4.,  8.,  5.,  3.],
           [ 2.,  5.,  9.,  6.],
           [ 1.,  3.,  6., 10.]])

    >>> # Matrix 3
    >>> ab = np.array(
    ...     [
    ...         [1., 2., 3., 4., 5.],
    ...     ]
    ... )
    >>> conv_upper_cho_banded_storage_to_sparse(ab=ab).toarray()
    array([[1., 0., 0., 0., 0.],
           [0., 2., 0., 0., 0.],
           [0., 0., 3., 0., 0.],
           [0., 0., 0., 4., 0.],
           [0., 0., 0., 0., 5.]])

    >>> # Matrix 4
    >>> ab = np.array(
    ...     [
    ...         [nan,  1.],
    ...         [ 2.,  3.],
    ...     ]
    ... )
    >>> conv_upper_cho_banded_storage_to_sparse(ab=ab).toarray()
    array([[2., 1.],
           [1., 3.]])

    >>> # Matrix 5
    >>> ab = np.array(
    ...     [
    ...         [nan, nan, nan, nan, nan, nan, nan, nan, nan,  1.],
    ...         [nan, nan, nan, nan, nan, nan, nan, nan,  2.,  3.],
    ...         [nan, nan, nan, nan, nan, nan, nan,  4.,  5.,  6.],
    ...         [nan, nan, nan, nan, nan, nan,  7.,  8.,  9., 10.],
    ...         [nan, nan, nan, nan, nan, 11., 12., 13., 14., 15.],
    ...         [nan, nan, nan, nan, 16., 17., 18., 19., 20., 21.],
    ...         [nan, nan, nan, 22., 23., 24., 25., 26., 27., 28.],
    ...         [nan, nan, 29., 30., 31., 32., 33., 34., 35., 36.],
    ...         [nan, 37., 38., 39., 40., 41., 42., 43., 44., 45.],
    ...         [46., 47., 48., 49., 50., 51., 52., 53., 54., 55.],
    ...     ]
    ... )
    >>> conv_upper_cho_banded_storage_to_sparse(ab=ab).toarray()
    array([[46., 37., 29., 22., 16., 11.,  7.,  4.,  2.,  1.],
           [37., 47., 38., 30., 23., 17., 12.,  8.,  5.,  3.],
           [29., 38., 48., 39., 31., 24., 18., 13.,  9.,  6.],
           [22., 30., 39., 49., 40., 32., 25., 19., 14., 10.],
           [16., 23., 31., 40., 50., 41., 33., 26., 20., 15.],
           [11., 17., 24., 32., 41., 51., 42., 34., 27., 21.],
           [ 7., 12., 18., 25., 33., 42., 52., 43., 35., 28.],
           [ 4.,  8., 13., 19., 26., 34., 43., 53., 44., 36.],
           [ 2.,  5.,  9., 14., 20., 27., 35., 44., 54., 45.],
           [ 1.,  3.,  6., 10., 15., 21., 28., 36., 45., 55.]])

    >>> conv_upper_cho_banded_storage_to_sparse(ab=ab[6::]).toarray()
    array([[46., 37., 29., 22.,  0.,  0.,  0.,  0.,  0.,  0.],
           [37., 47., 38., 30., 23.,  0.,  0.,  0.,  0.,  0.],
           [29., 38., 48., 39., 31., 24.,  0.,  0.,  0.,  0.],
           [22., 30., 39., 49., 40., 32., 25.,  0.,  0.,  0.],
           [ 0., 23., 31., 40., 50., 41., 33., 26.,  0.,  0.],
           [ 0.,  0., 24., 32., 41., 51., 42., 34., 27.,  0.],
           [ 0.,  0.,  0., 25., 33., 42., 52., 43., 35., 28.],
           [ 0.,  0.,  0.,  0., 26., 34., 43., 53., 44., 36.],
           [ 0.,  0.,  0.,  0.,  0., 27., 35., 44., 54., 45.],
           [ 0.,  0.,  0.,  0.,  0.,  0., 28., 36., 45., 55.]])

    """

    # the offset vector is initialised
    n_diags, n_cols = ab.shape
    n_diags -= 1
    main_diag_idx = n_diags
    offsets = np.arange(start=-n_diags, stop=n_diags + 1, step=1, dtype=np.int64)

    # then, the list of diagonals is created
    diagonals = []
    # the subdiagonals are added first ...
    for offset in range(n_diags, 0, -1):
        diagonals.append(ab[main_diag_idx - offset, offset:n_cols])

    # ... followed by the main diagonal ...
    diagonals.append(ab[main_diag_idx, ::])

    # ... and finally the superdiagonals
    for offset in range(1, n_diags + 1):
        diagonals.append(ab[main_diag_idx - offset, offset:n_cols])

    # the sparse matrix is created
    return sp_diags(  # type: ignore
        diagonals=diagonals,
        offsets=offsets,  # type: ignore
        shape=(n_cols, n_cols),
        format="csr",
    )


def conv_lu_banded_storage_to_sparse(
    ab: np.ndarray,
    l_and_u: Tuple[int, int],
) -> csr_matrix:
    """
    Converts a banded matrix stored in the banded storage used for LAPACK's banded LU
    decomposition into a sparse ``CSR`` matrix.
    For more information on the banded storage, please see the documentation of
    :func:`chemotools.utils.banded_linalg.conv_upper_chol_banded_to_lu_banded_storage`.

    Doctests
    --------
    >>> # Imports
    >>> import numpy as np
    >>> from numpy import nan
    >>> from tests.test_for_utils.utils_funcs import (
    ...     conv_lu_banded_storage_to_sparse,
    ... )

    >>> # Generating a set of test matrices
    >>> # Matrix 1
    >>> l_and_u = (1, 2)
    >>> ab = np.array(
    ...     [
    ...         [nan, nan,  1.,  2.,  3.],
    ...         [nan,  4.,  5.,  6.,  7.],
    ...         [ 8.,  9., 10., 11., 12.],
    ...         [13., 14., 15., 16., nan],
    ...     ]
    ... )
    >>> conv_lu_banded_storage_to_sparse(ab=ab, l_and_u=l_and_u).toarray()
    array([[ 8.,  4.,  1.,  0.,  0.],
           [13.,  9.,  5.,  2.,  0.],
           [ 0., 14., 10.,  6.,  3.],
           [ 0.,  0., 15., 11.,  7.],
           [ 0.,  0.,  0., 16., 12.]])

    >>> # Matrix 2
    >>> l_and_u = (2, 1)
    >>> ab = np.array(
    ...     [
    ...         [nan,  1.,  2.,  3.,  4.],
    ...         [ 5.,  6.,  7.,  8.,  9.],
    ...         [10., 11., 12., 13., nan],
    ...         [14., 15., 16., nan, nan],
    ...     ]
    ... )
    >>> conv_lu_banded_storage_to_sparse(ab=ab, l_and_u=l_and_u).toarray()
    array([[ 5.,  1.,  0.,  0.,  0.],
           [10.,  6.,  2.,  0.,  0.],
           [14., 11.,  7.,  3.,  0.],
           [ 0., 15., 12.,  8.,  4.],
           [ 0.,  0., 16., 13.,  9.]])

    >>> # Matrix 3
    >>> l_and_u = (0, 0)
    >>> ab = np.array(
    ...     [
    ...         [1., 2., 3., 4., 5.],
    ...     ]
    ... )
    >>> conv_lu_banded_storage_to_sparse(ab=ab, l_and_u=l_and_u).toarray()
    array([[1., 0., 0., 0., 0.],
           [0., 2., 0., 0., 0.],
           [0., 0., 3., 0., 0.],
           [0., 0., 0., 4., 0.],
           [0., 0., 0., 0., 5.]])

    >>> # Matrix 5
    >>> l_and_u = (5, 4)
    >>> ab = np.array(
    ...     [
    ...         [nan, nan, nan, nan,  1.,  2.,  3.,  4.,  5.],
    ...         [nan, nan, nan,  6.,  7.,  8.,  9., 10., 11.],
    ...         [nan, nan, 12., 13., 14., 15., 16., 17., 18.],
    ...         [nan, 19., 20., 21., 22., 23., 24., 25., 26.],
    ...         [27., 28., 29., 30., 31., 32., 33., 34., 35.],
    ...         [36., 37., 38., 39., 40., 41., 42., 43., nan],
    ...         [44., 45., 46., 47., 48., 49., 50., nan, nan],
    ...         [51., 52., 53., 54., 55., 56., nan, nan, nan],
    ...         [57., 58., 59., 60., 61., nan, nan, nan, nan],
    ...         [62., 63., 64., 65., nan, nan, nan, nan, nan],
    ...     ]
    ... )
    >>> conv_lu_banded_storage_to_sparse(ab=ab, l_and_u=l_and_u).toarray()
    array([[27., 19., 12.,  6.,  1.,  0.,  0.,  0.,  0.],
           [36., 28., 20., 13.,  7.,  2.,  0.,  0.,  0.],
           [44., 37., 29., 21., 14.,  8.,  3.,  0.,  0.],
           [51., 45., 38., 30., 22., 15.,  9.,  4.,  0.],
           [57., 52., 46., 39., 31., 23., 16., 10.,  5.],
           [62., 58., 53., 47., 40., 32., 24., 17., 11.],
           [ 0., 63., 59., 54., 48., 41., 33., 25., 18.],
           [ 0.,  0., 64., 60., 55., 49., 42., 34., 26.],
           [ 0.,  0.,  0., 65., 61., 56., 50., 43., 35.]])

    >>> l_and_u = (1, 4)
    >>> conv_lu_banded_storage_to_sparse(ab=ab[0:6, ::], l_and_u=l_and_u).toarray()
    array([[27., 19., 12.,  6.,  1.,  0.,  0.,  0.,  0.],
           [36., 28., 20., 13.,  7.,  2.,  0.,  0.,  0.],
           [ 0., 37., 29., 21., 14.,  8.,  3.,  0.,  0.],
           [ 0.,  0., 38., 30., 22., 15.,  9.,  4.,  0.],
           [ 0.,  0.,  0., 39., 31., 23., 16., 10.,  5.],
           [ 0.,  0.,  0.,  0., 40., 32., 24., 17., 11.],
           [ 0.,  0.,  0.,  0.,  0., 41., 33., 25., 18.],
           [ 0.,  0.,  0.,  0.,  0.,  0., 42., 34., 26.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0., 43., 35.]])

    >>> l_and_u = (2, 1)
    >>> conv_lu_banded_storage_to_sparse(ab=ab[3:7, ::], l_and_u=l_and_u).toarray()
    array([[27., 19.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [36., 28., 20.,  0.,  0.,  0.,  0.,  0.,  0.],
           [44., 37., 29., 21.,  0.,  0.,  0.,  0.,  0.],
           [ 0., 45., 38., 30., 22.,  0.,  0.,  0.,  0.],
           [ 0.,  0., 46., 39., 31., 23.,  0.,  0.,  0.],
           [ 0.,  0.,  0., 47., 40., 32., 24.,  0.,  0.],
           [ 0.,  0.,  0.,  0., 48., 41., 33., 25.,  0.],
           [ 0.,  0.,  0.,  0.,  0., 49., 42., 34., 26.],
           [ 0.,  0.,  0.,  0.,  0.,  0., 50., 43., 35.]])

    """

    # the offset vector is initialised
    n_low_bands, n_upp_bands = l_and_u
    main_diag_idx = n_upp_bands
    n_cols = ab.shape[1]
    offsets = np.arange(
        start=-n_low_bands,
        stop=n_upp_bands + 1,
        step=1,
        dtype=np.int64,
    )

    # then, the list of diagonals is created
    diagonals = []
    # the subdiagonals are added first ...
    for offset in range(n_low_bands, 0, -1):
        diagonals.append(ab[main_diag_idx + offset, 0 : n_cols - offset])

    # ... followed by the main diagonal ...
    diagonals.append(ab[main_diag_idx, ::])

    # ... and finally the superdiagonals
    for offset in range(1, n_upp_bands + 1):
        diagonals.append(ab[main_diag_idx - offset, offset:n_cols])

    # the matrix is created from the diagonals
    return sp_diags(  # type: ignore
        diagonals=diagonals,
        offsets=offsets,  # type: ignore
        shape=(n_cols, n_cols),
        format="csr",
    )


def multiply_vect_with_squ_fw_fin_diff_orig_first(
    differences: int,
    kernel: np.ndarray,
    vector: np.ndarray,
) -> np.ndarray:
    """
    Multiplies a vector with the squared forward finite difference matrix ``D @ D.T``
    where ``D`` is the forward finite difference matrix.
    Here, the original matrix ``D`` and not its transpose is used first.

    This is the same operation as a convolution with the flipped kernel after zero-
    padding the vector. Then, the result is again convolved with the kernel, but this
    time there is neither zero-padding nor flipping involved.
    ``y = D.T @ x`` is the zero-padding and flipping operation, and ``D @ y`` is the
    convolution without zero-padding and flipping.

    Doctests
    --------
    >>> # Imports
    >>> import numpy as np
    >>> from tests.test_for_utils.utils_funcs import (
    ...     multiply_vect_with_squ_fw_fin_diff_orig_first,
    ... )

    >>> # Test 1
    >>> differences = 1
    >>> kernel = np.array([-1, 1])
    >>> vector = np.array([1, 2])
    >>> multiply_vect_with_squ_fw_fin_diff_orig_first(
    ...     differences=differences,
    ...     kernel=kernel,
    ...     vector=vector,
    ... )
    array([0, 3])

    >>> # Test 2
    >>> differences = 1
    >>> kernel = np.array([-1, 1])
    >>> vector = np.array([-10,   3,  11])
    >>> multiply_vect_with_squ_fw_fin_diff_orig_first(
    ...     differences=differences,
    ...     kernel=kernel,
    ...     vector=vector,
    ... )
    array([-23,   5,  19])

    >>> # Test 3
    >>> differences = 1
    >>> kernel = np.array([-1, 1])
    >>> vector = np.array([ 25,  17, -13, -12])
    >>> multiply_vect_with_squ_fw_fin_diff_orig_first(
    ...     differences=differences,
    ...     kernel=kernel,
    ...     vector=vector,
    ... )
    array([ 33,  22, -31, -11])

    >>> # Test 4
    >>> differences = 2
    >>> kernel = np.array([1, -2, 1])
    >>> vector = np.array([1, 2, 3])
    >>> multiply_vect_with_squ_fw_fin_diff_orig_first(
    ...     differences=differences,
    ...     kernel=kernel,
    ...     vector=vector,
    ... )
    array([ 1, -4, 11])

    >>> # Test 5
    >>> differences = 2
    >>> kernel = np.array([1, -2, 1])
    >>> vector = np.array([-10,   3,  11,  27])
    >>> multiply_vect_with_squ_fw_fin_diff_orig_first(
    ...     differences=differences,
    ...     kernel=kernel,
    ...     vector=vector,
    ... )
    array([-61,  41, -64, 121])

    >>> # Test 6
    >>> differences = 2
    >>> kernel = np.array([1, -2, 1])
    >>> vector = np.array([ 25,  17, -13, -12,  38])
    >>> multiply_vect_with_squ_fw_fin_diff_orig_first(
    ...     differences=differences,
    ...     kernel=kernel,
    ...     vector=vector,
    ... )
    array([  69,   42,  -35, -155,  263])

    >>> # Test 7
    >>> differences = 3
    >>> kernel = np.array([-1, 3, -3, 1])
    >>> vector = np.array([1, 2, 3, 4])
    >>> multiply_vect_with_squ_fw_fin_diff_orig_first(
    ...     differences=differences,
    ...     kernel=kernel,
    ...     vector=vector,
    ... )
    array([  4,   4, -24,  46])

    >>> # Test 8
    >>> differences = 3
    >>> kernel = np.array([-1, 3, -3, 1])
    >>> vector = np.array([-10,   3,  11,  27,  -5])
    >>> multiply_vect_with_squ_fw_fin_diff_orig_first(
    ...     differences=differences,
    ...     kernel=kernel,
    ...     vector=vector,
    ... )
    array([-206,  212, -320,  478, -442])

    >>> # Test 9
    >>> differences = 3
    >>> kernel = np.array([-1, 3, -3, 1])
    >>> vector = np.array([ 25,  17, -13, -12,  38,  -8])
    >>> multiply_vect_with_squ_fw_fin_diff_orig_first(
    ...     differences=differences,
    ...     kernel=kernel,
    ...     vector=vector,
    ... )
    array([ 179,   50,   51, -586,  965, -789])


    """

    # first, the zero-padded vector is convolved with the flipped kernel
    vector_padded = np.pad(
        vector,
        pad_width=(differences, differences),
        mode="constant",
        constant_values=0,
    )
    # NOTE: since NumPy already flips the kernel internally, there is no need to flip it
    vector_conv = np.convolve(vector_padded, kernel, mode="valid")

    # then, the result is convolved with the kernel
    # NOTE: here, the kernel has to be flipped to counteract NumPy's internal flipping
    return np.convolve(vector_conv, np.flip(kernel), mode="valid")


def multiply_vect_with_squ_fw_fin_diff_transpose_first(
    differences: int,
    kernel: np.ndarray,
    vector: np.ndarray,
) -> np.ndarray:
    """
    Multiplies a vector with the squared forward finite difference matrix ``D.T @ D``
    where ``D`` is the forward finite difference matrix.
    Here, the transpose matrix ``D.T`` and not the original matrix is used first.

    This is the same operation as a convolution with the kernel followed by another
    convolution with the flipped kernel with an intermediate zero-padding.
    ``y = D @ x`` is the convolution with the kernel, and ``D.T @ y`` is the convolution
    with the flipped kernel and zero-padding.

    Doctests
    --------
    >>> # Imports
    >>> import numpy as np
    >>> from tests.test_for_utils.utils_funcs import (
    ...     multiply_vect_with_squ_fw_fin_diff_transpose_first,
    ... )

    >>> # Test 1
    >>> differences = 1
    >>> kernel = np.array([-1, 1])
    >>> vector = np.array([1, 2])
    >>> multiply_vect_with_squ_fw_fin_diff_transpose_first(
    ...     differences=differences,
    ...     kernel=kernel,
    ...     vector=vector,
    ... )
    array([-1,  1])

    >>> # Test 2
    >>> differences = 1
    >>> kernel = np.array([-1, 1])
    >>> vector = np.array([-10,   3,  11])
    >>> multiply_vect_with_squ_fw_fin_diff_transpose_first(
    ...     differences=differences,
    ...     kernel=kernel,
    ...     vector=vector,
    ... )
    array([-13,   5,   8])

    >>> # Test 3
    >>> differences = 1
    >>> kernel = np.array([-1, 1])
    >>> vector = np.array([ 25,  17, -13, -12])
    >>> multiply_vect_with_squ_fw_fin_diff_transpose_first(
    ...     differences=differences,
    ...     kernel=kernel,
    ...     vector=vector,
    ... )
    array([  8,  22, -31,   1])

    >>> # Test 4
    >>> differences = 2
    >>> kernel = np.array([1, -2, 1])
    >>> vector = np.array([1, 2, 3])
    >>> multiply_vect_with_squ_fw_fin_diff_transpose_first(
    ...     differences=differences,
    ...     kernel=kernel,
    ...     vector=vector,
    ... )
    array([0, 0, 0])

    >>> # Test 5
    >>> differences = 2
    >>> kernel = np.array([1, -2, 1])
    >>> vector = np.array([-10,   3,  11,  27])
    >>> multiply_vect_with_squ_fw_fin_diff_transpose_first(
    ...     differences=differences,
    ...     kernel=kernel,
    ...     vector=vector,
    ... )
    array([ -5,  18, -21,   8])

    >>> # Test 6
    >>> differences = 2
    >>> kernel = np.array([1, -2, 1])
    >>> vector = np.array([ 25,  17, -13, -12,  38])
    >>> multiply_vect_with_squ_fw_fin_diff_transpose_first(
    ...     differences=differences,
    ...     kernel=kernel,
    ...     vector=vector,
    ... )
    array([-22,  75, -35, -67,  49])

    >>> # Test 7
    >>> differences = 3
    >>> kernel = np.array([-1, 3, -3, 1])
    >>> vector = np.array([1, 2, 3, 4])
    >>> multiply_vect_with_squ_fw_fin_diff_transpose_first(
    ...     differences=differences,
    ...     kernel=kernel,
    ...     vector=vector,
    ... )
    array([0, 0, 0, 0])

    >>> # Test 8
    >>> differences = 3
    >>> kernel = np.array([-1, 3, -3, 1])
    >>> vector = np.array([-10,   3,  11,  27,  -5])
    >>> multiply_vect_with_squ_fw_fin_diff_transpose_first(
    ...     differences=differences,
    ...     kernel=kernel,
    ...     vector=vector,
    ... )
    array([ -13,   95, -207,  181,  -56])

    >>> # Test 9
    >>> differences = 3
    >>> kernel = np.array([-1, 3, -3, 1])
    >>> vector = np.array([ 25,  17, -13, -12,  38,  -8])
    >>> multiply_vect_with_squ_fw_fin_diff_transpose_first(
    ...     differences=differences,
    ...     kernel=kernel,
    ...     vector=vector,
    ... )
    array([ -53,  141,   40, -436,  453, -145])

    """

    # first, the vector is convolved with the kernel
    # NOTE: here, the kernel has to be flipped to counteract NumPy's internal flipping
    vector_conv = np.convolve(vector, np.flip(kernel), mode="valid")

    # then, the result is convolved with the flipped kernel and zero-padded
    vector_padded = np.pad(
        vector_conv,
        pad_width=(differences, differences),
        mode="constant",
        constant_values=0,
    )
    # NOTE: since NumPy already flips the kernel internally, there is no need to flip it
    return np.convolve(vector_padded, kernel, mode="valid")


def get_banded_slogdet(ab: np.ndarray) -> Tuple[float, float]:
    """
    Computes the sign and the logarithm of the determinant of a banded matrix stored
    in the upper banded storage used for LAPACK's banded Cholesky decomposition.

    Doctests
    --------
    >>> # Imports
    >>> import numpy as np
    >>> from tests.test_for_utils.utils_funcs import (
    ...     conv_upper_cho_banded_storage_to_sparse,
    ...     get_banded_slogdet,
    ... )

    >>> # Generating a set of test matrices
    >>> np.random.seed(42)

    >>> # Matrix 1 (positive definite)
    >>> semi_bw_plus_one = 3
    >>> # NOTE: the diagonal lifting makes the matrix positive definite
    >>> ab_for_chol = np.random.rand(semi_bw_plus_one, 100)
    >>> ab_for_chol[semi_bw_plus_one - 1, ::] += 1.0 + 2.0 * float(semi_bw_plus_one)
    >>> # the sign and the log determinant are computed by the utility function ...
    >>> sign, logabsdet = get_banded_slogdet(ab=ab_for_chol)
    >>> sign, logabsdet
    (1.0, 200.55218150013826)
    >>> # ... and by NumPy's dense log determinant function for comparison
    >>> ab_dense = conv_upper_cho_banded_storage_to_sparse(ab=ab_for_chol).toarray()
    >>> sign_ref, logabsdet_ref = np.linalg.slogdet(ab_dense)
    >>> sign_ref, logabsdet_ref
    (1.0, 200.55218150013826)
    >>> np.isclose(sign, sign_ref)
    True
    >>> np.isclose(logabsdet, logabsdet_ref)
    True

    >>> # Matrix 2 (positive definite)
    >>> semi_bw_plus_one = 5
    >>> ab_for_chol = np.random.rand(semi_bw_plus_one, 1000)
    >>> ab_for_chol[semi_bw_plus_one - 1, ::] += 1.0 + 2.0 * float(semi_bw_plus_one)
    >>> # the sign and the log determinant are computed by the utility function ...
    >>> sign, logabsdet = get_banded_slogdet(ab=ab_for_chol)
    >>> sign, logabsdet
    (1.0, 2432.2672133727287)
    >>> # ... and by NumPy's dense log determinant function for comparison
    >>> ab_dense = conv_upper_cho_banded_storage_to_sparse(ab=ab_for_chol).toarray()
    >>> sign_ref, logabsdet_ref = np.linalg.slogdet(ab_dense)
    >>> sign_ref, logabsdet_ref
    (1.0, 2432.267213372733)
    >>> np.isclose(sign, sign_ref)
    True
    >>> np.isclose(logabsdet, logabsdet_ref)
    True

    >>> # Matrix 3 (positive definite)
    >>> semi_bw_plus_one = 1
    >>> ab_for_chol = np.random.rand(semi_bw_plus_one, 5000)
    >>> ab_for_chol[semi_bw_plus_one - 1, ::] += 1.0 + 2.0 * float(semi_bw_plus_one)
    >>> # the sign and the log determinant are computed by the utility function ...
    >>> sign, logabsdet = get_banded_slogdet(ab=ab_for_chol)
    >>> sign, logabsdet
    (1.0, 6234.8131295042585)
    >>> # ... and by NumPy's dense log determinant function for comparison
    >>> ab_dense = conv_upper_cho_banded_storage_to_sparse(ab=ab_for_chol).toarray()
    >>> sign_ref, logabsdet_ref = np.linalg.slogdet(ab_dense)
    >>> sign_ref, logabsdet_ref
    (1.0, 6234.8131295042585)
    >>> np.isclose(sign, sign_ref)
    True
    >>> np.isclose(logabsdet, logabsdet_ref)
    True

    >>> # Matrix 4 (indefinite)
    >>> semi_bw_plus_one = 2
    >>> ab_for_chol = -1.0 + 2.0 * np.random.rand(semi_bw_plus_one, 1000)
    >>> # the sign and the log determinant are computed by the utility function ...
    >>> sign, logabsdet = get_banded_slogdet(ab=ab_for_chol)
    >>> sign, logabsdet
    (-1.0, -437.7731132082764)
    >>> # ... and by NumPy's dense log determinant function for comparison
    >>> ab_dense = conv_upper_cho_banded_storage_to_sparse(ab=ab_for_chol).toarray()
    >>> sign_ref, logabsdet_ref = np.linalg.slogdet(ab_dense)
    >>> sign_ref, logabsdet_ref
    (-1.0, -437.7731132082757)
    >>> np.isclose(sign, sign_ref)
    True
    >>> np.isclose(logabsdet, logabsdet_ref)
    True

    >>> # Matrix 5 (indefinite)
    >>> semi_bw_plus_one = 1
    >>> ab_for_chol = -1.0 + 2.0 * np.random.rand(semi_bw_plus_one, 5000)
    >>> # the sign and the log determinant are computed by the utility function ...
    >>> sign, logabsdet = get_banded_slogdet(ab=ab_for_chol)
    >>> sign, logabsdet
    (1.0, -5001.0078551404185)
    >>> # ... and by NumPy's dense log determinant function for comparison
    >>> ab_dense = conv_upper_cho_banded_storage_to_sparse(ab=ab_for_chol).toarray()
    >>> sign_ref, logabsdet_ref = np.linalg.slogdet(ab_dense)
    >>> sign_ref, logabsdet_ref
    (1.0, -5001.007855140422)
    >>> np.isclose(sign, sign_ref)
    True
    >>> np.isclose(logabsdet, logabsdet_ref)
    True

    """
    # since the log determinant can be expressed as the sum of the logarithms of the
    # absolute eigenvalues, an eigenvalue evaluation is sufficient to determine the
    # sign and the log determinant
    eigvals = eigvals_banded(a_band=ab, lower=False, select="a")
    if np.count_nonzero(eigvals < 0.0) % 2 == 0:  # type: ignore
        sign = 1.0
    else:
        sign = -1.0

    with np.errstate(divide="ignore", over="ignore"):
        logabsdet = np.log(np.abs(eigvals)).sum()  # type: ignore

    return sign, logabsdet


def get_sparse_fw_fin_diff_mat(n_data: int, differences: int) -> csc_matrix:
    """
    Creates a dense forward finite difference matrix ``D`` of a given difference order.

    Doctests
    --------
    >>> # Imports
    >>> from tests.test_for_utils.utils_funcs import get_sparse_fw_fin_diff_mat

    >>> # Matrix 1
    >>> n_data, differences = 5, 1
    >>> get_sparse_fw_fin_diff_mat(n_data=n_data, differences=differences).toarray()
    array([[-1.,  1.,  0.,  0.,  0.],
           [ 0., -1.,  1.,  0.,  0.],
           [ 0.,  0., -1.,  1.,  0.],
           [ 0.,  0.,  0., -1.,  1.]])

    >>> # Matrix 2
    >>> n_data, differences = 10, 1
    >>> get_sparse_fw_fin_diff_mat(n_data=n_data, differences=differences).toarray()
    array([[-1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0., -1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0., -1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0., -1.,  1.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0., -1.,  1.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0., -1.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0., -1.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  1.]])

    >>> # Matrix 3
    >>> n_data, differences = 5, 2
    >>> get_sparse_fw_fin_diff_mat(n_data=n_data, differences=differences).toarray()
    array([[ 1., -2.,  1.,  0.,  0.],
           [ 0.,  1., -2.,  1.,  0.],
           [ 0.,  0.,  1., -2.,  1.]])

    >>> # Matrix 4
    >>> n_data, differences = 10, 2
    >>> get_sparse_fw_fin_diff_mat(n_data=n_data, differences=differences).toarray()
    array([[ 1., -2.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  1., -2.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1., -2.,  1.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  1., -2.,  1.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  1., -2.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  1., -2.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  1., -2.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1., -2.,  1.]])

    >>> # Matrix 4
    >>> n_data, differences = 5, 3
    >>> get_sparse_fw_fin_diff_mat(n_data=n_data, differences=differences).toarray()
    array([[-1.,  3., -3.,  1.,  0.],
           [ 0., -1.,  3., -3.,  1.]])

    >>> # Matrix 5
    >>> n_data, differences = 10, 3
    >>> get_sparse_fw_fin_diff_mat(n_data=n_data, differences=differences).toarray()
    array([[-1.,  3., -3.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0., -1.,  3., -3.,  1.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0., -1.,  3., -3.,  1.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0., -1.,  3., -3.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0., -1.,  3., -3.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0., -1.,  3., -3.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0., -1.,  3., -3.,  1.]])

    """

    # first, the required constants are obtained from the ``WhittakerLikeSolver``-class
    dtype = WhittakerLikeSolver._WhittakerLikeSolver__dtype  # type: ignore

    # then, the dense finite difference matrix D is created from the forward difference
    # kernel
    diff_kernel = calc_forward_diff_kernel(differences=differences)
    offsets = np.arange(start=0, stop=diff_kernel.size, step=1, dtype=np.int64)
    return sp_diags(
        diagonals=diff_kernel,
        offsets=offsets,  # type: ignore
        shape=(n_data - diff_kernel.size + 1, n_data),
        dtype=dtype,
        format="csc",
    )


def sparse_slogdet_from_superlu(splu: spla.SuperLU) -> Tuple[float, float]:
    """
    Computes the sign and the logarithm of the determinant of a sparse matrix from its
    SuperLU decomposition.

    References
    ----------
    This function is based on the following GIST and its discussion:
    https://gist.github.com/luizfelippesr/5965a536d202b913beda9878a2f8ef3e

    Doctests
    --------
    >>> # Imports
    >>> import numpy as np
    >>> import scipy.sparse as sprs

    >>> from tests.test_for_utils.utils_funcs import (
    ...     sparse_slogdet_from_superlu,
    ... )

    >>> # Setup of a test with random matrices
    >>> np.random.seed(42)
    >>> n_rows = np.random.randint(low=10, high=1_001, size=20)
    >>> density = 0.5 # chosen to have a high probability of a solvable system
    >>> n_rows
    array([112, 445, 870, 280, 116,  81, 710,  30, 624, 131, 476, 224, 340,
           468,  97, 382, 109, 881, 673, 140])

    >>> # Running the tests in a loop
    >>> for m in n_rows:
    ...     iter_i = 0
    ...     attempts = 10
    ...     failed = False
    ...     while iter_i < 10:
    ...         # a random matrix is generated and if the LU decomposition fails, the
    ...         # test is repeated (this test is not there to test the LU decomposition)
    ...         attempts += 1
    ...         matrix = sprs.random(m=m, n=m, density=density, format="csc")
    ...         try:
    ...             splu = sprs.linalg.splu(matrix)
    ...         except RuntimeError:
    ...             continue
    ...
    ...         # first, the utility function is used to compute the sign and the log
    ...         # determinant of the matrix
    ...         sign, logabsdet = sparse_slogdet_from_superlu(splu=splu)
    ...
    ...         # then, the sign and the log determinant are computed by NumPy's dense
    ...         # log determinant function for comparison
    ...         sign_ref, logabsdet_ref = np.linalg.slogdet(matrix.toarray())
    ...
    ...         # the results are compared and if they differ, the test is stopped
    ...         # with a diagnostic message
    ...         if not (
    ...             np.isclose(sign, sign_ref)
    ...             and np.isclose(logabsdet, logabsdet_ref)
    ...         ):
    ...             print(
    ...                 f"Failed for matrix with shape {m}x{m}: "
    ...                 f"sign: {sign} vs. {sign_ref} and "
    ...                 f"logabsdet: {logabsdet} vs. {logabsdet_ref}"
    ...             )
    ...             failed = True
    ...             break
    ...
    ...         # if the test is successful, the loop is continued if the number of
    ...         # attempts is less than 100
    ...         del splu
    ...         iter_i += 1
    ...         if attempts >= 100:
    ...             print(
    ...                 f"Could not generate a solvable system for matrix with shape "
    ...                 f"{m}x{m}"
    ...             )
    ...
    ...     if failed:
    ...         break

    """

    ### Auxiliary Function ###

    def find_min_num_swaps(arr: np.ndarray):
        """
        Minimum number of swaps needed to order a permutation array.

        """
        # from https://www.thepoorcoder.com/hackerrank-minimum-swaps-2-solution/
        a = dict(enumerate(arr))
        b = {v: k for k, v in a.items()}
        count = 0
        for i in a:
            x = a[i]
            if x != i:
                y = b[i]
                a[y] = x
                b[x] = y
                count += 1

        return count

    ### Main Part ###

    # the logarithm of the determinant is the sum of the logarithms of the diagonal
    # elements of the LU decomposition, but since L is unit lower triangular, only the
    # diagonal elements of U are considered
    diagU = splu.U.diagonal()
    logabsdet = np.log(np.abs(diagU)).sum()

    # then, the sign is determined from the diagonal elements of U as well as the row
    # and column permutations
    # NOTE: odd number of negative elements/swaps leads to a negative sign
    fact_sign = -1 if np.count_nonzero(diagU < 0.0) % 2 == 1 else 1
    row_sign = -1 if find_min_num_swaps(splu.perm_r) % 2 == 1 else 1
    col_sign = -1 if find_min_num_swaps(splu.perm_c) % 2 == 1 else 1
    sign = -1.0 if fact_sign * row_sign * col_sign < 0 else 1.0

    return sign, logabsdet


def calc_whittaker_smooth_log_marginal_likelihood_const_term(
    differences: int,
    diff_mat: csc_matrix,
    weight_vect: np.ndarray,
) -> float:
    """
    Calculates the constant term of the log marginal likelihood of a Whittaker smoother
    with a given set of parameters.

    It is given by

    ``(n^ - d) * ln(2 * pi) - ln(pseudo_det(W)) - ln(pseudo_det(D.T @ D))``

    or better

    ``(n^ - d) * ln(2 * pi) - ln(pseudo_det(W)) - ln(det(D @ D.T))``

    For further details, please see the documentation of the function :func:`get_log_marginal_likelihood_constant_term`
    from the module :mod:`chemotools.utils.whittaker_base.logml`.

    Doctest
    -------
    >>> # Imports
    >>> import numpy as np
    >>> from tests.test_for_utils.utils_funcs import (
    ...     calc_whittaker_smooth_log_marginal_likelihood_const_term,
    ...     get_sparse_fw_fin_diff_mat,
    ... )

    >>> # Generation of the weight matrix W and the finite difference matrix D
    >>> weights = np.array([0.5, 1.0, 0.5, 1.0, 0.5])
    >>> n_data, differences = weights.size, 1
    >>> diff_mat = get_sparse_fw_fin_diff_mat(
    ...     n_data=n_data,
    ...     differences=differences,
    ... )
    >>> diff_mat_dense = diff_mat.toarray()

    >>> # Test 1 with all weights being non-zero

    >>> # Calculation of the log pseudo-determinant of the weight matrix W
    >>> # since it is diagonal, the log-determinant is the sum of the logarithms of the
    >>> # diagonal elements
    >>> log_pseudo_det_w = np.log(weights).sum()
    >>> log_pseudo_det_w
    -2.0794415416798357

    >>> # Calculation of the log pseudo-determinant via the Cholesky decomposition of
    >>> # the product D @ D.T
    >>> squ_diff_mat_chol = np.linalg.cholesky(diff_mat_dense @ diff_mat_dense.T)
    >>> squ_diff_mat_chol
    array([[ 1.41421356,  0.        ,  0.        ,  0.        ],
           [-0.70710678,  1.22474487,  0.        ,  0.        ],
           [ 0.        , -0.81649658,  1.15470054,  0.        ],
           [ 0.        ,  0.        , -0.8660254 ,  1.11803399]])
    >>> # the sum of the doubled logarithms of the main diagonal elements is the log
    >>> # pseudo-determinant of the matrix D.T @ D
    >>> log_pseudo_det_dtd = 2.0 * np.log(np.diag(squ_diff_mat_chol)).sum()
    >>> log_pseudo_det_dtd
    1.6094379124341003

    >>> # Calculation of the theoretical constant term
    >>> logml_theoretical = (
    ...    (n_data - differences) * np.log(2.0 * np.pi)
    ...    - log_pseudo_det_w
    ...    - log_pseudo_det_dtd
    ... )

    >>> # Calculation of the constant term via the utility function
    >>> logml_via_function = calc_whittaker_smooth_log_marginal_likelihood_const_term(
    ...     differences=differences,
    ...     diff_mat=diff_mat,
    ...     weight_vect=weights,
    ... )
    >>> logml_via_function
    7.821511894883117
    >>> np.isclose(logml_via_function, logml_theoretical)
    True

    >>> # Test 2 with 2 weights being zero
    >>> weights[1] = 0.0
    >>> weights[3] = 0.0
    >>> nonzero_weights_flags = weights > 0.0
    >>> log_pseudo_det_w = np.log(weights[nonzero_weights_flags]).sum()

    >>> # Calculation of the theoretical constant term
    >>> logml_theoretical = (
    ...    (nonzero_weights_flags.sum() - differences) * np.log(2.0 * np.pi)
    ...    - log_pseudo_det_w
    ...    - log_pseudo_det_dtd
    ... )

    >>> # Calculation of the constant term via the utility function
    >>> logml_via_function = calc_whittaker_smooth_log_marginal_likelihood_const_term(
    ...     differences=differences,
    ...     diff_mat=diff_mat,
    ...     weight_vect=weights,
    ... )
    >>> logml_via_function
    4.145757762064426
    >>> np.isclose(logml_via_function, logml_theoretical)
    True

    """  # noqa: E501

    ### Pre-computation of the constant term ###

    # first, the required constants are obtained from the ``WhittakerLikeSolver``-class
    zero_weight_tol = WhittakerLikeSolver._WhittakerLikeSolver__zero_weight_tol  # type: ignore

    # for W, the log pseudo-determinant is calculated ...
    w_nonzero_idxs = weight_vect > weight_vect.max() * zero_weight_tol
    nnz_w = w_nonzero_idxs.sum()
    w_log_pseudo_det = np.log(weight_vect[w_nonzero_idxs]).sum()

    # ... followed by the log pseudo-determinant of the penalty matrix D.T @ D which is
    # equivalent to the determinant of the flipped matrix D @ D.T which is not
    # rank-deficient
    _, penalty_log_pseudo_det = sparse_slogdet_from_superlu(
        splu=spla.splu(A=diff_mat @ diff_mat.transpose())
    )

    # from all of this, the constant term is computed
    return (
        (nnz_w - differences) * np.log(2.0 * np.pi)
        - w_log_pseudo_det
        - penalty_log_pseudo_det
    )


def find_whittaker_smooth_opt_lambda_log_marginal_likelihood(
    b_vect: np.ndarray,
    weight_vect: np.ndarray,
    differences: int,
    log_lambda_bounds: Tuple[float, float],
    n_opts: int,
) -> Tuple[float, float, np.ndarray]:
    """
    Finds the optimal lambda value for a Whittaker smoother by maximising the log
    marginal likelihood via a nested brute-force optimisation followed by a bounded
    scalar minimisation.

    Since it relies purely on dense linear algebra for highly sparse matrices, this
    utility function is only suitable for small to medium-sized datasets (n < 500 ...
    1000).

    """

    ### Definition of the target function ###

    def get_smooth_solution(
        log_lam: Union[np.ndarray, float]
    ) -> Tuple[np.ndarray, spla.SuperLU, float, float]:
        """
        Computes the smooth solution for the Whittaker smoother.

        """

        # first, the linear system (left hand side) has to be set up for calculating the
        # smooth solution
        if isinstance(log_lam, np.ndarray):
            log_lam = log_lam[0]

        lam = exp(log_lam)

        lhs_mat = lam * penalty_mat
        lhs_mat += sp_diags(
            diagonals=weight_vect,
            offsets=0,
            shape=(b_vect.size, b_vect.size),
            format="csc",
        )

        # then, the solution is obtained
        lhs_splu = spla.splu(A=lhs_mat)
        smooth_solution = lhs_splu.solve(rhs=weight_vect * b_vect)

        return (
            smooth_solution,
            lhs_splu,
            lam,
            log_lam,  # type: ignore
        )

    def logml_target_func(log_lam: Union[np.ndarray, float]) -> float:
        """
        The target function to minimize for maximizing the log marginal likelihood.

        """

        # first, the smooth solution is calculated together with the left-hand side
        # matrix and the lambda value
        smooth_solution, lhs_splu, lam, log_lam = get_smooth_solution(log_lam=log_lam)

        # the log-determinant of the lhs matrix is calculated
        _, logdet_lhs = sparse_slogdet_from_superlu(splu=lhs_splu)

        # finally, the log marginal likelihood is computed from:
        # 1) the weighted residual sum of squares
        wrss = (weight_vect * np.square(b_vect - smooth_solution)).sum()

        # 2) the sum of squared penalties
        # NOTE: the order of multiplications for the following term is important because
        #       the last multiplication is a matrix-vector resulting in another vector;
        #       the other way around would result in another matrix followed by
        #       a matrix-vector multiplication
        pss = lam * (smooth_solution @ (penalty_mat @ smooth_solution))

        # 3) the log-determinant of the lhs matrix and the constant term
        # NOTE: the sign is positive because the log marginal likelihood is maximised
        #       and not minimised
        return 0.5 * (
            wrss
            + pss
            - (b_vect.size - differences) * log_lam
            + logdet_lhs
            + logml_constant_term
        )

    ### Pre-computations ###

    # then, some pre-computations are made
    n_data = b_vect.size
    log_lambda_min, log_lambda_max = log_lambda_bounds
    diff_mat = get_sparse_fw_fin_diff_mat(
        n_data=n_data,
        differences=differences,
    )
    penalty_mat = (diff_mat.transpose() @ diff_mat).tocsc()  # type: ignore
    logml_constant_term = calc_whittaker_smooth_log_marginal_likelihood_const_term(
        differences=differences,
        diff_mat=diff_mat,
        weight_vect=weight_vect,
    )

    ### Running the optimisation ###

    # the first optimisation is run with the target function to narrow down the
    # search space
    opt_log_lam = brute(
        func=logml_target_func,
        ranges=((log_lambda_min, log_lambda_max),),
        Ns=n_opts,
        finish=None,
        full_output=False,
    )

    # the search space is narrowed down for the second optimisation to roughly one
    # decade in the natural log space
    log_lambda_min = opt_log_lam - 1.2  # type: ignore
    log_lambda_max = opt_log_lam + 1.2  # type: ignore

    # the second optimisation is run with the target function to find the optimal lambda
    opt_log_lam = brute(
        func=logml_target_func,
        ranges=((log_lambda_min, log_lambda_max),),
        Ns=n_opts,
        finish=None,
        full_output=False,
    )

    # one more optimisation is run to ensure that the optimal lambda is found
    log_lambda_min = opt_log_lam - 0.1  # type: ignore
    log_lambda_max = opt_log_lam + 0.1  # type: ignore
    opt_log_lam = minimize_scalar(
        fun=logml_target_func,
        bounds=(log_lambda_min, log_lambda_max),
        method="bounded",
    ).x

    # finally, the solutions for the optimal lambda are returned
    return (
        exp(opt_log_lam),
        (-1.0) * logml_target_func(log_lam=opt_log_lam),
        get_smooth_solution(log_lam=opt_log_lam)[0],
    )


### Doctests ###

if __name__ == "__main__":  # pragma: no cover

    import doctest

    doctest.testmod()

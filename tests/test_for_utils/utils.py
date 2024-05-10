"""
This script implements utility functions required for testing the
:mod:`chemotools.utils` module.

It contains doctests itself that are executed when running the script, but they are
automatically tested when running the whole test suite as well. This ensures that the
test utilities are working as expected as well.

"""

### Imports ###

from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import diags as sp_diags

### Utility Functions ###


def conv_upper_cho_banded_storage_to_sparse(
    ab: np.ndarray,
) -> csr_matrix:
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
    >>> from tests.test_for_utils.utils import conv_upper_cho_banded_storage_to_sparse

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
    ab: np.ndarray, l_and_u: Tuple[int, int]
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
    >>> from tests.test_for_utils.utils import conv_lu_banded_storage_to_sparse

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
    >>> from tests.test_for_utils.utils import (
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
    >>> from tests.test_for_utils.utils import (
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


### Doctests ###

if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod()

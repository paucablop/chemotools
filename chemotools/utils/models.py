"""
This utility submodule implements important models, i.e., constants, Enums, and
dataclasses used throughout the package.

"""

### Imports ###

from dataclasses import dataclass, field
from enum import Enum
from math import log
from typing import Literal, Tuple, Union

import numpy as np

### Enums ###

# an Enum class for the solve types used for solving linear systems that involve banded
# matrices


class BandedSolvers(str, Enum):
    """
    Defines the types of solvers that can be used to solve linear systems involving
    banded matrices, i.e.,

    - ``PIVOTED_LU``: LU decomposition with partial pivoting
    - ``PENTAPY``: pentadiagonal "decomposition" (it's actually a direct solve)

    """

    PIVOTED_LU = "partially pivoted LU decomposition"
    PENTAPY = "direct pentadiagonal solver"


# an Enum class for the kinds of smoothing by the Whittaker-Henderson smoother that can
# be applied to the data


class WhittakerSmoothMethods(str, Enum):
    """
    Defines the types of smoothing methods that can be applied to the data using the
    Whittaker-Henderson smoother, i.e.,

    - ``FIXED``: fixed penalty weight (shorthand "fixed")
    - ``LOGML``: smoothing based on the maximization of the log marginal likelihood
        (shorthand "logml")

    Except for ``FIXED``, the penalty weight is automatically determined when using the
    other methods.

    """

    FIXED = "fixed"
    LOGML = "logml"


# a type hint is defined for the Whittaker-Henderson smoother specification
_WhittakerSmoothMethodsAll = Union[WhittakerSmoothMethods, Literal["fixed", "logml"]]

### (Data) Classes ###


# a dataclass for specification of the smoothing penalty weight lambda for the
# Whittaker-Henderson smoother


@dataclass()
class WhittakerSmoothLambda:
    """
    A dataclass that holds the specification of the smoothing penalty weight or
    smoothing parameter lambda for the Whittaker-Henderson smoother.

    Attributes
    ----------
    bounds: int or float or (int or float, int or float)
        The bounds for the search space of the penalty weight lambda. The specification
        can be either

        - a single value for a fixed penalty weight (requires ``method`` to be set to
            ``WhittakerSmoothMethods.FIXED``), or
        - a tuple of two values for the lower and upper bounds of the search space
            (then ``method`` may not be set to ``WhittakerSmoothMethods.FIXED`` unless
            the bounds are too close to each other as described below).

        Independently of the specification, the values have to be greater than or equal
        to the zero tolerance ``1e-25``.
        If a lower and an upper bound are provided, they are flipped if necessary.
        After that, the difference ``abs(upp_bound - low_bound)`` has to be at least
        ``1e-5 * upp_bound`` for any method other than ``WhittakerSmoothMethods.FIXED``.
        Otherwise, the method is set to ``WhittakerSmoothMethods.FIXED`` and the
        ``fixed_lambda`` is set to the upper bound.
    method: WhittakerSmoothMethods or {"fixed", "logml"}
        The method to use for the selection of the penalty weight. If the bounds are too
        close to each other, this will be set to ``WhittakerSmoothMethods.FIXED``.

    Raises
    ------
    ValueError
        If ``method`` is invalid, i.e., it does not correspond to any of the
        ``WhittakerSmoothMethods`` or their shorthands, or if it cannot be used in
        combination with ``bounds``.
    ValueError
        If the bounds are invalid, i.e., they are not greater than or equal to the zero
        tolerance ``1e-25``.

    """

    bounds: Union[int, float, tuple[Union[int, float], Union[int, float]]]
    method: _WhittakerSmoothMethodsAll

    fixed_lambda: float = field(default=float("nan"), init=False)
    auto_bounds: tuple[float, float] = field(
        default=(float("nan"), float("nan")), init=False
    )
    method_used: WhittakerSmoothMethods = field(
        default=WhittakerSmoothMethods.FIXED, init=False
    )
    fit_auto: bool = field(default=False, init=False)

    __zero_tol: float = field(default=1e-25, init=False, repr=False)
    __diff_tol: float = field(default=1e-5, init=False, repr=False)

    def _validate_n_set_method(self) -> None:
        try:
            self.method_used = WhittakerSmoothMethods(self.method)
        except ValueError:
            raise ValueError(
                f"\nThe method '{self.method}' is not valid. "
                f"Please choose one of the following: "
                f"'fixed', 'logml', {WhittakerSmoothMethods.FIXED.name}, "
                f"{WhittakerSmoothMethods.LOGML.name}."
            )

    def __post_init__(self):
        # the bounds are checked for validity
        # Case 1: a single value is provided
        if isinstance(self.bounds, (int, float)):
            # first, the method is validated
            self._validate_n_set_method()

            # in this case, the method has to be set to FIXED
            if self.method_used != WhittakerSmoothMethods.FIXED:
                raise ValueError(
                    f"\nThe method '{self.method_used.name}' was selected for a fixed "
                    f"penalty weight (i.e., bounds are just a scalar)."
                )

            # the bound has to be greater than or equal to the zero tolerance
            if self.bounds < self.__zero_tol:
                raise ValueError(
                    f"\nThe penalty weight lambda has to be greater than or equal to "
                    f"the zero tolerance {self.__zero_tol}."
                )

            # the fixed lambda is set to the bound
            self.fixed_lambda = float(self.bounds)
            self.fit_auto = False

            return

        # Case 2: a tuple of two values is provided
        elif isinstance(self.bounds, tuple):

            # the bounds are flipped if necessary
            low_bound, upp_bound = sorted(self.bounds)

            # the bounds have to be greater than or equal to the zero tolerance
            if low_bound < self.__zero_tol or upp_bound < self.__zero_tol:
                raise ValueError(
                    f"\nThe bounds for the penalty weight lambda have to be greater "
                    f"than or equal to the zero tolerance {self.__zero_tol}, but "
                    f"they are {low_bound} and {upp_bound}."
                )

            # the difference has to be at least 1e-5 * upp_bound to be considered
            # as a search space
            if abs(upp_bound - low_bound) >= self.__diff_tol * upp_bound:
                # for this, the method is validated
                self._validate_n_set_method()

                # if the method is not FIXED, the bounds are set as the search space
                if self.method_used != WhittakerSmoothMethods.FIXED:
                    self.auto_bounds = (float(low_bound), float(upp_bound))
                    self.fit_auto = True
                    return

                # if the bounds are a search space, but the method is set to FIXED,
                # an error is raised
                raise ValueError(
                    f"\nThe bounds for the penalty weight lambda are a search space "
                    f"({low_bound}, {upp_bound}), but the method is set to FIXED."
                )

            # otherwise, if the penalty weights is fixed, the method is set to FIXED as
            # well
            self.method_used = WhittakerSmoothMethods.FIXED
            self.fixed_lambda = float(upp_bound)
            self.fit_auto = False

            return

        # Case 3: the bounds are neither a scalar nor a tuple of two values
        raise TypeError(
            f"\nThe bounds for the penalty weight lambda have to be either a scalar "
            f"or a tuple of two values, but they are {self.bounds}."
        )

    @property
    def log_auto_bounds(self) -> Tuple[float, float]:
        """
        The natural logarithms of the search space bounds for the penalty weight lambda.

        Returns
        -------
        log_auto_bounds : (float, float)
            The natural logarithms of the lower and upper bounds of the search space.

        """

        return (log(self.auto_bounds[0]), log(self.auto_bounds[1]))


# a fake class for representing the factorization of a pentadiagonal matrix with
# pentapy which is empty since pentapy does not factorize the matrix but directly solves
# the system of equations


class BandedPentapyFactorization:
    """
    A class that resembles the factorization of a pentadiagonal matrix with ``pentapy``.
    It has no attributes since the factorization is not stored, but the class is used to
    provide an easy way to check if the factorization is available.

    """

    pass


# a dataclass for the factorization of a banded matrix with LU decomposition with p
# partial pivoting


@dataclass()
class BandedLUFactorization:
    """
    A dataclass that holds the partially pivoted LU factorization of a banded matrix.

    Attributes
    ----------
    lub: ndarray of shape (n_rows, n_cols)
        The LU factorization of the matrix ``A`` in banded storage format.
    ipiv: ndarray of shape (n_rows,)
        The pivot indices.
    l_and_u: tuple[int, int]
        The number of lower and upper bands in the LU factorization.
    singular: bool
        If ``True``, the matrix ``A`` is singular.
    shape : (int, int)
        The shape of the matrix ``A`` in dense form.
    n_rows, n_cols : int
        The number of rows and columns of the matrix ``A`` in dense form.
    main_diag_row_idx : int
        The index of the main diagonal in the banded storage format.

    """

    lub: np.ndarray
    ipiv: np.ndarray
    l_and_u: tuple[int, int]
    singular: bool

    shape: tuple[int, int] = field(default=(-1, -1), init=False)
    n_rows: int = field(default=-1, init=False)
    n_cols: int = field(default=-1, init=False)
    main_diag_row_idx: int = field(default=-1, init=False)

    def __post_init__(self):
        self.shape = self.lub.shape  # type: ignore
        self.n_rows, self.n_cols = self.shape
        self.main_diag_row_idx = self.n_rows - 1 - self.l_and_u[0]

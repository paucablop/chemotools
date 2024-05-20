"""
This module contains the ``WhittakerSmooth`` transformer, which performs smoothing on
data according to the Whittaker-Henderson formulation of Penalized Least Squares.

References
----------
It's based on the algorithms described in [1]_ and [2]_ where an implementational
adaption of [2]_ was required to make it numerically stable ([3]_).

.. [1] Z.-M. Zhang, S. Chen, and Y.-Z. Liang, "Baseline correction using adaptive
   iteratively reweighted penalized least squares", Analyst 135 (5), 1138-1146 (2010)
.. [2] G. Biessy, "Revisiting Whittaker-Henderson smoothing", arXiv:2306.06932 (2023)
.. [3] https://math.stackexchange.com/q/4819039/1261538

"""

# Authors:
# Pau Cabaneros
# Niklas Zell <nik.zoe@web.de>

### Imports ###

from typing import Literal, Optional, Tuple, Union

from numpy import ndarray
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from chemotools.utils._types import RealNumeric
from chemotools.utils._whittaker_base import (
    WhittakerLikeSolver,
    WhittakerSmoothLambda,
    WhittakerSmoothMethods,
)
from chemotools.utils.check_inputs import check_input, check_weights

### Main Class ###


class WhittakerSmooth(
    OneToOneFeatureMixin,
    BaseEstimator,
    TransformerMixin,
    WhittakerLikeSolver,
):
    """
    A transformer that performs smoothing on data according to the Whittaker-Henderson
    formulation of Penalized Least Squares.

    Parameters
    ----------
    lam : float or int or (float or int, float or int, {"fixed", "logml"} or WhittakerSmoothMethods) or WhittakerSmoothLambda, default=1e2
        The lambda parameter, a.k.a. the penalty weight, for the Whittaker smooth. In
        general, higher values lead to smoother results, but changes take effect in a
        logarithmic rather than linear manner.
        It may thus not be zero or negative (``< 1e-25``). Also high values combined
        with high ``differences`` will lead to numerical instability.
        Please refer to the Notes section for further details.

    differences : int, default=1
        The number of differences to use for the Whittaker smooth. If the aim is to
        obtain a smooth estimate of the ``m``-th order derivative, this should be set to
        at least ``m + 2``.
        Currently, values ``>= 3`` are highly discouraged due to numerical instability
        that might obscure the smoothing effect.

    Attributes
    ----------
    n_features_in_ : int
        The number of features in the input data.

    _is_fitted : bool
        Whether the transformer has been fitted to data.

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the input data.

    transform(X, y=None, sample_weight=None)
        Transform the input data by calculating the (weighted) Whittaker smooth.

    fit_transform(X, y=None, sample_weight=None)
        Fit the transformer to the input data and transform it.

    References
    ----------
    It's based on the algorithms described in [1]_ and [2]_ where an implementational
    adaption of [2]_ was required to make it numerically stable ([3]_).

    .. [1] Z.-M. Zhang, S. Chen, and Y.-Z. Liang, "Baseline correction using adaptive
       iteratively reweighted penalized least squares", Analyst 135 (5), 1138-1146
       (2010)
    .. [2] G. Biessy, "Revisiting Whittaker-Henderson smoothing", arXiv:2306.06932
       (2023)
    .. [3] https://math.stackexchange.com/q/4819039/1261538

    Notes
    -----
    For a more convenient usage of the following, it is recommended to import
    ``WhittakerSmoothLambda`` and ``WhittakerSmoothMethods`` from ``chemotools.smooth``.

    The specification of ``lam`` controls how the lambda parameter is applied/chosen.
    It may not be zero or negative (``< 1e-25``), but aside from that, it can be
    specified in three different ways:

    - a scalar: A fixed lambda is used for all signals, which is a good starting
        point. However, it is important to notice that even similar signals might
        require quite different lambdas.

        ```python
        # fixed lambda of 100
        smoother = WhittakerSmooth(lam=1e2)
        ```

        Internally, it is represented by the method ``"fixed"`` or ``WhittakerSmoothMethods.FIXED``,
        but this never has to be specified explicitly.

    - a tuple of two scalars and a string: The scalars serve as the lower and upper
        bound for searching a lambda according to the method provided by the
        string.
        Each signal will then have its own optimized lambda.
        Currently available methods for automated selection are:

        - ``logml`` or ``WhittakerSmoothMethods.LOGML``: The lambda is chosen by
            maximizing the log marginal likelihood similar to the optimization used
            by the ``sklearn.gaussian_process.GaussianProcessRegressor``.
            It can only be used when ``sample_weight`` can be provided for the methods
            :meth:`transform` and :meth:`fit_transform`.

        ```python
        # will search the optimized lambda for each signal between 1e-5 and 1e10
        smoother = WhittakerSmooth(lam=(1e-5, 1e10, "logml"))

        # which is equivalent to
        smoother = WhittakerSmooth(lam=(1e-5, 1e10, WhittakerSmoothMethods.LOGML))
        ```

    - a ``WhittakerSmoothLambda`` object: This object serves as a convenient way for
        specifying the ``bounds`` for the search space and the ``method`` for the lambda
        selection. It covers both the fixed lambda and its automated selection.

        ```python
        # 1) fixed lambda of 100
        smoother = WhittakerSmooth(lam=WhittakerSmoothLambda(bounds=1e2))


        # which is equivalent to
        smoother = WhittakerSmooth(lam=WhittakerSmoothLambda(bounds=(1e2, 1e2)))

        # 2) will search the optimized lambda for each signal between 1e-5 and 1e10
        smoother = WhittakerSmooth(
            lam=WhittakerSmoothLambda(
                bounds=(1e-5, 1e10),
                method="logml",
            )
        )

        # which is equivalent to
        smoother = WhittakerSmooth(
            lam=WhittakerSmoothLambda(
                bounds=(1e-5, 1e10),
                method=WhittakerSmoothMethods.LOGML,
            )
        )
        ```

    If bounds are provided by either the tuple or the ``WhittakerSmoothLambda`` object,
    the class will fall back to a fixed lambda in case the bounds are apart by less than
    a factor of ``1e-5``, i.e., ``abs(upper - lower) < 1e-5 * upper``.

    """  # noqa: E501

    def __init__(
        self,
        lam: Union[
            RealNumeric,
            Tuple[
                RealNumeric,
                RealNumeric,
                Union[Literal["fixed", "logml"], WhittakerSmoothMethods],
            ],
            WhittakerSmoothLambda,
        ] = 1e2,
        differences: int = 1,
    ):
        self.lam: Union[
            RealNumeric,
            Tuple[
                RealNumeric,
                RealNumeric,
                Union[Literal["fixed", "logml"], WhittakerSmoothMethods],
            ],
            WhittakerSmoothLambda,
        ] = lam
        self.differences: int = differences

    def fit(self, X: ndarray, y: None = None) -> "WhittakerSmooth":
        """
        Fit the transformer to the input data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data to fit the transformer to. It is internally promoted to
            ``np.float64`` to avoid loss of precision.

        y : None
            Ignored.

        Returns
        -------
        self : WhittakerSmooth
            The fitted transformer.

        """
        # Check that X is a 2D array and has only finite values
        X = check_input(
            X,
            dtype=WhittakerLikeSolver._WhittakerLikeSolver__dtype,  # type: ignore
        )

        # Set the number of features ...
        self.n_features_in_ = X.shape[1]
        # ... and all the required attributes for fitting
        self._setup_for_fit(
            n_data=self.n_features_in_,
            lam=self.lam,
            differences=self.differences,
            child_class_name=self.__class__.__name__,
        )

        # Set the fitted attribute to True
        self._is_fitted = True

        return self

    def transform(
        self,
        X: ndarray,
        y: None = None,
        sample_weight: Optional[ndarray] = None,
    ) -> ndarray:
        """
        Transform the input data by calculating the Whittaker smooth.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data to transform. It is internally promoted to ``np.float64`` to
            avoid loss of precision.

        y : None
            Ignored.

        sample_weight : ndarray of shape (n_features,), (n_samples, n_features), (1, n_features), or None, default=None
            Individual weights for each of the input data. If only 1 weight vector is
            provided, it is assumed to be the same for the features all samples.
            If ``None``, all features are assumed to have the same weight.

        Returns
        -------
        X_smoothed : ndarray of shape (n_samples, n_features)
            The transformed data.

        """  # noqa: E501

        # Check that the estimator is fitted
        check_is_fitted(self, "_is_fitted")

        # Check that X is a 2D array and has only finite values
        X = check_input(
            X,
            dtype=WhittakerLikeSolver._WhittakerLikeSolver__dtype,  # type: ignore
        )
        X_ = X.copy()

        # Check that the number of features is the same as the fitted data
        if X_.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features but got {X_.shape[1]}"
            )

        # Check the weights
        sample_weight_checked, use_same_w_for_all = check_weights(
            weights=sample_weight, n_samples=X_.shape[0], n_features=X_.shape[1]
        )

        # Calculate the whittaker smooth
        return self._whittaker_solve(
            X=X_, weights=sample_weight_checked, use_same_w_for_all=use_same_w_for_all
        )[0]

    def fit_transform(
        self,
        X: ndarray,
        y: None = None,
        sample_weight: Optional[ndarray] = None,
    ) -> ndarray:
        """Fit the transformer to the input data and transform it.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data to fit and transform. It is internally promoted to
            ``np.float64`` to avoid loss of precision.

        y : None
            Ignored.

        sample_weight : ndarray of shape (n_features,), (n_samples, n_features), (1, n_features), or None, default=None
            Individual weights for each of the input data. If only 1 weight vector is
            provided, it is assumed to be the same for the features all samples.
            No weights may be negative (< 0.0) and at least one weight needs to be
            positive (> 0.0).
            If ``None``, all features are assumed to have the same weight.

        Returns
        -------
        X_smoothed : ndarray of shape (n_samples, n_features)
            The transformed data.

        """  # noqa: E501

        return self.fit(X=X).transform(X=X, sample_weight=sample_weight)

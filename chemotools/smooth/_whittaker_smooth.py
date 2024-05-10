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

from numpy import ndarray
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from chemotools.utils.check_inputs import check_input, check_weights
from chemotools.utils.whittaker_base import WhittakerLikeSolver


class WhittakerSmooth(
    OneToOneFeatureMixin, BaseEstimator, TransformerMixin, WhittakerLikeSolver
):
    """
    A transformer that performs smoothing on data according to the Whittaker-Henderson
    formulation of Penalized Least Squares.

    Parameters
    ----------
    lam : float or int, default=1e2
        The lambda parameter to use for the Whittaker smooth.

    differences : int, default=1
        The number of differences to use for the Whittaker smooth. If the aim is to
        obtain a smooth estimate of the ``m``-th order derivative, this should be set to
        at least ``m + 2``.
        Currently, values >= 6 are highly discouraged and might lead to obscured
        smoothing.

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

    """

    def __init__(
        self,
        lam: int | float = 1e2,
        differences: int = 1,
    ):
        self.lam = lam
        self.differences = differences

    def fit(self, X: ndarray, y=None) -> "WhittakerSmooth":
        """
        Fit the transformer to the input data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
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
        )

        # Set the fitted attribute to True
        self._is_fitted = True

        return self

    def transform(
        self,
        X: ndarray,
        y: None = None,
        sample_weight: ndarray | None = None,
    ) -> ndarray:
        """
        Transform the input data by calculating the Whittaker smooth.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to transform. It is internally promoted to ``np.float64`` to
            avoid loss of precision.

        y : None
            Ignored.

        sample_weight : np.ndarray of shape (n_features,), (n_samples, n_features), (1, n_features), or None, default=None
            Individual weights for each of the input data. If only 1 weight vector is
            provided, it is assumed to be the same for the features all samples.
            If ``None``, all features are assumed to have the same weight.

        Returns
        -------
        X_smoothed : np.ndarray of shape (n_samples, n_features)
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
            X=X_, w=sample_weight_checked, use_same_w_for_all=use_same_w_for_all
        )[0]

    def fit_transform(
        self, X: ndarray, y: None = None, sample_weight: ndarray | None = None
    ) -> ndarray:
        """Fit the transformer to the input data and transform it.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to fit and transform. It is internally promoted to
            ``np.float64`` to avoid loss of precision.

        y : None
            Ignored.

        sample_weight : np.ndarray of shape (n_features,), (n_samples, n_features), (1, n_features), or None, default=None
            Individual weights for each of the input data. If only 1 weight vector is
            provided, it is assumed to be the same for the features all samples.
            No weights may be negative (< 0.0) and at least one weight needs to be
            positive (> 0.0).
            If ``None``, all features are assumed to have the same weight.

        Returns
        -------
        X_smoothed : np.ndarray of shape (n_samples, n_features)
            The transformed data.

        """  # noqa: E501

        return self.fit(X=X).transform(X=X, sample_weight=sample_weight)

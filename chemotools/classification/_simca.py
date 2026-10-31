import numpy as np
from numpy.typing import NDArray
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
    clone,
    is_outlier_detector,
)
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted, validate_data


class SIMCA(ClassifierMixin, MetaEstimatorMixin, BaseEstimator):
    """Soft Independent Modeling of Class Analogy classifier."""

    def __sklearn_tags__(self):
        """Return the scikit-learn estimator tags for this classifier."""
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        tags.target_tags.required = True
        tags.target_tags.multi_output = True
        return tags

    def __init__(self, estimator: BaseEstimator):
        self.estimator = estimator

    def fit(self, X: NDArray, y: NDArray) -> "SIMCA":
        """Fit one class model per column of the binary target matrix."""
        # 1. Validation
        # Require an outlier detector that can predict new observations.
        if not is_outlier_detector(self.estimator) or not callable(
            getattr(self.estimator, "predict", None)
        ):
            raise TypeError(
                "estimator must be an outlier detector supporting predict(X)."
            )

        # Allow y to be multioutput
        X, y = validate_data(self, X, y, multi_output=True)

        # Require a dense indicator matrix containing only zeros and ones.
        if not isinstance(y, np.ndarray) or y.ndim != 2 or not np.isin(y, [0, 1]).all():
            raise ValueError("y must be a dense 2D binary indicator matrix.")

        if y.shape[1] == 0 or not np.any(y == 1, axis=0).all():
            raise ValueError("Every class must have at least one training sample.")

        self.classes_ = np.arange(y.shape[1])

        # 2. Clone and estimate
        self.estimators_ = []
        for k in self.classes_:
            mask = y[:, k] == 1
            X_class = X[mask, :]
            estimator = clone(self.estimator)
            estimator.fit(X_class)
            self.estimators_.append(estimator)
        return self

    def predict(self, X: NDArray) -> NDArray:
        """Return a binary membership matrix."""
        check_is_fitted(self, "estimators_")
        X = validate_data(self, X, reset=False)

        return np.column_stack(
            [estimator.predict(X) == 1 for estimator in self.estimators_]
        ).astype(int)

    @available_if(lambda self: hasattr(self.estimator, "decision_function"))
    def decision_function(self, X: NDArray) -> NDArray:
        """Returns per-class decision values.
        
        Returns
        -------
        decisions : ndarray of shape (n_samples, n_classes)
            Decision values, with columns ordered as ``classes_``
        """
        check_is_fitted(self, "estimators_")
        X = validate_data(self, X, reset=False)

        return np.column_stack(
            [estimator.decision_function(X) for estimator in self.estimators_]
        ).astype(int)

    @available_if(lambda self: hasattr(self.estimator, "decision_function"))
    def score_samples(self, X: NDArray) -> NDArray:
        """Returns per-sample scores, shape (n_samples, n_classes).
        
        Returns
        -------
        decisions : ndarray of shape (n_samples, n_classes)
            Decision values, with columns ordered as ``classes_``
        """
        check_is_fitted(self, "estimators_")
        X = validate_data(self, X, reset=False)

        return np.column_stack(
            [estimator.score_samples(X) for estimator in self.estimators_]
        ).astype(int)


    
"""Soft Independent Modeling of Class Analogy (SIMCA) classification."""

# Author: Pau Cabaneros & Ruggero Guerrini
# License: MIT

from numbers import Integral

import numpy as np
from numpy.typing import NDArray
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
    clone,
    is_outlier_detector,
)
from sklearn.utils._param_validation import HasMethods
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import check_is_fitted, validate_data


class SIMCA(ClassifierMixin, MetaEstimatorMixin, BaseEstimator):
    """Generalized Soft Independent Modeling of Class Analogy (SIMCA) classifier.

    Fit one independent model per class. Samples may belong to multiple
    classes or be rejected by all models.

    Parameters
    ----------
    estimator : estimator object
        Scikit-learn-compatible outlier detector with ``fit(X)`` and
        ``predict(X)`` for new observations. Predictions of 1 indicate
        membership. Cloned per class; the supplied estimator is unchanged.

    class_names : list of str or int or None, default=None
        Unique names in target-column order. None or an empty list uses
        integer class indices. Used for label prediction and model lookup,
        not the binary output of :meth:`predict`.

    n_jobs : int or None, default=None
        Parallel jobs for :meth:`fit`, also used by :meth:`fit_predict_labels`.
        None means 1 unless overridden by :obj:`joblib.parallel_config`.
        Use 1 for serial execution, -1 for all CPUs, or -2 for all but one;
        0 is invalid. Does not affect prediction, scoring, :meth:`add_model`,
        or the underlying estimator's parallelism settings.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Zero-based class indices in target-column order, followed by any
        classes appended with :meth:`add_model`.

    estimators_ : list of estimator objects
        One fitted estimator per class, ordered as ``classes_``.

    class_names_ : ndarray of shape (n_classes,)
        Supplied names or integer indices, ordered as ``classes_``.

    class_count_ : ndarray of shape (n_classes,)
        Training sample counts, ordered as ``classes_``.

    n_classes_ : int
        Number of fitted class models.

    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    feature_names_in_ : ndarray of shape (n_features_in_,)
        Feature names seen during :meth:`fit`, defined only when all names
        are strings.

    Methods
    -------
    fit(X, y)
        Fit one model per target column.

    predict(X)
        Return a binary class-membership matrix.

    predict_labels(X)
        Return accepted class names per sample.

    fit_predict_labels(X, y)
        Fit and predict class names for the training data.

    get_class_model(class_name)
        Retrieve a fitted model by class name.

    add_model(X, class_name=None)
        Append a class model without refitting existing models.

    decision_function(X)
        Return per-class decision values when supported by the estimator.

    score_samples(X)
        Return per-class sample scores when supported by the estimator.

    See Also
    --------
    chemotools.classification.OneClassPCA : PCA-based one-class classifier.
    sklearn.svm.OneClassSVM : SVM-based one-class classifier.

    Notes
    -----
    Targets are dense binary indicators. Each model uses only its class
    members; samples may train multiple models or none.

    :meth:`add_model` appends to the fitted class order without changing
    the ``class_names`` parameter. Calling :meth:`fit` replaces all models.

    Decision values and sample scores retain the underlying estimator's
    meaning and scale; they are not class probabilities.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.svm import OneClassSVM
    >>> from chemotools.classification._simca import SIMCA
    >>> X = np.array([[0.0], [0.1], [0.2], [2.0], [2.1], [2.2]])
    >>> y = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])
    >>> classifier = SIMCA(estimator=OneClassSVM(), class_names=["A", "B"])
    >>> labels = classifier.fit_predict_labels(X, y)
    >>> len(labels)
    6
    >>> classifier.predict(X).shape
    (6, 2)
    >>> classifier.get_class_model("A") is classifier.estimators_[0]
    True
    >>> _ = classifier.add_model(np.array([[4.0], [4.1], [4.2]]), class_name="C")
    >>> classifier.predict(X).shape
    (6, 3)
    """

    # TODO: Add parameter constraints

    _parameter_constraints = {
        "estimator": [HasMethods(["predict"])],
        "class_names": ["array-like", None],
        "n_jobs": [Integral, None],
    }

    def __sklearn_tags__(self):
        """Return scikit-learn estimator tags.

        Returns
        -------
        tags : sklearn.utils.Tags
            Tags declaring support for two-dimensional multilabel targets.
        """
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_label = True
        tags.classifier_tags.multi_class = False
        tags.target_tags.multi_output = True
        tags.target_tags.single_output = False
        tags.target_tags.two_d_labels = True
        return tags

    def __init__(
        self,
        estimator: BaseEstimator,
        class_names: None | list = None,
        *,
        n_jobs: None | int = None,
    ):
        """Initialize SIMCA; see the class docstring for parameters."""
        self.estimator = estimator
        self.class_names = class_names
        self.n_jobs = n_jobs

    def fit(self, X: NDArray, y: NDArray) -> "SIMCA":
        """Fit one class model per target column.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples.

        y : array-like of shape (n_samples, n_classes)
            Dense indicators: 1 for membership, 0 otherwise. Each column
            must contain at least one member.

        Returns
        -------
        self : SIMCA
            Fitted classifier.

        Raises
        ------
        TypeError
            If ``estimator`` is not an outlier detector with a callable
            ``predict`` method.

        ValueError
            If data or parameters are invalid, a class has no members, or
            class names are duplicated or do not match the target columns.
        """
        # 1. Validation
        # Validate parameters
        self._validate_params()

        # Require an outlier detector that can predict new observations.
        if not is_outlier_detector(self.estimator) or not callable(
            getattr(self.estimator, "predict", None)
        ):
            raise TypeError(
                "estimator must be an outlier detector supporting predict(X)."
            )

        # Allow y to be multioutput
        X, y = validate_data(self, X, y, multi_output=True)

        check_classification_targets(y)

        # Require a dense indicator matrix containing only zeros and ones.
        if not isinstance(y, np.ndarray) or y.ndim != 2 or not np.isin(y, [0, 1]).all():
            raise ValueError(
                "y must be a dense 2D binary class-membership indicator matrix."
            )

        if y.shape[1] == 0 or not np.any(y == 1, axis=0).all():
            raise ValueError("Every class must have at least one training sample.")

        self.classes_ = np.arange(y.shape[1])

        # Validate number of classes
        if self.class_names is None or len(self.class_names) == 0:
            self.class_names_ = np.array(self.classes_, dtype=object)
        else:
            self.class_names_ = np.array(self.class_names, dtype=object)

        if self.class_names_.shape != self.classes_.shape:
            raise ValueError("class_names must contain one name per target column.")

        if len(set(self.class_names_)) != len(self.classes_):
            raise ValueError("class_names must be unique")

        # 2. Set attributes
        self.class_count_ = np.count_nonzero(y, axis=0)
        self.n_classes_ = y.shape[1]

        # 3. Clone and estimate
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_class_model)(
                self.estimator,
                X,
                y[:, k] == 1,
            )
            for k in self.classes_
        )

        return self

    def predict(self, X: NDArray) -> NDArray:
        """Predict binary class memberships.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to classify.

        Returns
        -------
        memberships : ndarray of shape (n_samples, n_classes), dtype=int
            Membership indicators in ``classes_`` order: 1 where a class
            model predicts 1, otherwise 0. Multiple or no classes may match.
        """
        check_is_fitted(self, "estimators_")
        X = validate_data(self, X, ensure_2d=True, reset=False)

        return np.column_stack(
            [estimator.predict(X) == 1 for estimator in self.estimators_]
        ).astype(int)

    def predict_labels(self, X: NDArray) -> list:
        """Return accepted class names per sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to classify.

        Returns
        -------
        labels : list of tuple
            One tuple per sample, containing accepted names in
            ``class_names_`` order. Tuples may be empty or contain multiple
            names.

        See Also
        --------
        predict : Return class memberships as a binary indicator matrix.
        """
        masks = np.asarray(self.predict(X), dtype=bool)
        return [tuple(self.class_names_[mask].tolist()) for mask in masks]

    def fit_predict_labels(self, X: NDArray, y: NDArray):
        """Fit and predict class names for the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples used for both fitting and prediction.

        y : array-like of shape (n_samples, n_classes)
            Dense membership indicators with at least one member per class.

        Returns
        -------
        labels : list of tuple
            One tuple of accepted names per sample, as in
            :meth:`predict_labels`. Predictions may differ from ``y``.

        See Also
        --------
        fit : Fit one class model per target column.
        predict_labels : Predict class names using the fitted models.

        Notes
        -----
        Equivalent to ``self.fit(X, y).predict_labels(X)``; replaces existing
        models.
        """
        return self.fit(X, y).predict_labels(X)

    def get_class_model(self, class_name: str | int) -> BaseEstimator:
        """Retrieve a fitted model by class name.

        Parameters
        ----------
        class_name : str or int
            Name in ``class_names_``, or an integer index if names were omitted.

        Returns
        -------
        estimator : estimator object
            First matching stored estimator, not a copy. Changes to it
            affect this classifier.

        Raises
        ------
        NotFittedError
            If the classifier is not fitted.

        ValueError
            If ``class_name`` is not present in ``class_names_``.
        """
        check_is_fitted(self, "estimators_")
        if class_name not in self.class_names_:
            raise ValueError(
                f"Selected class: {class_name} not in model classes. "
                f"Available options are: {self.class_names_}"
            )

        return self.estimators_[np.where(self.class_names_ == class_name)[0][0]]

    def add_model(self, X: NDArray, class_name: None | str = None) -> "SIMCA":
        """Append a class model without refitting existing models.

        Fit a clone of ``estimator`` to all supplied samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New class members, with the same features and order as in
            :meth:`fit`.

        class_name : str or None, default=None
            New class name. None or an empty string uses the next zero-based
            class index. The resolved name must not already exist.

        Returns
        -------
        self : SIMCA
            Classifier updated in place.

        Raises
        ------
        NotFittedError
            If the classifier is not fitted.

        ValueError
            If the resolved name already exists or the training data are
            invalid, including mismatched features.

        See Also
        --------
        fit : Fit all class models from a binary target matrix.
        get_class_model : Retrieve a fitted estimator by class name.

        Notes
        -----
        Appends to the fitted model, index, name, and sample-count collections
        and increments ``n_classes_``. The new class is the last output column.

        """
        check_is_fitted(self, "estimators_")

        # Validate data
        X = validate_data(
            self,
            X,
            y="no_validation",
            ensure_2d=True,
            reset=False,
            dtype=np.float64,
        )

        new_class_name = (
            self.n_classes_ if class_name is None or class_name == "" else class_name
        )

        if new_class_name in self.class_names_:
            raise ValueError(f"Class name {new_class_name!r} already exists")

        # Clone and fit a new estimator
        estimator = clone(self.estimator)
        estimator.fit(X)

        # Update attributes
        self.estimators_.append(estimator)
        self.classes_ = np.append(self.classes_, len(self.classes_))
        self.class_names_ = np.append(self.class_names_, new_class_name)
        self.class_count_ = np.append(self.class_count_, X.shape[0])
        self.n_classes_ += 1

        return self

    @available_if(lambda self: hasattr(self.estimator, "decision_function"))
    def decision_function(self, X: NDArray) -> NDArray:
        """Compute per-class decision values.

        Available only when ``estimator`` exposes ``decision_function``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to evaluate.

        Returns
        -------
        decisions : ndarray of shape (n_samples, n_classes)
            Values in ``classes_`` order, using the underlying estimator's
            scale and threshold. Not class probabilities.
        """
        check_is_fitted(self, "estimators_")
        X = validate_data(self, X, reset=False)

        return np.column_stack(
            [estimator.decision_function(X) for estimator in self.estimators_]
        )

    @available_if(lambda self: hasattr(self.estimator, "score_samples"))
    def score_samples(self, X: NDArray) -> NDArray:
        """Compute per-class sample scores.

        Available only when ``estimator`` exposes ``score_samples``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to score.

        Returns
        -------
        scores : ndarray of shape (n_samples, n_classes)
            Scores in ``classes_`` order, on the underlying estimator's scale.
            Not probabilities or necessarily comparable across classes.
        """
        check_is_fitted(self, "estimators_")
        X = validate_data(self, X, reset=False)

        return np.column_stack(
            [estimator.score_samples(X) for estimator in self.estimators_]
        )

    def _fit_class_model(self, estimator, X, mask):
        """Clone and fit an estimator to the samples selected by the mask."""
        return clone(estimator).fit(X[mask, :])

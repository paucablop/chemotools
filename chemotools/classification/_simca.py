"""Soft Independent Modeling of Class Analogy (SIMCA) classification.

The module :mod:`chemotools.classification._simca` implements a classifier
that fits an independent one-class model for each target class.
"""

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
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import check_is_fitted, validate_data


class SIMCA(ClassifierMixin, MetaEstimatorMixin, BaseEstimator):
    """Soft Independent Modeling of Class Analogy (SIMCA) classifier.

    Fit a separate clone of a one-class estimator to the training samples
    belonging to each class. Class memberships are independent, so a sample
    can be accepted by multiple classes or rejected by all classes.
    Additional class models can be fitted with :meth:`add_model` without
    refitting the existing models.

    Parameters
    ----------
    estimator : estimator object
        Scikit-learn-compatible outlier detector implementing ``fit(X)`` and
        ``predict(X)``. Predictions equal to 1 indicate class membership.
        The estimator must support prediction on new observations. A separate
        clone is fitted for each class; the supplied estimator is not fitted
        or modified.

    class_names : list of str or int or None, default=None
        Names of the classes, with one entry per column of the target matrix
        in the same order. If None or an empty list, the integer indices in
        ``classes_`` are used. Names are used by :meth:`predict_labels` and
        :meth:`get_class_model`; they do not change the binary membership matrix
        returned by :meth:`predict`.

    n_jobs : int or None, default=None
        Number of parallel jobs used to fit the independent class models
        during :meth:`fit`, including calls through :meth:`fit_predict_labels`.
        None means 1 unless a :obj:`joblib.parallel_config` context specifies
        otherwise. Use 1 for serial execution and -1 to use all available CPUs.
        Other negative values use fewer CPUs; for example, -2 uses all but one.
        A value of 0 is invalid.

        This parameter does not parallelize prediction, scoring, or
        :meth:`add_model`, and does not change parallelism settings of the
        underlying estimator.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class indices, from 0 to ``n_classes - 1``, corresponding to the
        columns of the binary target matrix passed to :meth:`fit`, followed
        by any classes appended with :meth:`add_model`.

    estimators_ : list of length (n_classes)
        Fitted one-class estimators, ordered as ``classes_``. Stored as a list
        after :meth:`fit` and an array after :meth:`add_model`.

    class_names_ : ndarray of shape (n_classes,)
        Class names in target-column order. Contains the supplied names
        converted to an array, or ``classes_`` when no names are supplied.
        Names for additional models are appended by :meth:`add_model`.

    class_count_ : ndarray of shape (n_classes,)
        Number of training samples used to fit each class model, ordered as
        ``classes_``.

    n_classes_ : int
        Number of fitted class models, including any added with
        :meth:`add_model`.

    n_features_in_ : int
        Number of features seen during the most recent :meth:`fit` or
        :meth:`add_model` call.

    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during the most recent :meth:`fit` or
        :meth:`add_model` call. Defined only when the input has feature names
        that are all strings.

    Methods
    -------
    fit(X, y)
        Fit an independent one-class estimator for each target column.

    predict(X)
        Return a binary class-membership matrix.

    predict_labels(X)
        Return a tuple of accepted class names for each sample.

    fit_predict_labels(X, y)
        Fit the classifier and return class-name tuples for the training data.

    get_class_model(class_name)
        Return the fitted estimator associated with a class name.

    add_model(X, class_name=None)
        Fit and append a new class model without refitting existing models.

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
    The target must be a dense binary indicator matrix, not a vector of
    class labels. Each column defines a class and must contain at least one
    positive training sample. A sample with multiple positive entries is used
    to fit each corresponding class model; a sample with no positive entries
    is not used to fit any class model.

    The :meth:`decision_function` and :meth:`score_samples` methods are
    available only when the supplied estimator exposes the corresponding
    method. Their outputs retain the scale and meaning of the underlying
    estimator; they are not converted to class probabilities.

    Use :meth:`add_model` to extend a fitted classifier with samples from a
    single new class. A supplied class name already present in
    ``class_names_`` is rejected. The new model is appended to the class
    order used by prediction and scoring methods. This updates fitted
    attributes only; the constructor parameter ``class_names`` is unchanged.
    Calling :meth:`fit` again replaces all fitted models, including added models.

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
    >>> classifier.add_model(np.array([[4.0], [4.1], [4.2]]), class_name="C")
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
        """Return the scikit-learn estimator tags for this classifier.

        Returns
        -------
        tags : sklearn.utils.Tags
            Estimator tags declaring support for two-dimensional multilabel
            targets rather than single-output multiclass targets.
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
        """Initialize the classifier with a one-class estimator.

        Parameters
        ----------
        estimator : estimator object
            Scikit-learn-compatible outlier detector supporting ``fit(X)``
            and ``predict(X)`` on new observations. Cloned for each class
            during :meth:`fit`.

        class_names : list of str or int or None, default=None
            Names of the classes, with one entry per target column in the
            same order. If None or an empty list, integer class indices are
            used.

        n_jobs : int or None, default=None
            Number of parallel jobs used to fit the independent class models.
            None means 1 unless a :obj:`joblib.parallel_config` context
            specifies otherwise. Use 1 for serial execution and -1 to use all
            available CPUs. Other negative values use fewer CPUs; for example,
            -2 uses all but one. A value of 0 is invalid. Does not parallelize
            prediction, scoring, or :meth:`add_model`, or change parallelism
            settings of the underlying estimator.
        """
        self.estimator = estimator
        self.class_names = class_names
        self.n_jobs = n_jobs

    def fit(self, X: NDArray, y: NDArray) -> "SIMCA":
        """Fit one class model per column of the binary target matrix.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Each class model is fitted only to samples whose
            corresponding target entry is 1.

        y : array-like of shape (n_samples, n_classes)
            Dense binary indicator matrix with 1 for class membership and 0
            otherwise. Each class must have at least one training sample.

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
            If the input data are invalid, the target is not a dense
            two-dimensional binary indicator matrix, or any class has no
            training samples.
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

        # Require a dense indicator matrix containing only zeros and ones.
        if not isinstance(y, np.ndarray) or y.ndim != 2 or not np.isin(y, [0, 1]).all():
            raise ValueError("y must be a dense 2D binary indicator matrix.")

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
        """Predict class memberships for the input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to classify.

        Returns
        -------
        memberships : ndarray of shape (n_samples, n_classes), dtype=int
            Binary membership matrix with columns ordered as ``classes_``.
            An entry is 1 when the corresponding class estimator predicts 1,
            and 0 otherwise. Each sample may belong to zero, one, or multiple
            classes.
        """
        check_is_fitted(self, "estimators_")
        X = validate_data(self, X, ensure_2d=True, reset=False)

        return np.column_stack(
            [estimator.predict(X) == 1 for estimator in self.estimators_]
        ).astype(int)

    def predict_labels(self, X: NDArray) -> list:
        """Predict class names for the input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to classify.

        Returns
        -------
        labels : list of tuple
            List of length ``n_samples``, with one tuple per sample containing
            the entries of ``class_names_``
            for all accepted classes, in target-column order. An empty tuple
            indicates rejection by all class models. A tuple may contain
            multiple names when several class models accept the sample.

        See Also
        --------
        predict : Return class memberships as a binary indicator matrix.
        """
        masks = np.asarray(self.predict(X), dtype=bool)
        return [tuple(self.class_names_[mask].tolist()) for mask in masks]

    def fit_predict_labels(self, X: NDArray, y: NDArray):
        """Fit the classifier and predict class names for the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples, also used to predict class memberships after
            fitting.

        y : array-like of shape (n_samples, n_classes)
            Dense binary indicator matrix with 1 for class membership and 0
            otherwise. Each class must have at least one training sample.

        Returns
        -------
        labels : list of tuple
            List of length ``n_samples`` containing the accepted class names
            for each training sample, in target-column order. Tuples may be
            empty or contain multiple names. Predicted memberships need not
            match the training targets.

        See Also
        --------
        fit : Fit one class model per target column.
        predict_labels : Predict class names using the fitted models.

        Notes
        -----
        Equivalent to ``self.fit(X, y).predict_labels(X)``. Any previously
        fitted class models are replaced.
        """
        return self.fit(X, y).predict_labels(X)

    def get_class_model(self, class_name: str | int) -> BaseEstimator:
        """Return the fitted one-class estimator for a class.

        Parameters
        ----------
        class_name : str or int
            Class name present in ``class_names_``. When no custom names were
            supplied, use the integer class index.

        Returns
        -------
        estimator : estimator object
            Fitted estimator for the first matching entry in ``class_names_``.
            This is the stored estimator, not a copy; modifying it also
            modifies the corresponding class model in this classifier.

        Raises
        ------
        ValueError
            If ``class_name`` is not present in ``class_names_``.

        Notes
        -----
        The classifier must be fitted before calling this method.
        """
        check_is_fitted(self, "estimators_")
        if class_name not in self.class_names_:
            raise ValueError(
                f"Selected class: {class_name} not in model classes. "
                f"Available options are: {self.class_names_}"
            )

        return self.estimators_[np.where(self.class_names_ == class_name)[0][0]]

    def add_model(self, X: NDArray, class_name: None | str = None) -> "SIMCA":
        """Fit and append a one-class model to the fitted classifier.

        All input samples are treated as members of the new class. A clone
        of ``estimator`` is fitted to these samples; existing class models
        are not refitted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples for the new class. Provide the same features
            in the same order as for the existing class models.

        class_name : str or None, default=None
            Name to append to ``class_names_``. The supplied value must not
            already be present in ``class_names_``. If None or an empty
            string, the number of classes after adding the model is used
            instead. This automatically assigned name is one greater than
            the new zero-based index in ``classes_``.

        Returns
        -------
        None
            The classifier is updated in place.

        Raises
        ------
        NotFittedError
            If the classifier has not been fitted.

        ValueError
            If the supplied ``class_name`` is already present in
            ``class_names_``, or if the training data fail input validation
            or are rejected by the underlying estimator.

        See Also
        --------
        fit : Fit all class models from a binary target matrix.
        get_class_model : Retrieve a fitted estimator by class name.

        Notes
        -----
        Appends to ``estimators_``, ``classes_``, ``class_names_``, and
        ``class_count_``, and increments ``n_classes_``. The new class is the
        last column in subsequent prediction and scoring outputs.

        The supplied ``class_name`` is checked for duplicates before a
        default name is assigned. Automatically assigned names are not
        checked for collisions with existing class names.

        Input validation resets feature metadata rather than checking it
        against the existing models.
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

        new_class_name = self.n_classes_ if class_name is None else class_name

        if new_class_name in self.class_names_:
            raise ValueError(f"Class name {new_class_name!r} already exists")

        # Clone and fit a new estimator
        estimator = clone(self.estimator)
        estimator.fit(X)

        # Update attributes
        self.estimators_.append(estimator)
        self.classes_ = np.append(self.classes_, len(self.classes_))
        self.class_names_ = np.append(
            self.class_names_, class_name if class_name else len(self.class_names_)
        )
        self.class_count_ = np.append(self.class_count_, X.shape[0])
        self.n_classes_ += 1

        return self

    @available_if(lambda self: hasattr(self.estimator, "decision_function"))
    def decision_function(self, X: NDArray) -> NDArray:
        """Compute per-class decision values for the input samples.

        Available only when ``estimator`` exposes ``decision_function``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to evaluate.

        Returns
        -------
        decisions : ndarray of shape (n_samples, n_classes)
            Decision values returned by each fitted class estimator, with
            columns ordered as ``classes_``. The sign convention, scale, and
            acceptance threshold are determined by the underlying estimator.
            Values are not class probabilities.
        """
        check_is_fitted(self, "estimators_")
        X = validate_data(self, X, reset=False)

        return np.column_stack(
            [estimator.decision_function(X) for estimator in self.estimators_]
        )

    @available_if(lambda self: hasattr(self.estimator, "score_samples"))
    def score_samples(self, X: NDArray) -> NDArray:
        """Compute per-class sample scores for the input samples.

        Available only when ``estimator`` exposes ``score_samples``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to score.

        Returns
        -------
        scores : ndarray of shape (n_samples, n_classes)
            Sample scores returned by each fitted class estimator, with
            columns ordered as ``classes_``. Scores retain the meaning and
            scale of the underlying estimator and are not class probabilities
            or necessarily comparable across classes.
        """
        check_is_fitted(self, "estimators_")
        X = validate_data(self, X, reset=False)

        return np.column_stack(
            [estimator.score_samples(X) for estimator in self.estimators_]
        )

    def _fit_class_model(self, estimator, X, mask):
        return clone(estimator).fit(X[mask, :])

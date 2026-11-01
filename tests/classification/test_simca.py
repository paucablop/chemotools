"""Tests for SIMCA classifier."""

import pytest
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.utils.estimator_checks import check_estimator

from chemotools.classification import SIMCA, OneClassPCA


def expected_failed_checks(estimator):
    return {
        "check_classifiers_train": (
            "Assumes 1D predictions; SIMCA returns a 2D membership matrix."
        ),
        "check_estimators_dtypes": (
            "The check supplies non-binary targets; SIMCA requires indicators."
        ),
        "check_classifier_data_not_an_array": (
            "The check supplies non-binary targets; SIMCA requires indicators."
        ),
        "check_classifiers_classes": (
            "The check assumes single-label targets and predictions.",
        ),
        "check_fit2d_1feature": (
            "The check supplies non-binary targets; SIMCA requires indicators."
        ),
        "check_classifier_not_supporting_multiclass": (
            "Requires a binary-only error message; SIMCA instead requires "
            "a dense 2D multilabel indicator matrix."
        ),
    }


# @parametrize_with_checks(
#    [
#        SIMCA(estimator=OneClassPCA(n_components=1)),
#    ],
#    expected_failed_checks=expected_failed_checks,
#    xfail_strict = True,
# )
# def test_sklearn_compatibility(estimator, check):
#    check(estimator)


@pytest.mark.parametrize(
    "estimator",
    [
        OneClassPCA(n_components=1),
        OneClassSVM(),
        SGDOneClassSVM(),
        IsolationForest(),
        LocalOutlierFactor(novelty=True),
        EllipticEnvelope(),
    ],
)
def test_sklearn_compatibility(estimator):
    estimator = SIMCA(estimator=estimator)

    check_estimator(
        estimator,
        expected_failed_checks=expected_failed_checks(estimator),
    )

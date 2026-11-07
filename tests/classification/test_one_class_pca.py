"""Tests for OneClassPCA classifier."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator

from chemotools.classification import OneClassPCA


class TestSklearnCompatibility:
    @pytest.mark.parametrize(
        "estimator",
        [
            OneClassPCA(n_components=1, method="robust-numeric"),
            OneClassPCA(n_components=1, method="robust-approx"),
            OneClassPCA(n_components=1, method="classic"),
        ],
    )
    def test_sklearn_compatibility(self, estimator):
        """Verify compatibility with scikit-learn estimator conventions."""
        check_estimator(estimator)


@pytest.mark.parametrize("method", ["robust-numeric", "robust-approx", "classic"])
@pytest.mark.parametrize("boundary", ["triangle", "rectangle"])
def test_fit_and_predict(method, boundary):
    """Verify predictions and training categories across methods and boundaries."""
    # Arrange
    X = np.random.default_rng(0).normal(size=(60, 5))
    X_distant = np.full((1, 5), 1_000)
    clf = OneClassPCA(n_components=2, method=method, boundary=boundary)

    # Act
    clf.fit(X)
    predictions = clf.predict(X)
    decisions = clf.decision_function(X)
    distant_prediction = clf.predict(X_distant)

    # Assert
    assert predictions.shape == (60,)
    assert set(np.unique(predictions)) <= {-1, 1}
    np.testing.assert_array_equal(predictions, np.where(decisions < 0, -1, 1))
    np.testing.assert_array_equal(
        np.sum(
            [
                clf.training_regular_,
                clf.training_extremes_,
                clf.training_outliers_,
            ],
            axis=0,
        ),
        np.ones(60, dtype=int),
    )

    if boundary == "triangle":
        assert np.isscalar(clf.thresholds_)
    else:
        assert len(clf.thresholds_) == 2

    assert distant_prediction.item() == -1


@pytest.mark.parametrize(
    "attribute",
    ["components_", "explained_variance_", "explained_variance_ratio_", "mean_"],
)
def test_pca_properties(attribute):
    """Verify PCA properties require fitting and expose the underlying arrays."""
    # Arrange
    clf = OneClassPCA(method="classic")
    X = np.random.default_rng(0).normal(size=(60, 5))

    # Act / Assert
    with pytest.raises(NotFittedError):
        getattr(clf, attribute)

    # Act
    clf.fit(X)
    result = getattr(clf, attribute)

    # Assert
    assert result is getattr(clf.pca_, attribute)


@pytest.mark.parametrize("case", ["constant", "degenerate", "outlier_probability"])
def test_invalid_fit(case):
    """Reject degenerate training data and incompatible outlier probabilities."""
    # Arrange
    X = np.random.default_rng(0).normal(size=(60, 5))
    clf = OneClassPCA(method="classic")
    if case == "constant":
        X = np.ones((60, 5))
        message = "Retained singular values"
    elif case == "degenerate":
        X[:, 1:] *= 1e-20
        message = "numerically degenerate"
    else:
        clf.set_params(alpha=1e-6, gamma=0.9)
        message = "per-sample outlier probability smaller than alpha"

    # Act / Assert
    with pytest.raises(ValueError, match=message):
        clf.fit(X)


@pytest.mark.parametrize(
    "median, iqr",
    [(0, 1), (1, 0), (-1, 1), (1, -1), (np.inf, 1), (1, np.nan)],
)
def test_robust_fast_invalid_statistics(median, iqr):
    """Reject nonpositive or nonfinite robust-fast summary statistics."""
    # Arrange
    clf = OneClassPCA(method="robust-approx")

    # Act / Assert
    with pytest.raises(ValueError, match="finite and strictly positive"):
        clf._estimate_robust_fast_dof(median, iqr)


def test_robust_fast_nonfinite_estimate():
    """Reject a robust-fast degrees-of-freedom estimate that overflows."""
    # Arrange
    clf = OneClassPCA(method="robust-approx")

    # Act / Assert
    with pytest.warns(RuntimeWarning, match="overflow"):
        with pytest.raises(ValueError, match="DoF estimate is not finite"):
            clf._estimate_robust_fast_dof(1e300, 1e-300)


def test_robust_fast_high_dof_warning():
    """Warn but retain estimates outside the robust-fast accuracy range."""
    # Arrange
    clf = OneClassPCA(method="robust-approx")

    # Act
    with pytest.warns(UserWarning, match="outside its reported N < 50 range"):
        dof = clf._estimate_robust_fast_dof(1, 0.1)

    # Assert
    assert dof >= 50
    assert isinstance(dof, int)

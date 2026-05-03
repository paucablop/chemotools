import numpy as np
import pytest
from sklearn.exceptions import InvalidParameterError
from sklearn.utils.estimator_checks import check_estimator

from chemotools.adaptation import XAxisInterpolator


# Test compliance with scikit-learn
def test_compliance_x_axis_interpolator():
    """Verifies sklearn API checks with expected metadata-related exceptions."""
    # Arrange
    transformer = XAxisInterpolator(common_x_axis=np.linspace(0, 1, 5))

    # Act & Assert
    check_estimator(
        transformer,
        expected_failed_checks={
            "check_dict_unchanged": "transform requires x_axis metadata",
            "check_dtype_object": "transform requires x_axis metadata",
            "check_estimators_dtypes": "transform requires x_axis metadata",
            "check_estimators_pickle": "transform requires x_axis metadata",
            "check_f_contiguous_array_estimator": (
                "transform requires x_axis metadata"
            ),
            "check_fit_idempotent": "transform requires x_axis metadata",
            "check_fit_score_takes_y": "transform requires x_axis metadata",
            "check_methods_sample_order_invariance": (
                "transform requires x_axis metadata"
            ),
            "check_methods_subset_invariance": "transform requires x_axis metadata",
            "check_pipeline_consistency": "transform requires x_axis metadata",
            "check_transformer_data_not_an_array": (
                "transform requires x_axis metadata"
            ),
            "check_transformer_general": "transform requires x_axis metadata",
            "check_transformer_preserve_dtypes": ("transform requires x_axis metadata"),
        },
    )


# Test functionality
def test_transform_output_matches_target_grid_size():
    """Confirms interpolation output columns match the configured target axis size."""
    # Arrange
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x_axis = np.array([0.0, 2.0, 4.0])
    target = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    # Act
    est = XAxisInterpolator(common_x_axis=target, method="linear")
    out = est.fit(X).transform(X, x_axis=x_axis)

    # Assert
    assert out.shape == (2, 5)


def test_transform_rejects_non_increasing_x_axis():
    """Ensures a non-monotonic source axis raises a clear validation error."""
    # Arrange
    X = np.array([[1.0, 2.0, 3.0]])
    x_axis = np.array([0.0, 2.0, 2.0])

    est = XAxisInterpolator(common_x_axis=np.array([0.0, 1.0, 2.0]), method="linear")
    est.fit(X)

    # Act & Assert
    with pytest.raises(ValueError, match="strictly increasing"):
        est.transform(X, x_axis=x_axis)


@pytest.mark.parametrize("method", ["linear", "cubic", "pchip"])
def test_none_left_right_use_row_endpoints(method):
    """Checks out-of-range samples use row endpoints when fill values are omitted."""
    # Arrange
    X = np.array([[10.0, 20.0, 40.0]])
    x_axis = np.array([0.0, 1.0, 2.0])
    target = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])

    est = XAxisInterpolator(
        common_x_axis=target,
        method=method,
        left=None,
        right=None,
    )
    est.fit(X)

    # Act
    out = est.transform(X, x_axis=x_axis)

    # Assert
    # Out-of-domain values should follow endpoint semantics when left/right=None.
    assert out[0, 0] == pytest.approx(X[0, 0])
    assert out[0, -1] == pytest.approx(X[0, -1])


def test_fit_rejects_non_finite_common_axis():
    """Validates that non-finite points in the target axis are rejected during fit."""
    # Arrange
    X = np.array([[1.0, 2.0, 3.0]])
    est = XAxisInterpolator(common_x_axis=np.array([0.0, np.nan, 2.0]))

    # Act & Assert
    with pytest.raises(ValueError, match="finite"):
        est.fit(X)


def test_rejects_invalid_methods():
    """Validates that non-finite points in the target axis are rejected during fit."""
    # Arrange
    X = np.array([[1.0, 2.0, 3.0]])
    est = XAxisInterpolator(common_x_axis=np.array([0.0, 1.0, 2.0]), method="banana")

    # Act & Assert
    with pytest.raises(
        InvalidParameterError,
        match="The 'method' parameter of XAxisInterpolator must be a str among "
        "{'linear', 'cubic', 'pchip'}. Got 'banana' instead.",
    ):
        est.fit(X)

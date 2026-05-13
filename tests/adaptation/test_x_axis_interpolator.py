import numpy as np
import pytest
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


class TestEstimator:
    def test_transform_output_matches_target_grid_size(self):
        """
        Confirms interpolation output columns match the configured target axis size.
        """
        # Arrange
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        x_axis = np.array([0.0, 2.0, 4.0])
        target = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        # Act
        est = XAxisInterpolator(common_x_axis=target, method="linear")
        out = est.fit(X).transform(X, x_axis=x_axis)

        # Assert
        assert out.shape == (2, 5)

    @pytest.mark.parametrize("method", ["linear", "cubic", "pchip"])
    def test_none_left_right_use_row_endpoints(self, method):
        """
        Checks out-of-range samples use row endpoints when fill values are omitted.
        """
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

    def test_get_feature_names_out_uses_common_axis_positions(self):
        """Ensures output feature names map to the validated target axis."""
        # Arrange
        X = np.array([[1.0, 2.0, 3.0]])
        target = np.array([0.0, 0.5, 2.0])
        est = XAxisInterpolator(common_x_axis=target)
        est.fit(X)

        # Act
        names = est.get_feature_names_out()

        # Assert
        np.testing.assert_array_equal(
            names, np.array(["x_0", "x_0.5", "x_2"], dtype=object)
        )


class TestFitExceptions:
    def test_fit_rejects_invalid_methods(self):
        """
        Validates that unsupported interpolation methods are rejected during fit.
        """
        # Arrange
        X = np.array([[1.0, 2.0, 3.0]])
        est = XAxisInterpolator(common_x_axis=np.array([0.0, 1.0, 2.0]), method="apple")

        # Act & Assert
        with pytest.raises(
            ValueError,
            match="The 'method' parameter of XAxisInterpolator must be a str among ",
        ):
            est.fit(X)

    def test_fit_common_x_axis_2_dimensions(self):
        # Arrange
        X = np.array([[1.0, 2.0, 3.0]])
        est = XAxisInterpolator(common_x_axis=np.array([[0.0, 1.0, 2.0]]))

        # Act & Assert
        with pytest.raises(
            ValueError,
            match="common_x_axis must be 1D, got shape",
        ):
            est.fit(X)

    def test_fit_common_x_axis_length(self):
        # Arrange
        X = np.array([[1.0, 2.0, 3.0]])
        est = XAxisInterpolator(common_x_axis=np.array([0.0]))

        # Act & Assert
        with pytest.raises(
            ValueError,
            match="common_x_axis must contain at least 2 points.",
        ):
            est.fit(X)

    def test_fit_rejects_non_finite_common_axis(self):
        """
        Validates that non-finite points in the target axis are rejected during fit.
        """
        # Arrange
        X = np.array([[1.0, 2.0, 3.0]])
        est = XAxisInterpolator(common_x_axis=np.array([0.0, np.nan, 2.0]))

        # Act & Assert
        with pytest.raises(ValueError, match="finite"):
            est.fit(X)

    def test_fit_common_x_axis_must_be_finite(self):
        # Arrange
        X = np.array([[1.0, 2.0, 3.0]])
        est = XAxisInterpolator(common_x_axis=np.array([0.0, np.inf]))

        # Act & Assert
        with pytest.raises(
            ValueError,
            match="common_x_axis must contain only finite values.",
        ):
            est.fit(X)

    def test_fit_rejects_non_increasing_common_x_axis(self):
        """Ensures a non-monotonic source axis raises a clear validation error."""
        # Arrange
        X = np.array([[1.0, 2.0, 3.0]])

        est = XAxisInterpolator(
            common_x_axis=np.array([0.0, 1.0, 0.0]), method="linear"
        )

        # Act & Assert
        with pytest.raises(
            ValueError, match="common_x_axis must be strictly increasing."
        ):
            est.fit(X)


class TestTransformExceptions:
    def test_transform_rejects_non_increasing_x_axis(self):
        """Ensures a non-monotonic source axis raises a clear validation error."""
        # Arrange
        X = np.array([[1.0, 2.0, 3.0]])
        x_axis = np.array([0.0, 2.0, 2.0])

        est = XAxisInterpolator(
            common_x_axis=np.array([0.0, 1.0, 2.0]), method="linear"
        )
        est.fit(X)

        # Act & Assert
        with pytest.raises(ValueError, match="strictly increasing"):
            est.transform(X, x_axis=x_axis)

    def test_transform_rejects_wrong_method(self):
        """Test defensive mechanism in case method changes after fit."""
        # Arrange
        X = np.array([[1.0, 2.0, 3.0]])
        x_axis = np.array([0.0, 2.0, 3.0])

        est = XAxisInterpolator(
            common_x_axis=np.array([0.0, 1.0, 2.0]), method="linear"
        )
        est.fit(X)
        est.method = "banana"

        # Act & Assert
        with pytest.raises(ValueError, match="Unknown interpolation method"):
            est.transform(X, x_axis=x_axis)

    def test_transform_rejects_1d_x_axis_with_wrong_length(self):
        """Validates shape checks for 1D x_axis against the feature count."""
        # Arrange
        X = np.array([[1.0, 2.0, 3.0]])
        x_axis = np.array([0.0, 1.0])
        est = XAxisInterpolator(common_x_axis=np.array([0.0, 1.0, 2.0]))
        est.fit(X)

        # Act & Assert
        with pytest.raises(ValueError, match="expected \(3,\) or \(1, 3\)"):
            est.transform(X, x_axis=x_axis)

    def test_transform_rejects_2d_x_axis_with_wrong_shape(self):
        """Validates shape checks for 2D x_axis against X shape."""
        # Arrange
        X = np.array([[1.0, 2.0, 3.0]])
        x_axis = np.array([[0.0, 1.0], [2.0, 3.0]])
        est = XAxisInterpolator(common_x_axis=np.array([0.0, 1.0, 2.0]))
        est.fit(X)

        # Act & Assert
        with pytest.raises(ValueError, match="x_axis has shape"):
            est.transform(X, x_axis=x_axis)

    def test_transform_rejects_x_axis_with_invalid_dimensions(self):
        """Ensures x_axis dimensionality must be either 1D or 2D."""
        # Arrange
        X = np.array([[1.0, 2.0, 3.0]])
        x_axis = np.array([[[0.0, 1.0, 2.0]]])
        est = XAxisInterpolator(common_x_axis=np.array([0.0, 1.0, 2.0]))
        est.fit(X)

        # Act & Assert
        with pytest.raises(ValueError, match="x_axis must be 1D or 2D"):
            est.transform(X, x_axis=x_axis)

    def test_transform_rejects_non_finite_x_axis(self):
        """Ensures x_axis finite-value validation is enforced."""
        # Arrange
        X = np.array([[1.0, 2.0, 3.0]])
        x_axis = np.array([0.0, np.nan, 2.0])
        est = XAxisInterpolator(common_x_axis=np.array([0.0, 1.0, 2.0]))
        est.fit(X)

        # Act & Assert
        with pytest.raises(ValueError, match="x_axis must contain only finite values"):
            est.transform(X, x_axis=x_axis)

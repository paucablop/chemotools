import pytest

from chemotools.adaptation.functions import scale_by_factor
from chemotools.adaptation.validation import check_metadata_function


class TestCheckMetadataFunction:
    def test_raises_type_error_if_func_is_not_callable(self):
        """Test that a TypeError is raised if the func argument is not callable."""
        with pytest.raises(TypeError):
            check_metadata_function(func=42, X=[[1, 2], [3, 4]])

    def test_raises_type_error_unintrospectable_signature(self):
        """
        Test that a TypeError is raised if the func signature cannot be introspected.
        """

        with pytest.raises(TypeError, match="Could not inspect the signature of"):
            check_metadata_function(func=dict.update, X=[[1, 2], [3, 4]])

    def test_raises_value_error_if_func_signature_is_invalid(self):
        """Test that a ValueError is raised if the func signature is invalid."""

        with pytest.raises(
            ValueError, match="cannot be called as `func\\(X, \\*\\*metadata\\)`"
        ):
            check_metadata_function(func=scale_by_factor, X=[[1, 2], [3, 4]])

    @pytest.mark.parametrize("reserved_name", ["X", "y"])
    def test_raises_value_error_if_metadata_name_is_reserved(self, reserved_name):
        """Test that estimator API argument names cannot be metadata keys."""

        with pytest.raises(
            ValueError,
            match=rf"`{reserved_name}` cannot be requested in `metadata`",
        ):
            check_metadata_function(
                func=lambda data, **metadata: data,
                X=[[1, 2], [3, 4]],
                metadata={reserved_name: object()},
            )

    def test_raises_value_error_if_func_output_is_invalid(self):
        """Test that a ValueError is raised if the func output is invalid."""

        def invalid_func(X):
            return X[0]

        with pytest.raises(ValueError, match="must return a numeric 2-D array"):
            check_metadata_function(func=invalid_func, X=[[1, 2], [3, 4]])

    def test_raises_value_error_if_func_changes_number_of_samples(self):
        """
        Test that a ValueError is raised if the func changes the number of samples.
        """

        def invalid_func(X):
            return X[:-1]

        with pytest.raises(ValueError, match="changed the number of samples"):
            check_metadata_function(func=invalid_func, X=[[1, 2], [3, 4]])

    def test_raises_value_error_if_func_changes_number_of_features(self):
        """
        Test that a ValueError is raised if the func changes the number of features.
        """

        def invalid_func(X):
            return [row[:-1] for row in X]

        with pytest.raises(ValueError, match="changed the number of features"):
            check_metadata_function(func=invalid_func, X=[[1, 2], [3, 4]])

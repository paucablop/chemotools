import re

import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from chemotools.adaptation import MetadataFunctionTransformer
from chemotools.adaptation.functions import (
    add_offset,
    divide_by_reference,
    scale_by_factor,
    subtract_reference,
)


def _identity(X):
    """Module-level identity function — picklable for sklearn estimator checks."""
    return X


_METADATA_EXPECTED_FAILURES = {
    "check_dict_unchanged": "transform requires metadata",
    "check_dtype_object": "transform requires metadata",
    "check_estimators_dtypes": "transform requires metadata",
    "check_estimators_pickle": "transform requires metadata",
    "check_f_contiguous_array_estimator": "transform requires metadata",
    "check_fit_idempotent": "transform requires metadata",
    "check_fit_score_takes_y": "transform requires metadata",
    "check_methods_sample_order_invariance": "transform requires metadata",
    "check_methods_subset_invariance": "transform requires metadata",
    "check_pipeline_consistency": "transform requires metadata",
    "check_transformer_data_not_an_array": "transform requires metadata",
    "check_transformer_general": "transform requires metadata",
    "check_transformer_preserve_dtypes": "transform requires metadata",
    "check_transformer_unfitted": "transform requires metadata",
}


class TestSklearnCompliance:
    """Tests for sklearn estimator API compliance."""

    def test_compliance_no_metadata(self):
        """Verifies that MetadataFunctionTransformer with no metadata passes all
        sklearn estimator checks."""
        # Arrange
        transformer = MetadataFunctionTransformer(func=_identity)

        # Act & Assert
        check_estimator(transformer)

    @pytest.mark.parametrize(
        "func, metadata_key",
        [
            (subtract_reference, "reference"),
            (divide_by_reference, "reference"),
            (scale_by_factor, "factor"),
            (add_offset, "offset"),
        ],
    )
    def test_compliance_predefined_functions(self, func, metadata_key):
        """Verifies that MetadataFunctionTransformer with predefined metadata
        functions passes all sklearn estimator checks, with expected failures
        for checks that require metadata to be passed."""
        # Arrange
        transformer = MetadataFunctionTransformer(func=func, metadata=(metadata_key,))

        # Act & Assert
        check_estimator(transformer, expected_failed_checks=_METADATA_EXPECTED_FAILURES)


class TestMetadataFunctionTransformerFit:
    """Tests for the fit method of MetadataFunctionTransformer."""

    def test_fit_raises_on_non_callable_func(self):
        """Verifies that fit raises a TypeError when func is not callable."""
        # Arrange
        transformer = MetadataFunctionTransformer(
            func="Could not inspect the signature of"
        )

        # Act & Assert
        with pytest.raises(
            TypeError,
            match="The 'func' parameter of MetadataFunctionTransformer must be a "
            "callable. Got 'Could not inspect the signature of' instead.",
        ):
            transformer.fit(X=[[1, 2], [3, 4]])

    def test_fit_raises_when_metadata_key_not_in_function_signature(self):
        """Verifies that fit raises a ValueError when a metadata key is not in the
        function signature."""
        # Arrange

        transformer = MetadataFunctionTransformer(
            func=_identity, metadata=("missing_key",)
        )

        # Act & Assert
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"[MetadataFunctionTransformer] The function '_identity' does not "
                f"accept the following arguments requested in `metadata`: "
                f"{list(transformer.metadata)}"
            ),
        ):
            transformer.fit(X=[[1, 2], [3, 4]])

    def test_fit_raises_when_func_requires_arg_missing_from_metadata(self):
        """Verifies that fit raises a ValueError when a function requires an argument
        that is not provided in the metadata."""

        # Arrange
        def func_with_required_arg(X, required_arg):
            return X + required_arg

        transformer = MetadataFunctionTransformer(
            func=func_with_required_arg, metadata=()
        )

        # Act & Assert
        with pytest.raises(
            ValueError,
            match=re.escape(
                "[MetadataFunctionTransformer] The function 'func_with_required_arg' "
                "requires the following arguments without defaults, which are missing "
                "from `metadata`: ['required_arg']"
            ),
        ):
            transformer.fit(X=[[1, 2], [3, 4]])

    def test_fit_accepts_func_with_var_keyword_args(self):
        """Verifies that fit accepts a function that has **kwargs in its signature."""

        # Arrange
        def func_with_kwargs(X, **kwargs):
            return X + sum(kwargs.values())

        transformer = MetadataFunctionTransformer(
            func=func_with_kwargs, metadata=("arg1", "arg2")
        )

        # Act & Assert
        try:
            transformer.fit(X=[[1, 2], [3, 4]], arg1=1, arg2=2)
        except Exception as e:
            pytest.fail(f"fit raised an unexpected exception: {e}")

    def test_fit_raises_when_required_arg_missing_from_metadata_with_var_keyword(self):
        """Verifies that a required positional arg is still checked even when the
        function also accepts **kwargs."""

        # Arrange
        def func_with_required_and_kwargs(X, required, **kwargs):
            return X + required

        transformer = MetadataFunctionTransformer(
            func=func_with_required_and_kwargs, metadata=()
        )

        # Act & Assert
        with pytest.raises(
            ValueError,
            match=re.escape(
                "[MetadataFunctionTransformer] The function "
                "'func_with_required_and_kwargs' requires the following arguments "
                "without defaults, which are missing from `metadata`: ['required']"
            ),
        ):
            transformer.fit(X=[[1, 2], [3, 4]])

    def test_fit_raises_when_metadata_key_is_positional_only(self):
        """Verifies that a metadata key matching a positional-only parameter is
        rejected because the transformer forwards metadata as keyword arguments."""

        # Arrange
        def func_with_positional_only(X, ref, /):
            return X - ref

        transformer = MetadataFunctionTransformer(
            func=func_with_positional_only, metadata=("ref",)
        )

        # Act & Assert
        with pytest.raises(
            ValueError,
            match=re.escape(
                "[MetadataFunctionTransformer] The following keys in `metadata` "
                "correspond to positional-only parameters of "
                "'func_with_positional_only' and cannot be forwarded as keyword "
                "arguments: ['ref']"
            ),
        ):
            transformer.fit(X=[[1, 2], [3, 4]])


class TestMetadataFunctionTransformerTransform:
    """Tests for the transform method of MetadataFunctionTransformer."""

    def test_transform_routes_metadata_to_func(self):
        """Verifies that transform correctly routes metadata to the function."""

        # Arrange
        def func(X, reference):
            return X - reference

        transformer = MetadataFunctionTransformer(func=func, metadata=("reference",))
        X = [[1, 2], [3, 4]]
        reference = [[0.5, 0.5]]
        expected = [[0.5, 1.5], [2.5, 3.5]]

        # Act
        transformer.fit(X, reference=reference)
        X_transformed = transformer.transform(X, reference=reference)

        # Assert
        assert np.allclose(X_transformed, expected), (
            "Transformed output does not match expected result."
        )

    def test_transform_ignores_extra_metadata_keys(self):
        """Verifies that transform ignores metadata keys that are not specified in
        the transformer's metadata parameter."""

        # Arrange
        def func(X, reference):
            return X - reference

        transformer = MetadataFunctionTransformer(func=func, metadata=("reference",))
        X = [[1, 2], [3, 4]]
        reference = [[0.5, 0.5]]
        extra_metadata = {"extra_key": 42}
        expected = [[0.5, 1.5], [2.5, 3.5]]

        # Act
        transformer.fit(X, reference=reference)
        X_transformed = transformer.transform(X, reference=reference, **extra_metadata)

        # Assert
        assert np.allclose(X_transformed, expected), (
            "Transformed output does not match expected result."
        )


class TestMetadataFunctionTransformerFitTransform:
    """Tests for the fit_transform method of MetadataFunctionTransformer."""

    def test_fit_transform_threads_metadata_to_both_steps(self):
        """Verifies that fit_transform combines the effects of fit and transform."""

        # Arrange
        def func(X, reference):
            return X - reference

        transformer = MetadataFunctionTransformer(func=func, metadata=("reference",))
        X = [[1, 2], [3, 4]]
        reference = [[0.5, 0.5]]
        expected = [[0.5, 1.5], [2.5, 3.5]]

        # Act
        X_transformed = transformer.fit_transform(X, reference=reference)

        # Assert
        assert np.allclose(X_transformed, expected), (
            "Transformed output does not match expected result."
        )


class TestMetadataFunctionTransformerRoutingRegisters:
    """Tests for the routing of metadata to the function in
    MetadataFunctionTransformer."""

    def test_routing_registers_all_keys(self):
        """Verifies that the routing of metadata correctly registers the keys
        specified in the transformer's metadata parameter."""

        # Arrange
        def func(X, reference):
            return X - reference

        transformer = MetadataFunctionTransformer(func=func, metadata=("reference",))
        X = [[1, 2], [3, 4]]
        reference = [[0.5, 0.5]]

        # Act
        transformer.fit(X, reference=reference)
        _ = transformer.transform(X, reference=reference)

        # Assert
        assert hasattr(transformer, "metadata"), (
            "Transformer should have a metadata attribute."
        )
        assert "reference" in transformer.metadata, (
            "Metadata keys should include 'reference'."
        )
        assert transformer.get_metadata_routing().fit.__dict__ == {
            "_requests": {"reference": True},
            "owner": "MetadataFunctionTransformer",
            "method": "fit",
        }
        assert transformer.get_metadata_routing().transform.__dict__ == {
            "_requests": {"reference": True},
            "owner": "MetadataFunctionTransformer",
            "method": "transform",
        }

"""Tests for the ikpls-backed PLSRegression with automatic variance calculation."""

import numpy as np
import pytest
from ikpls.sklearn import PLS as IkplsPLS
from sklearn.cross_decomposition import PLSRegression as SklearnPLSRegression
from sklearn.utils.estimator_checks import check_estimator

from chemotools.regression import PLSRegression


# Test compliance with scikit-learn
def test_compliance_pls_regression():
    # Arrange
    transformer = PLSRegression()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
class TestPLSRegressionCompatibility:
    """Test that the ikpls-backed PLSRegression maintains sklearn API compatibility.

    - Score/weight/loading vectors are defined only up to a per-component sign
      (the sign is arbitrary in PLS), so they are compared via ``np.abs``.
    - With a single response variable the two implementations agree to machine
      precision. With multiple response variables they solve the multi-target
      problem differently, so results agree only approximately.
    """

    def test_same_predictions_as_sklearn(self):
        """Test that predictions match sklearn's PLSRegression (1D y)."""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        # Fit both models without scaling
        sklearn_pls = SklearnPLSRegression(n_components=5, scale=False)
        chemotools_pls = PLSRegression(n_components=5, scale=False)

        sklearn_pls.fit(X, y)
        chemotools_pls.fit(X, y)

        # Act
        sklearn_pred = sklearn_pls.predict(X)
        chemotools_pred = chemotools_pls.predict(X)

        # Assert - predictions should be identical
        np.testing.assert_array_almost_equal(
            sklearn_pred,
            chemotools_pred,
            decimal=10,
            err_msg="Predictions should match sklearn",
        )

    @pytest.mark.parametrize("algorithm", [1, 2])
    @pytest.mark.parametrize("y_2d, decimal", [(False, 10), (True, 2)])
    def test_same_transform_as_sklearn(self, algorithm, y_2d, decimal):
        """transform() produces the same X-scores as sklearn, up to the arbitrary
        per-component sign. For a single-target y the two agree to machine
        precision; for a multivariate y they solve the multi-target problem
        differently, so the scores agree only approximately."""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100, 3) if y_2d else np.random.randn(100)

        sklearn_pls = SklearnPLSRegression(n_components=5).fit(X, y)
        chemotools_pls = PLSRegression(n_components=5, algorithm=algorithm).fit(X, y)

        # Act & Assert - each component's sign is arbitrary, so compare magnitudes
        np.testing.assert_array_almost_equal(
            np.abs(sklearn_pls.transform(X)),
            np.abs(chemotools_pls.transform(X)),
            decimal=decimal,
            err_msg="Transform scores should match sklearn up to sign",
        )

    @pytest.mark.parametrize("algorithm", [1, 2])
    def test_same_attributes_as_sklearn(self, algorithm):
        """Test that the sklearn fitted attributes are present and identical
        (up to the arbitrary per-component sign for scores/weights/loadings)."""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        sklearn_pls = SklearnPLSRegression(n_components=5)
        chemotools_pls = PLSRegression(n_components=5, algorithm=algorithm)

        sklearn_pls.fit(X, y)
        chemotools_pls.fit(X, y)

        # Assert - sign-indeterminate attributes match in magnitude
        sign_indeterminate = [
            "x_weights_",
            "y_weights_",
            "x_loadings_",
            "y_loadings_",
            "x_scores_",
            "x_rotations_",
            "y_rotations_",
        ]
        for attr in sign_indeterminate:
            np.testing.assert_array_almost_equal(
                np.abs(getattr(sklearn_pls, attr)),
                np.abs(getattr(chemotools_pls, attr)),
                decimal=10,
                err_msg=f"Attribute {attr} should match sklearn up to sign",
            )

        # Sign-invariant attributes match exactly
        np.testing.assert_array_almost_equal(
            sklearn_pls.coef_,
            chemotools_pls.coef_,
            decimal=10,
            err_msg="coef_ should match sklearn",
        )
        np.testing.assert_array_almost_equal(
            sklearn_pls.intercept_,
            chemotools_pls.intercept_,
            decimal=10,
            err_msg="intercept_ should match sklearn",
        )
        assert sklearn_pls.n_features_in_ == chemotools_pls.n_features_in_

    @pytest.mark.parametrize("algorithm", [1, 2])
    def test_x_scores_attribute_matches_transform(self, algorithm):
        """The fitted x_scores_ attribute equals transform(X) on the training
        data. (There is no y_scores_ attribute; the Y-scores are obtained via
        transform(X, y).)"""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        # Act
        pls = PLSRegression(n_components=5, algorithm=algorithm).fit(X, y)

        # Assert
        np.testing.assert_array_almost_equal(
            pls.x_scores_, pls.transform(X), decimal=12
        )
        assert not hasattr(pls, "y_scores_"), (
            "PLSRegression should not expose a y_scores_ attribute"
        )

    def test_transform_y_scores_match_sklearn_not_u(self):
        """transform(X, y)[1] returns the y_rotations_ projection -- matching
        sklearn's transform(X, y), NOT the u-vectors that sklearn stores in its
        y_scores_ attribute (the two are different quantities)."""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        sklearn_pls = SklearnPLSRegression(n_components=5)
        chemotools_pls = PLSRegression(n_components=5)
        sklearn_pls.fit(X, y)
        chemotools_pls.fit(X, y)

        # Act
        _, sklearn_y_scores = sklearn_pls.transform(X, y)
        _, chemotools_y_scores = chemotools_pls.transform(X, y)

        # Assert - chemotools matches sklearn's transform up to the arbitrary sign
        np.testing.assert_array_almost_equal(
            np.abs(sklearn_y_scores),
            np.abs(chemotools_y_scores),
            decimal=10,
            err_msg="transform Y-scores should match sklearn's transform(X, y)",
        )
        # sklearn's own y_scores_ attribute (the fit-time u-vectors) is a
        # different quantity: it does NOT equal what transform(X, y) returns.
        assert not np.allclose(
            np.abs(sklearn_pls.y_scores_), np.abs(sklearn_y_scores)
        ), "sklearn's y_scores_ attribute should differ from its transform Y-scores"

    def test_same_score_as_sklearn(self):
        """Test that score() method produces same R² as sklearn."""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(100) * 0.1

        sklearn_pls = SklearnPLSRegression(n_components=5)
        chemotools_pls = PLSRegression(n_components=5)

        sklearn_pls.fit(X, y)
        chemotools_pls.fit(X, y)

        # Act
        sklearn_r2 = sklearn_pls.score(X, y)
        chemotools_r2 = chemotools_pls.score(X, y)

        # Assert
        np.testing.assert_almost_equal(
            sklearn_r2,
            chemotools_r2,
            decimal=10,
            err_msg="R² score should match sklearn",
        )

    def test_works_with_scale_true(self):
        """Test that predictions match sklearn's PLSRegression with scaling on."""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(100, 50) * 100  # Large scale
        y = np.random.randn(100) * 10

        # scale=True (center + scale X and Y) reproduces sklearn's scale=True.
        sklearn_pls = SklearnPLSRegression(n_components=5, scale=True)
        chemotools_pls = PLSRegression(n_components=5, scale=True)

        sklearn_pls.fit(X, y)
        chemotools_pls.fit(X, y)

        # Act
        sklearn_pred = sklearn_pls.predict(X)
        chemotools_pred = chemotools_pls.predict(X)

        # Assert
        np.testing.assert_array_almost_equal(
            sklearn_pred,
            chemotools_pred,
            decimal=10,
            err_msg="Predictions with scaling should match sklearn",
        )

    def test_works_with_scale_false(self):
        """Test that predictions match sklearn's PLSRegression with scaling off
        (both center X and Y but do not scale)."""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(100, 50) * 100  # Large scale
        y = np.random.randn(100) * 10

        # scale=False centers but does not scale, matching sklearn's scale=False.
        sklearn_pls = SklearnPLSRegression(n_components=5, scale=False)
        chemotools_pls = PLSRegression(n_components=5, scale=False)

        sklearn_pls.fit(X, y)
        chemotools_pls.fit(X, y)

        # Act
        sklearn_pred = sklearn_pls.predict(X)
        chemotools_pred = chemotools_pls.predict(X)

        # Assert
        np.testing.assert_array_almost_equal(
            sklearn_pred,
            chemotools_pred,
            decimal=10,
            err_msg="Predictions without scaling should match sklearn",
        )

    def test_works_with_multivariate_y(self):
        """Test that it works with multiple y variables.

        ikpls and sklearn solve the multi-target (PLS2) problem differently, so
        predictions agree only approximately on unstructured random data.
        """
        # Arrange
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100, 3)  # 3 y variables

        sklearn_pls = SklearnPLSRegression(n_components=5)
        chemotools_pls = PLSRegression(n_components=5)

        sklearn_pls.fit(X, y)
        chemotools_pls.fit(X, y)

        # Act
        sklearn_pred = sklearn_pls.predict(X)
        chemotools_pred = chemotools_pls.predict(X)

        # Assert
        np.testing.assert_array_almost_equal(
            sklearn_pred,
            chemotools_pred,
            decimal=2,
            err_msg="Multivariate predictions should match sklearn approximately",
        )

    def test_fit_transform_tuple_convention(self):
        """fit_transform(X, y) returns (x_scores, y_scores), like sklearn's
        cross-decomposition estimators (and the previous NIPALS backend)."""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        # Act
        result = PLSRegression(n_components=5).fit_transform(X, y)

        # Assert
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0].shape == (100, 5)
        assert result[1].shape == (100, 5)


class TestIkplsBackend:
    """Test the ikpls backend guarantees and the extended parameter surface."""

    def test_public_param_surface(self):
        """The public API is exactly {n_components, scale, algorithm, copy,
        dtype}. The ikpls backend's independent center/scale flags, ddof, and
        sample_weight are intentionally hidden and driven by read-only properties
        (ddof fixed to 1), so re-exposing one would break this contract
        (check_estimator only enforces __init__<->get_params consistency)."""
        pls = PLSRegression()

        # Exactly the five public parameters
        assert set(pls.get_params()) == {
            "n_components",
            "scale",
            "algorithm",
            "copy",
            "dtype",
        }
        for hidden in (
            "center_X",
            "center_Y",
            "scale_X",
            "scale_Y",
            "ddof",
            "sample_weight",
        ):
            assert hidden not in pls.get_params()

        # Fixed/derived properties feed the inner model correctly...
        assert pls.center_X is True and pls.center_Y is True
        assert pls.ddof == 1
        assert pls.scale_X is True and pls.scale_Y is True
        pls.set_params(scale=False)
        assert pls.scale_X is False and pls.scale_Y is False

        # ...and the hidden flags cannot be set as parameters
        with pytest.raises(ValueError):
            PLSRegression().set_params(scale_X=False)
        with pytest.raises(ValueError):
            PLSRegression().set_params(ddof=0)

    def test_exact_match_with_ikpls_wrapper(self):
        """chemotools' PLSRegression delegates all math to ikpls.sklearn.PLS,
        so their results are bit-identical for equal parameters."""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        # Act
        chemotools_pls = PLSRegression(n_components=5).fit(X, y)
        ikpls_pls = IkplsPLS(n_components=5, ddof=1).fit(X, y)

        # Assert
        np.testing.assert_array_equal(chemotools_pls.coef_, ikpls_pls.coef_)
        np.testing.assert_array_equal(chemotools_pls.predict(X), ikpls_pls.predict(X))
        np.testing.assert_array_equal(
            chemotools_pls.transform(X), ikpls_pls.transform(X)
        )

    def test_predict_all_components(self):
        """predict_all_components returns predictions for every component
        count 1..n_components in one call, each equal to a refit with that
        number of components (IKPLS extracts components sequentially)."""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        pls = PLSRegression(n_components=5).fit(X, y)

        # Act
        all_preds = pls.predict_all_components(X)

        # Assert
        assert all_preds.shape == (5, 100, 1)
        for k in [1, 3, 5]:
            refit = PLSRegression(n_components=k).fit(X, y)
            np.testing.assert_array_almost_equal(
                all_preds[k - 1].ravel(),
                refit.predict(X),
                decimal=12,
                err_msg=f"predict_all_components[{k - 1}] should equal an "
                f"n_components={k} refit",
            )

    def test_algorithm_two_matches_algorithm_one(self):
        """Algorithms 1 and 2 give the same results."""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        # Act
        pls1 = PLSRegression(n_components=5, algorithm=1).fit(X, y)
        pls2 = PLSRegression(n_components=5, algorithm=2).fit(X, y)

        # Assert
        np.testing.assert_array_almost_equal(pls1.coef_, pls2.coef_, decimal=10)
        np.testing.assert_array_almost_equal(
            pls1.explained_y_variance_ratio_,
            pls2.explained_y_variance_ratio_,
            decimal=10,
        )
        # explained_x depends on x_scores_, which the two algorithms build via
        # different code paths (algo 1: inner_.T; algo 2: transform) -- assert they
        # agree too.
        np.testing.assert_array_almost_equal(
            pls1.explained_x_variance_ratio_,
            pls2.explained_x_variance_ratio_,
            decimal=10,
        )
        np.testing.assert_array_almost_equal(
            np.abs(pls1.x_scores_), np.abs(pls2.x_scores_), decimal=10
        )

    def test_copy_false_matches_copy_true(self):
        """copy=False must give the same fitted results as copy=True.

        The inner PLS centers/scales X and y in place when copy=False; fit
        snapshots the raw inputs so x_scores_ and the explained-variance ratios
        are computed from unmutated data. Previously they were double-preprocessed
        -- negative X ratios (both algorithms) and, for algorithm 2, wrong
        x_scores_ from re-transforming already-preprocessed X.
        """
        # Arrange - nonzero means/scales so preprocessing is non-trivial
        np.random.seed(42)
        X0 = np.random.randn(30, 6) * 3 + 10
        Y0 = np.random.randn(30, 2)

        # Act & Assert
        for algo in (1, 2):
            ref = PLSRegression(n_components=4, algorithm=algo, copy=True).fit(
                X0.copy(), Y0.copy()
            )
            m = PLSRegression(n_components=4, algorithm=algo, copy=False).fit(
                X0.copy(), Y0.copy()
            )
            np.testing.assert_array_almost_equal(
                m.x_scores_,
                ref.x_scores_,
                decimal=10,
                err_msg=f"x_scores_ mismatch copy=False vs True, algorithm={algo}",
            )
            np.testing.assert_array_almost_equal(
                m.explained_x_variance_ratio_,
                ref.explained_x_variance_ratio_,
                decimal=10,
                err_msg=f"explained_x mismatch, algorithm={algo}",
            )
            np.testing.assert_array_almost_equal(
                m.explained_y_variance_ratio_,
                ref.explained_y_variance_ratio_,
                decimal=10,
                err_msg=f"explained_y mismatch, algorithm={algo}",
            )
            assert np.all(m.explained_x_variance_ratio_ >= 0), (
                f"copy=False produced negative X ratios for algorithm={algo}"
            )

    def test_x_scores_does_not_alias_inner_model(self):
        """x_scores_ is an independent array, not a view of the inner model's
        stored scores. Algorithm 1 previously aliased inner_.T (np.asarray does
        not copy), so mutating x_scores_ corrupted the inner model."""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(40, 6)
        y = np.random.randn(40)

        # Act
        pls = PLSRegression(n_components=3, algorithm=1).fit(X, y)

        # Assert - inner_.T exists for algorithm 1 but x_scores_ must not alias it
        assert pls.inner_.T is not None
        assert not np.shares_memory(pls.x_scores_, pls.inner_.T), (
            "x_scores_ must not share memory with the inner model's stored T"
        )
        inner_T_before = pls.inner_.T.copy()
        pls.x_scores_[:] = 0.0
        np.testing.assert_array_equal(
            pls.inner_.T,
            inner_T_before,
            err_msg="mutating x_scores_ must not corrupt the inner model",
        )

    def test_dataframe_algorithm2_copy_false_no_feature_name_warning(self):
        """Fitting a DataFrame with algorithm=2, copy=False must not emit
        scikit-learn's spurious 'X does not have valid feature names' warning: the
        internal algorithm-2 x_scores_ recompute goes through the inner model, not
        the sklearn wrapper's feature-name-validating transform.

        The DataFrame must use *string* column names so that sklearn records
        ``feature_names_in_``; only then does the wrapper's ``transform`` path
        emit the warning, so this is what makes the test able to catch a
        regression (integer column labels are ignored by sklearn and would make
        the test vacuous).
        """
        pytest.importorskip("pandas")
        import warnings

        import pandas as pd

        np.random.seed(42)
        X = pd.DataFrame(
            np.random.randn(40, 6), columns=[f"feature_{i}" for i in range(6)]
        )
        y = np.random.randn(40)

        # Turn *only* the spurious feature-name warning into an error so the fit
        # fails at the point of emission; unrelated warnings still pass through.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error", message="X does not have valid feature names"
            )
            PLSRegression(n_components=3, algorithm=2, copy=False).fit(X, y)

    def test_invalid_parameters_raise(self):
        """Constructor parameters are validated at fit time (sklearn-style)."""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(20, 5)
        y = np.random.randn(20)

        # Act & Assert
        with pytest.raises(ValueError, match="n_components"):
            PLSRegression(n_components=0).fit(X, y)
        with pytest.raises(ValueError, match="algorithm"):
            PLSRegression(algorithm=3).fit(X, y)


class TestPLSRegressionVarianceCalculation:
    """Test the variance calculation features."""

    def test_has_explained_variance_attributes(self):
        """Test that explained variance attributes are created after fitting."""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        # Act
        pls = PLSRegression(n_components=50)
        pls.fit(X, y)

        # Assert
        assert hasattr(pls, "explained_x_variance_ratio_")
        assert hasattr(pls, "explained_y_variance_ratio_")
        assert len(pls.explained_x_variance_ratio_) == 50
        assert len(pls.explained_y_variance_ratio_) == 50

    def test_x_variance_sums_to_one(self):
        """X-space variance ratios sum to 1.0 at full rank -- including for
        sub-unit-std features (the common spectral case). This regresses if the
        deflation rescales by anything other than the model's own fitted std
        (e.g. a clamp at 1.0 would leave std<1 features unscaled)."""
        # Arrange - std < 1 per feature exercises the scaling path
        np.random.seed(42)
        X = np.random.randn(100, 50) * 0.3
        y = np.random.randn(100)

        # Act
        pls = PLSRegression(n_components=50)
        pls.fit(X, y)

        # Assert - exact, since the deflation happens in the model's own
        # (inner_.X_mean / inner_.X_std) preprocessed space
        np.testing.assert_almost_equal(
            pls.explained_x_variance_ratio_.sum(),
            1.0,
            decimal=6,
            err_msg="X-space variance should sum to 1.0 at full rank",
        )

    def test_y_variance_high_with_strong_correlation(self):
        """Test Y variance calculation against known literature example.

        Abdi, H. (2003). Partial Least Squares (PLS) Regression.
        In Lewis-Beck M., Bryman A., Futing T. (Eds.),
        Encyclopedia of Social Sciences Research Methods.
        Thousand Oaks (CA): Sage.
        """
        # Arrange - Known example from literature
        X = np.array(
            [
                [7, 7, 13, 7],
                [4, 3, 14, 7],
                [10, 5, 12, 5],
                [16, 7, 11, 3],
                [13, 3, 10, 3],
            ],
            dtype=float,
        )

        y = np.array(
            [[14, 7, 8], [10, 7, 6], [8, 5, 5], [2, 4, 7], [6, 2, 4]], dtype=float
        )

        # Act
        pls = PLSRegression(n_components=3)
        pls.fit(X, y)

        # Assert - Expected values from literature
        expected_y_var = np.array([0.63331666, 0.22064505, 0.10437163])

        np.testing.assert_array_almost_equal(
            pls.explained_y_variance_ratio_,
            expected_y_var,
            decimal=5,
            err_msg="Y variance should match literature values",
        )

    def test_x_variance_all_positive(self):
        """Test that X-space variance ratios are all positive."""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        # Act
        pls = PLSRegression(n_components=5)
        pls.fit(X, y)

        # Assert
        assert np.all(pls.explained_x_variance_ratio_ >= 0), (
            "X-space variance should be non-negative"
        )

    def test_y_variance_all_positive(self):
        """Test that Y-space variance ratios are all positive."""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        # Act
        pls = PLSRegression(n_components=5)
        pls.fit(X, y)

        # Assert
        assert np.all(pls.explained_y_variance_ratio_ >= 0), (
            "Y-space variance should be non-negative"
        )

    def test_y_variance_is_float_array(self):
        """Test that Y-space variance is a proper numpy array."""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(100) * 0.1

        # Act
        pls = PLSRegression(n_components=3)
        pls.fit(X, y)

        # Assert
        assert isinstance(pls.explained_y_variance_ratio_, np.ndarray)
        assert pls.explained_y_variance_ratio_.dtype == np.float64

    def test_variance_calculation_with_pandas(self):
        """Test that variance calculation works with pandas DataFrame/Series."""
        # Arrange
        pytest.importorskip("pandas")
        import pandas as pd

        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 50))
        y = pd.Series(np.random.randn(100))

        # Act
        pls = PLSRegression(n_components=5)
        pls.fit(X, y)

        # Assert
        assert hasattr(pls, "explained_x_variance_ratio_")
        assert hasattr(pls, "explained_y_variance_ratio_")
        assert len(pls.explained_x_variance_ratio_) == 5
        assert len(pls.explained_y_variance_ratio_) == 5

    def test_variance_with_different_n_components(self):
        """Test variance calculation with different numbers of components."""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        # Act & Assert for different component counts
        for n_comp in [2, 5, 10]:
            pls = PLSRegression(n_components=n_comp)
            pls.fit(X, y)

            assert len(pls.explained_x_variance_ratio_) == n_comp
            assert len(pls.explained_y_variance_ratio_) == n_comp

    def test_variance_respects_preprocessing_flags(self):
        """With the default centering, the full-rank X variance sums to 1 whether
        or not scaling is enabled -- including for sub-unit-std, offset features,
        which regresses if the deflation does not use the model's own fitted std.
        Disabling centering is a documented limitation and is not asserted here."""
        # Arrange - std < 1 (and a mean offset) so scale=True is truly exercised
        np.random.seed(42)
        X = np.random.randn(100, 20) * 0.3 + 5
        y = np.random.randn(100)

        # Act & Assert - full-rank X variance sums to 1 in the centered space
        for kwargs in [
            {},
            {"scale": False},
        ]:
            pls = PLSRegression(n_components=20, **kwargs)
            pls.fit(X, y)
            np.testing.assert_almost_equal(
                pls.explained_x_variance_ratio_.sum(),
                1.0,
                decimal=6,
                err_msg=f"X variance should sum to 1 for {kwargs}",
            )

    def test_variance_ratios_finite_and_shaped(self):
        """Sanity: with the public API (always centered) the ratios are finite
        and correctly shaped for a well-posed fit."""
        # Arrange
        np.random.seed(42)
        n_components = 6
        X = np.random.randn(60, 10)
        y = np.random.randn(60)

        # Act
        pls = PLSRegression(n_components=n_components).fit(X, y)

        # Assert
        assert pls.explained_x_variance_ratio_.shape == (n_components,)
        assert pls.explained_y_variance_ratio_.shape == (n_components,)
        assert np.all(np.isfinite(pls.explained_x_variance_ratio_))
        assert np.all(np.isfinite(pls.explained_y_variance_ratio_))

    def test_variance_ratios_valid_beyond_numerical_rank(self):
        """When n_components exceeds the numerical rank of the (centered) X, the
        null-space components carry no variance and are given zero, so the X
        ratios stay non-negative, sum to 1, and agree between algorithms 1 and 2.
        Covers wide (p > n; centered rank n-1) and collinear (duplicated column)
        data -- the cases that previously produced negative, non-normalized, or
        algorithm-dependent ratios."""
        np.random.seed(42)
        X_wide = np.random.randn(15, 60)  # centered rank 14; ask for min(n, p) = 15
        X_coll = np.random.randn(30, 6)  # duplicated column -> rank 5
        X_coll[:, 5] = X_coll[:, 0]

        for X, k in [(X_wide, 15), (X_coll, 6)]:
            y = np.random.randn(X.shape[0])
            pls1 = PLSRegression(n_components=k, algorithm=1).fit(X, y)
            pls2 = PLSRegression(n_components=k, algorithm=2).fit(X, y)
            for pls in (pls1, pls2):
                xr = pls.explained_x_variance_ratio_
                assert np.all(xr >= 0), f"negative X ratio for {X.shape}: {xr}"
                np.testing.assert_almost_equal(
                    xr.sum(),
                    1.0,
                    decimal=6,
                    err_msg=f"X ratios should sum to 1 past rank for {X.shape}",
                )
                assert np.all(np.isfinite(pls.explained_y_variance_ratio_))
            np.testing.assert_array_almost_equal(
                pls1.explained_x_variance_ratio_,
                pls2.explained_x_variance_ratio_,
                decimal=10,
                err_msg=f"algorithms should agree past rank for {X.shape}",
            )

    def test_variance_calculation_preserves_sklearn_behavior(self):
        """Test that adding variance calculation doesn't change predictions."""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(100) * 0.1

        sklearn_pls = SklearnPLSRegression(n_components=5)
        chemotools_pls = PLSRegression(n_components=5)

        # Fit both
        sklearn_pls.fit(X, y)
        chemotools_pls.fit(X, y)

        # Calculate predictions before accessing variance
        sklearn_pred = sklearn_pls.predict(X)
        chemotools_pred = chemotools_pls.predict(X)

        # Access variance (this shouldn't change anything)
        _ = chemotools_pls.explained_x_variance_ratio_
        _ = chemotools_pls.explained_y_variance_ratio_

        # Recalculate predictions
        chemotools_pred_after = chemotools_pls.predict(X)

        # Assert
        np.testing.assert_array_almost_equal(
            sklearn_pred,
            chemotools_pred,
            decimal=10,
            err_msg="Predictions should match sklearn",
        )
        np.testing.assert_array_almost_equal(
            chemotools_pred,
            chemotools_pred_after,
            decimal=10,
            err_msg="Variance calculation shouldn't change predictions",
        )


class TestPLSRegressionEdgeCases:
    """Test edge cases and error handling."""

    def test_works_with_minimum_samples(self):
        """Test that it works with minimum number of samples."""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(10, 5)  # Only 10 samples
        y = np.random.randn(10)

        # Act
        pls = PLSRegression(n_components=3)
        pls.fit(X, y)

        # Assert
        assert hasattr(pls, "explained_x_variance_ratio_")
        assert hasattr(pls, "explained_y_variance_ratio_")

    def test_repr_before_fitting(self):
        """Test that __repr__ works before fitting (no variance info)."""
        # Arrange
        pls = PLSRegression(n_components=5)

        # Act
        repr_str = repr(pls)

        # Assert
        assert "n_components=5" in repr_str
        assert "X-space variance" not in repr_str  # Not fitted yet


class TestChemotoolsIntegration:
    """The ikpls-backed PLSRegression stays a drop-in model for the rest of
    chemotools (outlier detection, feature selection, inspector, plotting)."""

    @staticmethod
    def _fitted_model():
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(100) * 0.1
        return PLSRegression(n_components=3).fit(X, y), X, y

    def test_outlier_detectors_accept_pls_regression(self):
        # Arrange
        from chemotools.outliers import (
            HotellingT2,
            Leverage,
            QResiduals,
            StudentizedResiduals,
        )

        pls, X, y = self._fitted_model()

        # Act & Assert
        for detector_cls, needs_y in [
            (HotellingT2, False),
            (Leverage, False),
            (QResiduals, False),
            (StudentizedResiduals, True),
        ]:
            detector = detector_cls(model=pls)
            detector.fit(X, y)
            outliers = detector.predict(X, y) if needs_y else detector.predict(X)
            assert outliers.shape == (100,), detector_cls.__name__

    def test_feature_selectors_accept_pls_regression(self):
        # Arrange
        from chemotools.feature_selection import SRSelector, VIPSelector

        pls, X, y = self._fitted_model()

        # Act & Assert
        for selector_cls in (VIPSelector, SRSelector):
            selector = selector_cls(model=pls)
            X_selected = selector.fit_transform(X)
            assert X_selected.shape[0] == 100, selector_cls.__name__

    def test_inspector_accepts_pls_regression(self):
        # Arrange
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        from chemotools.inspector import PLSRegressionInspector

        pls, X, y = self._fitted_model()

        # Act
        inspector = PLSRegressionInspector(pls, X, y)

        # Assert
        assert inspector.get_x_scores("train").shape == (100, 3)
        assert inspector.get_y_scores("train").shape == (100, 3)

    def test_inspector_accepts_pipeline_with_pls_regression(self):
        # Arrange
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        from chemotools.inspector import PLSRegressionInspector

        _, X, y = self._fitted_model()
        pipeline = make_pipeline(StandardScaler(), PLSRegression(n_components=3))
        pipeline.fit(X, y)

        # Act & Assert
        PLSRegressionInspector(pipeline, X, y)

    def test_explained_variance_plot_accepts_ratios(self):
        # Arrange
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        from chemotools.plotting import ExplainedVariancePlot

        pls, _, _ = self._fitted_model()

        # Act & Assert
        ExplainedVariancePlot(pls.explained_y_variance_ratio_)

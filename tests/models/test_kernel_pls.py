import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KernelCenterer, StandardScaler
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted

from chemotools.models._kernel_pls import KernelPLS


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(17)
    X = rng.normal(size=(100, 20))
    y = (np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.1 * rng.normal(size=100)).reshape(
        -1, 1
    )
    X_test = rng.normal(size=(10, 20))
    return X, y, X_test


@pytest.fixture
def sample_data_2d_y(sample_data):
    X, y, X_test = sample_data
    y_2d = np.column_stack([y, y**2])
    return X, y_2d, X_test


class TestSklearnCompliance:
    """Tests for sklearn estimator API compliance."""

    def test_compliance_KernelPLS(self):
        """Verifies that KernelPLS passes all sklearn estimator
        checks."""
        # Arrange
        transformer = KernelPLS()
        # Act & Assert
        check_estimator(transformer)


class TestFitPredict:
    def test_works_without_parameters(self, sample_data) -> None:
        """Verifies that KernelPLS can be fitted with default parameters."""
        # Arrange
        X, y, X_test = sample_data
        # Act
        model = KernelPLS().fit(X, y)
        # Assert
        check_is_fitted(model)

    def test_predict_shape(self, sample_data) -> None:
        """Verifies that predict returns the correct number of samples."""
        # Arrange
        X, y, X_test = sample_data
        # Act
        model = KernelPLS().fit(X, y)
        # Assert
        y_hat = model.predict(X_test)
        assert y_hat.shape[0] == X_test.shape[0]

    def test_predict_returns_numpy_array(self, sample_data) -> None:
        """Verifies that predict returns a NumPy array."""
        # Arrange
        X, y, X_test = sample_data

        # Act
        y_hat = KernelPLS().fit(X, y).predict(X_test)

        # Assert
        assert isinstance(y_hat, np.ndarray)

    def test_predict_is_deterministic(self, sample_data) -> None:
        """Verifies that predictions are deterministic across repeated calls."""
        # Arrange
        X, y, X_test = sample_data
        model = KernelPLS().fit(X, y)

        # Act & Assert
        np.testing.assert_array_equal(
            model.predict(X_test),
            model.predict(X_test),
        )

    def test_predict_single_sample(self, sample_data) -> None:
        X, y, X_test = sample_data
        y_hat = KernelPLS().fit(X, y).predict(X_test[:1])
        assert y_hat.shape[0] == 1

    def test_predict_single_sample_1d_raises_error(self, sample_data) -> None:
        """Verifies that predict rejects 1D input samples."""
        # Arrange
        X, y, X_test = sample_data
        model = KernelPLS().fit(X, y)

        # Act & Assert
        with pytest.raises(ValueError, match="Expected 2D array"):
            model.predict(X_test[0])


class TestAttributes:
    def test_has_attributes(self, sample_data) -> None:
        """Verifies that the model has all the attributes"""
        # Arrange
        X, y, X_test = sample_data
        # Act
        model = KernelPLS().fit(X, y)
        # Assert
        attributes = [
            "X_train_",
            "X_mean_",
            "X_std_",
            "x_weights_",
            "y_weights_",
            "K_train_c_",
            "x_loadings_",
            "y_loadings_",
            "x_scores_",
            "y_scores_",
            "x_rotations_",
            "y_rotations_",
            "coef_",
            "intercept_",
            "n_iter_",
            "K_fit_rows_",
            "K_fit_all_",
            "y_mean_",
            "y_std_",
            "y_was_1d_",
        ]
        for attr in attributes:
            assert hasattr(model, attr), f"Missing attribute: {attr}"

    def test_fitted_attribute_shapes(self, sample_data) -> None:
        """Verifies shapes of important fitted attributes."""
        # Arrange
        X, y, _ = sample_data
        n_components = 3

        # Act
        model = KernelPLS(n_components=n_components).fit(X, y)

        # Assert
        assert model.X_train_.shape == X.shape
        assert model.K_train_c_.shape == (X.shape[0], X.shape[0])
        assert model.x_scores_.shape == (X.shape[0], n_components)
        assert model.x_weights_.shape[1] == n_components
        assert model.coef_.ndim in {1, 2}
        assert len(model.n_iter_) == n_components

    def test_x_mean_std_identity_when_scale_X_false(self, sample_data) -> None:
        """Verifies that X_mean_ and X_std_ are identity values when scale_X=False."""
        # Arrange
        X, y, _ = sample_data

        # Act
        model = KernelPLS(scale_X=False).fit(X, y)

        # Assert
        np.testing.assert_array_equal(model.X_mean_, np.zeros(X.shape[1]))
        np.testing.assert_array_equal(model.X_std_, np.ones(X.shape[1]))

    def test_x_mean_std_set_when_scale_X_true(self, sample_data) -> None:
        """Verifies that X_mean_ and X_std_ match the data statistics
        when scale_X=True."""
        # Arrange
        X, y, _ = sample_data

        # Act
        model = KernelPLS(scale_X=True).fit(X, y)

        # Assert
        np.testing.assert_array_almost_equal(model.X_mean_, X.mean(axis=0))
        np.testing.assert_array_almost_equal(model.X_std_, X.std(axis=0))


class TestStd:
    @pytest.mark.parametrize("scale", [False, True])
    @pytest.mark.parametrize("kernel", ["rbf", "linear", "poly", "sigmoid"])
    def test_predictions_match_sklearn_kernelcenterer_pls_with_scale(
        self, sample_data, kernel, scale
    ) -> None:
        """Verifies that KernelPLS predictions match a reference sklearn pipeline
        (KernelCenterer + PLSRegression) for different kernels and scaling options."""
        # Arrange
        X, y, X_test = sample_data
        gamma = 0.5
        n_components = 2

        model = KernelPLS(
            n_components=n_components,
            kernel=kernel,
            gamma=gamma,
            scale=scale,
        ).fit(X, y)

        if kernel in {"rbf", "poly", "sigmoid"}:
            K_train = pairwise_kernels(X, X, metric=kernel, gamma=gamma)
            K_test = pairwise_kernels(X_test, X, metric=kernel, gamma=gamma)
        else:
            K_train = pairwise_kernels(X, X, metric=kernel)
            K_test = pairwise_kernels(X_test, X, metric=kernel)

        centerer = KernelCenterer().fit(K_train)
        K_train_c = centerer.transform(K_train)
        K_test_c = centerer.transform(K_test)

        reference_model = PLSRegression(
            n_components=n_components,
            scale=scale,
        ).fit(K_train_c, y.reshape(-1, 1))

        y_hat_reference = reference_model.predict(K_test_c)

        # Act
        y_hat = model.predict(X_test)

        # Assert
        np.testing.assert_array_almost_equal(
            y_hat,
            y_hat_reference,
            decimal=10,
        )


class TestKernels:
    @pytest.mark.parametrize("kernel", ["rbf", "linear", "poly"])
    def test_manual_kernel_centering_matches_sklearn_training_kernel(
        self, sample_data, kernel
    ) -> None:
        """Verifies that manual kernel centering matches sklearn
        KernelCenterer output."""
        # Arrange
        X, y, X_test = sample_data
        gamma = 0.5
        model = KernelPLS(kernel=kernel, gamma=gamma).fit(X, y)
        model.predict(X_test)
        if kernel in {"rbf", "poly", "sigmoid"}:
            K_train = pairwise_kernels(X, X, metric=kernel, gamma=gamma)
        else:
            K_train = pairwise_kernels(X, X, metric=kernel)

        sklearn_centerer = KernelCenterer().fit(K_train)
        K_train_sklearn = sklearn_centerer.transform(K_train)

        # Assert
        np.testing.assert_array_almost_equal(
            model.K_train_c_,
            K_train_sklearn,
            decimal=10,
        )


class TestScalingAndTargets:
    def test_scale_X_changes_predictions(self, sample_data) -> None:
        """Verifies that enabling/disabling input scaling changes
        the model predictions."""
        # Arrange
        X, y, X_test = sample_data

        # Act
        y1 = KernelPLS(scale_X=False).fit(X, y).predict(X_test)
        y2 = KernelPLS(scale_X=True).fit(X, y).predict(X_test)

        # Assert
        assert np.max(np.abs(y1 - y2)) > 1e-8

    def test_scale_X_equivalent_to_manual_scaling(self, sample_data) -> None:
        """Verifies that scale_X=True is equivalent to manually scaling
        X before fitting."""
        # Arrange
        X, y, X_test = sample_data
        scaler = StandardScaler(with_std=True).fit(X)
        X_sc = scaler.transform(X)
        X_test_sc = scaler.transform(X_test)

        # Act
        y_manual = KernelPLS(scale_X=False).fit(X_sc, y).predict(X_test_sc)
        y_auto = KernelPLS(scale_X=True).fit(X, y).predict(X_test)

        # Assert
        np.testing.assert_array_almost_equal(y_manual, y_auto, decimal=8)

    def test_n_components_too_large_raises_error(self, sample_data) -> None:
        """Ensures that requesting more components than available
        samples raises a ValueError."""
        # Arrange
        X, y, _ = sample_data

        # Act and Assert
        with pytest.raises(ValueError, match="n_components"):
            KernelPLS(n_components=X.shape[0] + 1).fit(X, y)

    def test_works_with_multivariate_y(self, sample_data_2d_y) -> None:
        """Verifies that the model supports multivariate (multi-output)
        regression targets."""
        # Arrange
        X, y, X_test = sample_data_2d_y

        # Act
        y_hat = KernelPLS().fit(X, y).predict(X_test)

        # Assert
        assert y_hat.shape == (X_test.shape[0], 2)

    def test_1d_y_returns_1d_prediction(self, sample_data):
        """Ensures that 1D targets produce 1D predictions."""

        # Arrange
        X, y, X_test = sample_data

        # Act
        if y.ndim == 2:
            y = y.ravel()
        y_hat = KernelPLS().fit(y=y, X=X).predict(X_test)

        # Assert
        assert y_hat.ndim == 1

    def test_2d_y_returns_2d_prediction(self, sample_data) -> None:
        """Ensures that 2D targets produce 2D predictions."""
        # Arrange
        X, y, X_test = sample_data
        y_2d = y.reshape(-1, 1)

        # Act
        y_hat = KernelPLS().fit(X, y_2d).predict(X_test)

        # Assert
        assert y_hat.ndim == 2

    @pytest.mark.parametrize("kernel", ["rbf", "linear", "poly", "sigmoid"])
    def test_predictions_match_sklearn_kernelcenterer_pls(
        self, sample_data, kernel
    ) -> None:
        """Verifies that KernelPLS predictions match a reference implementation using
        sklearn KernelCenterer + PLSRegression on centered kernel matrices."""
        # Arrange
        X, y, X_test = sample_data
        gamma = 0.5
        n_components = 2
        scale = False

        model = KernelPLS(
            n_components=n_components,
            kernel=kernel,
            gamma=gamma,
            scale=scale,
        ).fit(X, y)

        if kernel in {"rbf", "poly", "sigmoid"}:
            K_train = pairwise_kernels(X, X, metric=kernel, gamma=gamma)
            K_test = pairwise_kernels(X_test, X, metric=kernel, gamma=gamma)
        else:
            K_train = pairwise_kernels(X, X, metric=kernel)
            K_test = pairwise_kernels(X_test, X, metric=kernel)

        centerer = KernelCenterer().fit(K_train)
        K_train_c = centerer.transform(K_train)
        K_test_c = centerer.transform(K_test)

        reference_model = PLSRegression(
            n_components=n_components,
            scale=scale,
        ).fit(K_train_c, y.reshape(-1, 1))

        y_hat_reference = reference_model.predict(K_test_c)

        # Act
        y_hat = model.predict(X_test)

        # Assert
        np.testing.assert_array_almost_equal(
            y_hat,
            y_hat_reference,
            decimal=10,
        )


class TestTransform:
    def test_transform_shape(self, sample_data) -> None:
        """Ensures that transform output has shape (n_samples, n_components)."""
        # Arrange
        X, y, X_test = sample_data
        n_components = 3

        # Act
        T = KernelPLS(n_components=n_components).fit(X, y).transform(X_test)

        # Assert
        assert T.shape == (X_test.shape[0], n_components)

    def test_transform_is_deterministic(self, sample_data) -> None:
        """Ensures that repeated calls to transform return identical results."""
        # Arrange
        X, y, X_test = sample_data

        # Act
        model = KernelPLS().fit(X, y)

        # Assert
        np.testing.assert_array_equal(model.transform(X_test), model.transform(X_test))

    def test_transform_before_fit_raises_error(self, sample_data) -> None:
        """Ensures that calling transform before fit raises NotFittedError."""
        # Arrange
        _, _, X_test = sample_data

        # Act & Assert
        with pytest.raises(NotFittedError):
            KernelPLS().transform(X_test)


class TestErrorHandling:
    def test_predict_before_fit_raises_error(self, sample_data) -> None:
        """Ensures that calling predict before fit raises NotFittedError."""
        # Arrange
        _, _, X_test = sample_data

        # Act & Assert
        with pytest.raises(NotFittedError):
            KernelPLS().predict(X_test)

    def test_unsupported_kernel_raises_error(self, sample_data) -> None:
        """Ensures that providing an unsupported kernel raises a ValueError."""
        # Arrange
        X, y, _ = sample_data

        # Act & Assert
        with pytest.raises(ValueError, match="kernel"):
            KernelPLS(kernel="wrong_kernel").fit(X, y)


class TestPipelineGridSearchCV:
    def test_pipeline_compatibility(self, sample_data) -> None:
        """Ensures that KernelPLS works correctly inside a scikit-learn Pipeline."""
        # Arrange
        X, y, X_test = sample_data

        # Act
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KernelPLS(n_components=2, kernel="rbf", gamma=0.5)),
            ]
        )
        pipe.fit(X, y)
        y_hat = pipe.predict(X_test)

        # Assert
        assert y_hat.shape[0] == X_test.shape[0]
        assert isinstance(y_hat, np.ndarray)

    def test_pipeline_gridsearchcv_compatibility(self, sample_data) -> None:
        """Ensures that KernelPLS is compatible with GridSearchCV
        hyperparameter tuning."""
        # Arrange
        X, y, _ = sample_data

        # Act
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KernelPLS()),
            ]
        )
        param_grid = [
            {"model__kernel": ["linear"], "model__n_components": [1, 2]},
            {
                "model__kernel": ["rbf"],
                "model__gamma": [0.1, 0.5],
                "model__n_components": [1, 2],
            },
        ]
        grid = GridSearchCV(pipe, param_grid, cv=3, scoring="r2", error_score="raise")
        grid.fit(X, y)

        # Assert
        assert grid.best_estimator_ is not None
        assert np.isfinite(grid.best_score_)


class TestSklearnAPIExtended:
    """Additional tests for sklearn estimator API compatibility."""

    def test_clone_preserves_parameters(self) -> None:
        """Verifies that sklearn.clone preserves init parameters."""
        # Arrange
        model = KernelPLS(
            n_components=3,
            kernel="rbf",
            gamma=0.5,
            scale=False,
            scale_X=True,
        )

        # Act
        cloned_model = clone(model)

        # Assert
        assert cloned_model.n_components == model.n_components
        assert cloned_model.kernel == model.kernel
        assert cloned_model.gamma == model.gamma
        assert cloned_model.scale == model.scale
        assert cloned_model.scale_X == model.scale_X

    def test_clone_can_be_fitted(self, sample_data) -> None:
        """Verifies that a cloned KernelPLS estimator can be fitted."""
        # Arrange
        X, y, X_test = sample_data
        model = KernelPLS(n_components=2, kernel="rbf", gamma=0.5)

        # Act
        cloned_model = clone(model).fit(X, y)
        y_hat = cloned_model.predict(X_test)

        # Assert
        check_is_fitted(cloned_model)
        assert y_hat.shape[0] == X_test.shape[0]

    def test_n_features_in_attribute_is_set(self, sample_data) -> None:
        """Verifies that n_features_in_ is set after fit."""
        # Arrange
        X, y, _ = sample_data

        # Act
        model = KernelPLS().fit(X, y)

        # Assert
        assert hasattr(model, "n_features_in_")
        assert model.n_features_in_ == X.shape[1]

    def test_predict_with_wrong_number_of_features_raises_error(
        self, sample_data
    ) -> None:
        """Verifies that predict validates the number of input features."""
        # Arrange
        X, y, _ = sample_data
        X_wrong = X[:, :-1]
        model = KernelPLS().fit(X, y)

        # Act & Assert
        with pytest.raises(ValueError):
            model.predict(X_wrong)

    def test_transform_with_wrong_number_of_features_raises_error(
        self, sample_data
    ) -> None:
        """Verifies that transform validates the number of input features."""
        # Arrange
        X, y, _ = sample_data
        X_wrong = X[:, :-1]
        model = KernelPLS().fit(X, y)

        # Act & Assert
        with pytest.raises(ValueError):
            model.transform(X_wrong)


class TestFitTransform:
    """Tests for fit_transform behavior."""

    def test_fit_transform_shape(self, sample_data) -> None:
        """Verifies that fit_transform returns the expected scores shape."""
        # Arrange
        X, y, _ = sample_data
        n_components = 3

        # Act
        T = KernelPLS(n_components=n_components).fit_transform(X, y)

        # Assert
        assert T.shape == (X.shape[0], n_components)

    def test_fit_transform_matches_fit_then_transform(self, sample_data) -> None:
        """fit_transform should match fit followed by transform."""
        # Arrange
        X, y, _ = sample_data
        model_1 = KernelPLS(n_components=2)
        model_2 = KernelPLS(n_components=2)

        # Act
        T_fit_transform = model_1.fit_transform(X, y)
        T_fit_then_transform = model_2.fit(X, y).transform(X)

        # Assert
        np.testing.assert_array_almost_equal(
            T_fit_transform,
            T_fit_then_transform,
            decimal=10,
        )

    def test_fit_transform_returns_numpy_array(self, sample_data) -> None:
        """Verifies that fit_transform returns a numpy array."""
        # Arrange
        X, y, _ = sample_data

        # Act
        T = KernelPLS().fit_transform(X, y)

        # Assert
        assert isinstance(T, np.ndarray)


class TestInputValidation:
    """Tests for input validation and invalid data."""

    def test_fit_with_nan_in_X_raises_error(self, sample_data) -> None:
        """Verifies that NaN values in X are rejected."""
        # Arrange
        X, y, _ = sample_data
        X_nan = X.copy()
        X_nan[0, 0] = np.nan

        # Act & Assert
        with pytest.raises(ValueError):
            KernelPLS().fit(X_nan, y)

    def test_fit_with_inf_in_X_raises_error(self, sample_data) -> None:
        """Verifies that infinite values in X are rejected."""
        # Arrange
        X, y, _ = sample_data
        X_inf = X.copy()
        X_inf[0, 0] = np.inf

        # Act & Assert
        with pytest.raises(ValueError):
            KernelPLS().fit(X_inf, y)

    def test_fit_with_nan_in_y_raises_error(self, sample_data) -> None:
        """Verifies that NaN values in y are rejected."""
        # Arrange
        X, y, _ = sample_data
        y_nan = y.copy()
        y_nan[0, 0] = np.nan

        # Act & Assert
        with pytest.raises(ValueError):
            KernelPLS().fit(X, y_nan)

    def test_fit_with_mismatched_X_y_lengths_raises_error(self, sample_data) -> None:
        """Verifies that inconsistent X and y lengths are rejected."""
        # Arrange
        X, y, _ = sample_data

        # Act & Assert
        with pytest.raises(ValueError):
            KernelPLS().fit(X[:-1], y)

    def test_n_components_zero_raises_error(self, sample_data) -> None:
        """Verifies that n_components=0 is rejected."""
        # Arrange
        X, y, _ = sample_data

        # Act & Assert
        with pytest.raises(ValueError, match="n_components"):
            KernelPLS(n_components=0).fit(X, y)

    def test_n_components_negative_raises_error(self, sample_data) -> None:
        """Verifies that negative n_components is rejected."""
        # Arrange
        X, y, _ = sample_data

        # Act & Assert
        with pytest.raises(ValueError, match="n_components"):
            KernelPLS(n_components=-1).fit(X, y)

    def test_predict_with_nan_raises_error(self, sample_data) -> None:
        """Verifies that NaN values in prediction data are rejected."""
        # Arrange
        X, y, X_test = sample_data
        X_test_nan = X_test.copy()
        X_test_nan[0, 0] = np.nan
        model = KernelPLS().fit(X, y)

        # Act & Assert
        with pytest.raises(ValueError):
            model.predict(X_test_nan)

    def test_transform_with_nan_raises_error(self, sample_data) -> None:
        """Verifies that NaN values in transform data are rejected."""
        # Arrange
        X, y, X_test = sample_data
        X_test_nan = X_test.copy()
        X_test_nan[0, 0] = np.nan
        model = KernelPLS().fit(X, y)

        # Act & Assert
        with pytest.raises(ValueError):
            model.transform(X_test_nan)


class TestTransformConsistency:
    """Tests for mathematical consistency of transform."""

    def test_transform_training_data_matches_x_scores(self, sample_data) -> None:
        """Transforming the training data should reproduce x_scores_."""
        # Arrange
        X, y, _ = sample_data
        model = KernelPLS(n_components=2).fit(X, y)

        # Act
        T = model.transform(X)

        # Assert
        np.testing.assert_array_almost_equal(
            T,
            model.x_scores_,
            decimal=10,
        )

    def test_transform_single_sample(self, sample_data) -> None:
        """Verifies that transform works with a single 2D sample."""
        # Arrange
        X, y, X_test = sample_data
        model = KernelPLS(n_components=2).fit(X, y)

        # Act
        T = model.transform(X_test[:1])

        # Assert
        assert T.shape == (1, 2)

    def test_transform_single_sample_1d_raises_error(self, sample_data) -> None:
        """Verifies that transform rejects 1D samples."""
        # Arrange
        X, y, X_test = sample_data
        model = KernelPLS().fit(X, y)

        # Act & Assert
        with pytest.raises(ValueError, match="Expected 2D array"):
            model.transform(X_test[0])


class TestPredictionConsistency:
    """Tests for consistency of predictions across equivalent target shapes."""

    def test_1d_y_and_2d_single_target_y_give_same_predictions(
        self, sample_data
    ) -> None:
        """1D y and 2D single-target y should give numerically equivalent
        predictions."""
        # Arrange
        X, y, X_test = sample_data
        y_1d = y.ravel()
        y_2d = y.reshape(-1, 1)

        # Act
        y_hat_1d = KernelPLS().fit(X, y_1d).predict(X_test)
        y_hat_2d = KernelPLS().fit(X, y_2d).predict(X_test)

        # Assert
        np.testing.assert_array_almost_equal(
            y_hat_1d,
            y_hat_2d.ravel(),
            decimal=10,
        )

    def test_predict_training_data_shape(self, sample_data) -> None:
        """Verifies prediction shape on the training data."""
        # Arrange
        X, y, _ = sample_data

        # Act
        y_hat = KernelPLS().fit(X, y).predict(X)

        # Assert
        assert y_hat.shape == y.shape


class TestSerialization:
    """Tests for model serialization."""

    def test_pickle_roundtrip_preserves_predictions(self, sample_data) -> None:
        """Verifies that pickling and unpickling preserve predictions."""
        # Arrange
        X, y, X_test = sample_data
        model = KernelPLS(n_components=2, kernel="rbf", gamma=0.5).fit(X, y)
        y_hat_before = model.predict(X_test)

        # Act

        # Safe: using pickle on trusted in-memory objects (test context)
        loaded_model = pickle.loads(pickle.dumps(model))  # nosec B301

        y_hat_after = loaded_model.predict(X_test)

        # Assert
        check_is_fitted(loaded_model)
        np.testing.assert_array_almost_equal(
            y_hat_before,
            y_hat_after,
            decimal=10,
        )


class TestKernelParameters:
    """Tests for kernel-specific parameters."""

    @pytest.mark.parametrize("degree", [2, 3, 5])
    def test_poly_kernel_degree_parameter(self, sample_data, degree) -> None:
        """Verifies that the polynomial degree parameter is accepted."""
        # Arrange
        X, y, X_test = sample_data

        # Act
        model = KernelPLS(
            n_components=2,
            kernel="poly",
            gamma=0.5,
            degree=degree,
        ).fit(X, y)
        y_hat = model.predict(X_test)

        # Assert
        assert y_hat.shape[0] == X_test.shape[0]
        assert np.all(np.isfinite(y_hat))

    @pytest.mark.parametrize("coef0", [0.0, 1.0, 2.5])
    def test_poly_kernel_coef0_parameter(self, sample_data, coef0) -> None:
        """Verifies that the polynomial coef0 parameter is accepted."""
        # Arrange
        X, y, X_test = sample_data

        # Act
        model = KernelPLS(
            n_components=2,
            kernel="poly",
            gamma=0.5,
            coef0=coef0,
        ).fit(X, y)
        y_hat = model.predict(X_test)

        # Assert
        assert y_hat.shape[0] == X_test.shape[0]
        assert np.all(np.isfinite(y_hat))

    @pytest.mark.parametrize("coef0", [0.0, 1.0, 2.5])
    def test_sigmoid_kernel_coef0_parameter(self, sample_data, coef0) -> None:
        """Verifies that the sigmoid coef0 parameter is accepted."""
        # Arrange
        X, y, X_test = sample_data

        # Act
        model = KernelPLS(
            n_components=2,
            kernel="sigmoid",
            gamma=0.1,
            coef0=coef0,
        ).fit(X, y)
        y_hat = model.predict(X_test)

        # Assert
        assert y_hat.shape[0] == X_test.shape[0]
        assert np.all(np.isfinite(y_hat))

    def test_poly_kernel_matches_sklearn_with_degree_and_coef0(
        self, sample_data
    ) -> None:
        """Verifies poly kernel predictions against KernelCenterer + PLSRegression."""
        # Arrange
        X, y, X_test = sample_data
        n_components = 2
        gamma = 0.5
        degree = 4
        coef0 = 1.5
        scale = False

        model = KernelPLS(
            n_components=n_components,
            kernel="poly",
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            scale=scale,
        ).fit(X, y)

        K_train = pairwise_kernels(
            X,
            X,
            metric="poly",
            gamma=gamma,
            degree=degree,
            coef0=coef0,
        )
        K_test = pairwise_kernels(
            X_test,
            X,
            metric="poly",
            gamma=gamma,
            degree=degree,
            coef0=coef0,
        )

        centerer = KernelCenterer().fit(K_train)
        K_train_c = centerer.transform(K_train)
        K_test_c = centerer.transform(K_test)

        reference_model = PLSRegression(
            n_components=n_components,
            scale=scale,
        ).fit(K_train_c, y.reshape(-1, 1))

        y_hat_reference = reference_model.predict(K_test_c)

        # Act
        y_hat = model.predict(X_test)

        # Assert
        np.testing.assert_array_almost_equal(
            y_hat,
            y_hat_reference,
            decimal=10,
        )


class TestParameterValidation:
    """Tests for invalid estimator parameters."""

    @pytest.mark.parametrize("gamma", [0, -1.0])
    def test_invalid_gamma_raises_error(self, sample_data, gamma) -> None:
        """Verifies that non-positive gamma values are rejected."""
        # Arrange
        X, y, _ = sample_data

        # Act & Assert
        with pytest.raises(ValueError, match="gamma"):
            KernelPLS(kernel="rbf", gamma=gamma).fit(X, y)

    @pytest.mark.parametrize("degree", [0, -1])
    def test_invalid_degree_raises_error(self, sample_data, degree) -> None:
        """Verifies that invalid polynomial degrees are rejected."""
        # Arrange
        X, y, _ = sample_data

        # Act & Assert
        with pytest.raises(ValueError, match="degree"):
            KernelPLS(kernel="poly", degree=degree).fit(X, y)

    @pytest.mark.parametrize("scale", ["yes", 1])
    def test_invalid_scale_raises_error(self, sample_data, scale) -> None:
        """Verifies that scale must be boolean."""
        # Arrange
        X, y, _ = sample_data

        # Act & Assert
        with pytest.raises(ValueError, match="scale"):
            KernelPLS(scale=scale).fit(X, y)

    @pytest.mark.parametrize("scale_X", ["yes", 1])
    def test_invalid_scale_X_raises_error(self, sample_data, scale_X) -> None:
        """Verifies that scale_X must be boolean."""
        # Arrange
        X, y, _ = sample_data

        # Act & Assert
        with pytest.raises(ValueError, match="scale_X"):
            KernelPLS(scale_X=scale_X).fit(X, y)


class TestScore:
    """Tests for regressor score behavior."""

    def test_score_returns_finite_float(self, sample_data) -> None:
        """Verifies that score returns a finite scalar."""
        # Arrange
        X, y, _ = sample_data
        model = KernelPLS(n_components=2).fit(X, y)

        # Act
        score = model.score(X, y)

        # Assert
        assert isinstance(score, float)
        assert np.isfinite(score)

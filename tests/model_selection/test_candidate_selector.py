import numpy as np
import pytest
from sklearn.linear_model import Ridge

from chemotools.model_selection import BaseFittedModel, CandidateSelector


# -- Fixtures ------------------------------------------------------------------


@pytest.fixture
def fitted_selector(dummy_data_loader):
    """Return a fitted CandidateSelector for testing."""
    X, y = dummy_data_loader
    selector = CandidateSelector(
        estimator=Ridge(random_state=0),
        param_grid={"alpha": [0.1, 1.0, 10.0]},
        cv=3,
        scoring="neg_root_mean_squared_error",
        return_train_score=True,
        n_jobs=1,
    )
    selector.fit(X, y)
    return selector


@pytest.fixture
def unfitted_selector():
    """Return an unfitted CandidateSelector for testing."""
    return CandidateSelector(
        estimator=Ridge(random_state=0),
        param_grid={"alpha": [0.1, 1.0, 10.0]},
        cv=3,
        scoring="neg_root_mean_squared_error",
        return_train_score=True,
        n_jobs=1,
    )


# -- Test instantiation --------------------------------------------------------


def test_instantiation_with_valid_params():
    # Arrange
    estimator = Ridge(random_state=0)
    param_grid = {"alpha": [0.1, 1.0, 10.0]}

    # Act
    selector = CandidateSelector(
        estimator=estimator,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=5,
    )

    # Assert
    assert selector.estimator is estimator
    assert selector.param_grid == param_grid
    assert selector.scoring == "neg_root_mean_squared_error"
    assert selector.cv == 5


def test_instantiation_with_default_params():
    # Arrange
    estimator = Ridge(random_state=0)
    param_grid = {"alpha": [1.0]}

    # Act
    selector = CandidateSelector(estimator=estimator, param_grid=param_grid)

    # Assert
    assert selector.cv == 5
    assert selector.n_jobs is None
    assert selector.verbose == 0
    assert selector.return_train_score is True


# -- Test fit ------------------------------------------------------------------


def test_fit_sets_attributes(dummy_data_loader):
    # Arrange
    X, y = dummy_data_loader
    selector = CandidateSelector(
        estimator=Ridge(random_state=0),
        param_grid={"alpha": [0.1, 1.0, 10.0]},
        cv=3,
        scoring="neg_root_mean_squared_error",
        return_train_score=True,
        n_jobs=1,
    )

    # Act
    selector.fit(X, y)

    # Assert
    assert hasattr(selector, "cv_results_")
    assert hasattr(selector, "best_estimator_")
    assert hasattr(selector, "best_params_")
    assert hasattr(selector, "best_score_")
    assert hasattr(selector, "candidates_")
    assert len(selector.candidates_) == 3  # 3 alpha values


def test_fit_returns_self(dummy_data_loader):
    # Arrange
    X, y = dummy_data_loader
    selector = CandidateSelector(
        estimator=Ridge(random_state=0),
        param_grid={"alpha": [0.1, 1.0, 10.0]},
        cv=3,
        scoring="neg_root_mean_squared_error",
    )

    # Act
    result = selector.fit(X, y)

    # Assert
    assert result is selector


# -- Test get_candidates -------------------------------------------------------


def test_get_candidates_returns_all(fitted_selector):
    # Arrange & Act
    candidates = fitted_selector.get_candidates()

    # Assert
    assert len(candidates) == 3
    assert all(isinstance(c, BaseFittedModel) for c in candidates)


def test_get_candidates_returns_n(fitted_selector):
    # Arrange & Act
    candidates = fitted_selector.get_candidates(n=2)

    # Assert
    assert len(candidates) == 2
    assert all(isinstance(c, BaseFittedModel) for c in candidates)


def test_get_candidates_unfitted_raises_error(unfitted_selector):
    # Arrange & Act & Assert
    with pytest.raises(Exception):
        unfitted_selector.get_candidates()


# -- Test get_candidate --------------------------------------------------------


def test_get_candidate_returns_correct_rank(fitted_selector):
    # Arrange & Act
    candidate = fitted_selector.get_candidate(rank=1)

    # Assert
    assert isinstance(candidate, BaseFittedModel)
    assert candidate.rank == 1


def test_get_candidate_invalid_rank_raises_error(fitted_selector):
    # Arrange & Act & Assert
    with pytest.raises(ValueError, match="No candidate with rank"):
        fitted_selector.get_candidate(rank=999)


def test_get_candidate_unfitted_raises_error(unfitted_selector):
    # Arrange & Act & Assert
    with pytest.raises(Exception):
        unfitted_selector.get_candidate(rank=1)


# -- Test filter_candidates ----------------------------------------------------


def test_filter_candidates_by_rmse_ratio(fitted_selector):
    # Arrange & Act
    filtered = fitted_selector.filter_candidates(
        metric="rmse_ratio", threshold=2.0, mode="<="
    )

    # Assert
    assert isinstance(filtered, list)
    assert all(isinstance(c, BaseFittedModel) for c in filtered)
    assert all(c.rmse_ratio <= 2.0 for c in filtered if c.rmse_ratio is not None)


def test_filter_candidates_by_variance(fitted_selector):
    # Arrange & Act
    filtered = fitted_selector.filter_candidates(
        metric="variance", threshold=1.0, mode="<="
    )

    # Assert
    assert isinstance(filtered, list)
    assert all(c.variance <= 1.0 for c in filtered if c.variance is not None)


def test_filter_candidates_mode_ge(fitted_selector):
    # Arrange & Act
    filtered = fitted_selector.filter_candidates(
        metric="rmse_ratio", threshold=0.5, mode=">="
    )

    # Assert
    assert isinstance(filtered, list)
    assert all(c.rmse_ratio >= 0.5 for c in filtered if c.rmse_ratio is not None)


def test_filter_candidates_mode_lt(fitted_selector):
    # Arrange & Act
    filtered = fitted_selector.filter_candidates(
        metric="rmse_ratio", threshold=10.0, mode="<"
    )

    # Assert
    assert isinstance(filtered, list)
    assert all(c.rmse_ratio < 10.0 for c in filtered if c.rmse_ratio is not None)


def test_filter_candidates_mode_gt(fitted_selector):
    # Arrange & Act
    filtered = fitted_selector.filter_candidates(
        metric="rmse_ratio", threshold=0.0, mode=">"
    )

    # Assert
    assert isinstance(filtered, list)
    assert all(c.rmse_ratio > 0.0 for c in filtered if c.rmse_ratio is not None)


def test_filter_candidates_invalid_mode_raises_error(fitted_selector):
    # Arrange & Act & Assert
    with pytest.raises(ValueError, match="mode must be one of"):
        fitted_selector.filter_candidates(metric="rmse_ratio", threshold=1.0, mode="!=")


def test_filter_candidates_unfitted_raises_error(unfitted_selector):
    # Arrange & Act & Assert
    with pytest.raises(Exception):
        unfitted_selector.filter_candidates()


# -- Test predict and score ----------------------------------------------------


def test_predict_returns_array(fitted_selector, dummy_data_loader):
    # Arrange
    X, _ = dummy_data_loader

    # Act
    predictions = fitted_selector.predict(X)

    # Assert
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape[0] == X.shape[0]


def test_predict_unfitted_raises_error(unfitted_selector, dummy_data_loader):
    # Arrange
    X, _ = dummy_data_loader

    # Act & Assert
    with pytest.raises(Exception):
        unfitted_selector.predict(X)


def test_score_returns_float(fitted_selector, dummy_data_loader):
    # Arrange
    X, y = dummy_data_loader

    # Act
    score = fitted_selector.score(X, y)

    # Assert
    assert isinstance(score, float)


def test_score_unfitted_raises_error(unfitted_selector, dummy_data_loader):
    # Arrange
    X, y = dummy_data_loader

    # Act & Assert
    with pytest.raises(Exception):
        unfitted_selector.score(X, y)


# -- Test __len__ and __iter__ -------------------------------------------------


def test_len_returns_candidate_count(fitted_selector):
    # Arrange & Act
    length = len(fitted_selector)

    # Assert
    assert length == 3


def test_len_unfitted_raises_error(unfitted_selector):
    # Arrange & Act & Assert
    with pytest.raises(Exception):
        len(unfitted_selector)


def test_iter_yields_candidates(fitted_selector):
    # Arrange & Act
    candidates = list(fitted_selector)

    # Assert
    assert len(candidates) == 3
    assert all(isinstance(c, BaseFittedModel) for c in candidates)


def test_iter_unfitted_raises_error(unfitted_selector):
    # Arrange & Act & Assert
    with pytest.raises(Exception):
        list(unfitted_selector)


# -- Test summary --------------------------------------------------------------


def test_summary_returns_string(fitted_selector):
    # Arrange & Act
    summary = fitted_selector.summary()

    # Assert
    assert isinstance(summary, str)
    assert "CandidateSelector Summary" in summary
    assert "Total candidates:" in summary
    assert "Best score:" in summary


def test_summary_with_n_parameter(fitted_selector):
    # Arrange & Act
    summary = fitted_selector.summary(n=2)

    # Assert
    assert isinstance(summary, str)
    assert "Top 2 Candidates:" in summary


def test_summary_unfitted_raises_error(unfitted_selector):
    # Arrange & Act & Assert
    with pytest.raises(Exception):
        unfitted_selector.summary()


# -- Test to_dataframe ---------------------------------------------------------


def test_to_dataframe_returns_dataframe(fitted_selector):
    # Arrange & Act
    df = fitted_selector.to_dataframe()

    # Assert
    import pandas as pd

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "rank" in df.columns
    assert "rmsecv" in df.columns
    assert "rmse_ratio" in df.columns
    assert "param_alpha" in df.columns


def test_to_dataframe_has_correct_columns(fitted_selector):
    # Arrange & Act
    df = fitted_selector.to_dataframe()

    # Assert
    expected_columns = [
        "rank",
        "params",
        "mean_test_score",
        "std_test_score",
        "variance",
        "mean_train_score",
        "rmsecv",
        "rmse_train",
        "rmse_ratio",
    ]
    for col in expected_columns:
        assert col in df.columns


def test_to_dataframe_unfitted_raises_error(unfitted_selector):
    # Arrange & Act & Assert
    with pytest.raises(Exception):
        unfitted_selector.to_dataframe()


# -- Test RMSE metrics ---------------------------------------------------------


def test_candidates_have_rmse_metrics(fitted_selector):
    # Arrange & Act
    candidates = fitted_selector.get_candidates()

    # Assert
    for c in candidates:
        assert c.rmsecv is not None
        assert c.rmse_train is not None
        assert c.rmse_ratio is not None
        assert c.rmsecv > 0
        assert c.rmse_train > 0
        assert c.rmse_ratio > 0


def test_candidates_sorted_by_rank(fitted_selector):
    # Arrange & Act
    candidates = fitted_selector.get_candidates()

    # Assert
    ranks = [c.rank for c in candidates]
    assert ranks == sorted(ranks)


# -- Test clone_estimator from candidate ---------------------------------------


def test_candidate_clone_estimator_works(fitted_selector, dummy_data_loader):
    # Arrange
    X, y = dummy_data_loader
    best_candidate = fitted_selector.get_candidate(rank=1)

    # Act
    cloned = best_candidate.clone_estimator()
    cloned.fit(X, y)
    predictions = cloned.predict(X[:3])

    # Assert
    assert predictions.shape == (3,)


# -- Test plot methods ---------------------------------------------------------


def test_plot_cv_metrics_returns_axes(fitted_selector):
    # Arrange & Act
    ax = fitted_selector.plot_cv_metrics()

    # Assert

    assert ax is not None
    assert hasattr(ax, "get_xlabel")
    assert ax.get_xlabel() == "RMSECV"
    assert ax.get_ylabel() == "RMSECV / RMSEC"


def test_plot_cv_metrics_with_color_by(fitted_selector):
    # Arrange & Act
    ax = fitted_selector.plot_cv_metrics(color_by="alpha")

    # Assert
    assert ax is not None


def test_plot_cv_metrics_with_custom_title(fitted_selector):
    # Arrange & Act
    ax = fitted_selector.plot_cv_metrics(title="Custom Title")

    # Assert
    assert ax.get_title() == "Custom Title"


def test_plot_cv_metrics_unfitted_raises_error(unfitted_selector):
    # Arrange & Act & Assert
    with pytest.raises(Exception):
        unfitted_selector.plot_cv_metrics()


def test_plot_score_vs_variance_returns_axes(fitted_selector):
    # Arrange & Act
    ax = fitted_selector.plot_score_vs_variance()

    # Assert
    assert ax is not None
    assert ax.get_xlabel() == "Variance"
    assert ax.get_ylabel() == "Mean Test Score"


def test_plot_score_vs_variance_with_custom_title(fitted_selector):
    # Arrange & Act
    ax = fitted_selector.plot_score_vs_variance(title="Custom Title")

    # Assert
    assert ax.get_title() == "Custom Title"


def test_plot_score_vs_variance_unfitted_raises_error(unfitted_selector):
    # Arrange & Act & Assert
    with pytest.raises(Exception):
        unfitted_selector.plot_score_vs_variance()


# -- Test best_estimator_ usage ------------------------------------------------


def test_best_estimator_is_fitted(fitted_selector, dummy_data_loader):
    # Arrange
    X, _ = dummy_data_loader

    # Act
    predictions = fitted_selector.best_estimator_.predict(X)

    # Assert
    assert predictions.shape[0] == X.shape[0]


def test_best_params_matches_best_candidate(fitted_selector):
    # Arrange & Act
    best_candidate = fitted_selector.get_candidate(rank=1)

    # Assert
    assert fitted_selector.best_params_ == best_candidate.params

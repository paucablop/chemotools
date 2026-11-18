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


# -- Test instantiation and fitting --------------------------------------------


def test_instantiation_and_fit(dummy_data_loader):
    """Test that CandidateSelector can be instantiated with valid params and fitted."""
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
    result = selector.fit(X, y)

    # Assert
    assert result is selector
    assert hasattr(selector, "cv_results_")
    assert hasattr(selector, "best_estimator_")
    assert hasattr(selector, "best_params_")
    assert hasattr(selector, "best_score_")
    assert hasattr(selector, "candidates_")
    assert len(selector.candidates_) == 3


# -- Test get_candidates and get_candidate -------------------------------------


def test_get_candidates(fitted_selector):
    """Test retrieving all candidates and specific candidates by rank."""
    # Act
    all_candidates = fitted_selector.get_candidates()
    top_2 = fitted_selector.get_candidates(n=2)
    best = fitted_selector.get_candidate(rank=1)

    # Assert
    assert len(all_candidates) == 3
    assert all(isinstance(c, BaseFittedModel) for c in all_candidates)
    assert len(top_2) == 2
    assert isinstance(best, BaseFittedModel)
    assert best.rank == 1
    # Verify candidates are sorted by rank
    ranks = [c.rank for c in all_candidates]
    assert ranks == sorted(ranks)


def test_get_candidate_invalid_rank_raises_error(fitted_selector):
    """Test that requesting an invalid rank raises an error."""
    with pytest.raises(ValueError, match="No candidate with rank"):
        fitted_selector.get_candidate(rank=999)


# -- Test filter_candidates ----------------------------------------------------


def test_filter_candidates(fitted_selector):
    """Test filtering candidates by different metrics and modes."""
    # Test filter by rmse_ratio with <= mode
    filtered_le = fitted_selector.filter_candidates(
        metric="rmse_ratio", threshold=2.0, mode="<="
    )
    assert isinstance(filtered_le, list)
    assert all(isinstance(c, BaseFittedModel) for c in filtered_le)

    # Test filter with >= mode
    filtered_ge = fitted_selector.filter_candidates(
        metric="rmse_ratio", threshold=0.5, mode=">="
    )
    assert all(c.rmse_ratio >= 0.5 for c in filtered_ge if c.rmse_ratio is not None)


def test_filter_candidates_invalid_mode_raises_error(fitted_selector):
    """Test that an invalid filter mode raises an error."""
    with pytest.raises(ValueError, match="mode must be one of"):
        fitted_selector.filter_candidates(metric="rmse_ratio", threshold=1.0, mode="!=")


# -- Test predict and score ----------------------------------------------------


def test_predict_and_score(fitted_selector, dummy_data_loader):
    """Test prediction and scoring methods."""
    X, y = dummy_data_loader

    # Act
    predictions = fitted_selector.predict(X)
    score = fitted_selector.score(X, y)

    # Assert
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape[0] == X.shape[0]
    assert isinstance(score, float)


# -- Test __len__ and __iter__ -------------------------------------------------


def test_len_and_iter(fitted_selector):
    """Test length and iteration over candidates."""
    # Act
    length = len(fitted_selector)
    candidates = list(fitted_selector)

    # Assert
    assert length == 3
    assert len(candidates) == 3
    assert all(isinstance(c, BaseFittedModel) for c in candidates)


# -- Test summary and to_dataframe ---------------------------------------------


def test_summary(fitted_selector):
    """Test summary method returns expected format."""
    # Act
    summary = fitted_selector.summary()
    summary_n = fitted_selector.summary(n=2)

    # Assert
    assert isinstance(summary, str)
    assert "CandidateSelector Summary" in summary
    assert "Total candidates:" in summary
    assert "Best score:" in summary
    assert "Top 2 Candidates:" in summary_n


# -- Test RMSE metrics and candidate properties --------------------------------


def test_candidates_have_rmse_metrics(fitted_selector):
    """Test that candidates have valid RMSE metrics."""
    candidates = fitted_selector.get_candidates()

    for c in candidates:
        assert c.rmsecv is not None and c.rmsecv > 0
        assert c.rmse_train is not None and c.rmse_train > 0
        assert c.rmse_ratio is not None and c.rmse_ratio > 0


def test_candidate_clone_estimator(fitted_selector, dummy_data_loader):
    """Test cloning an estimator from a candidate."""
    X, y = dummy_data_loader
    best_candidate = fitted_selector.get_candidate(rank=1)

    # Act
    cloned = best_candidate.clone_estimator()
    cloned.fit(X, y)
    predictions = cloned.predict(X[:3])

    # Assert
    assert predictions.shape == (3,)


# -- Test plot methods ---------------------------------------------------------


def test_plot_cv_metrics(fitted_selector):
    """Test plot_cv_metrics returns valid axes."""
    ax = fitted_selector.plot_cv_metrics()

    assert ax is not None
    assert hasattr(ax, "get_xlabel")
    assert ax.get_xlabel() == "RMSECV"
    assert ax.get_ylabel() == "RMSECV / RMSEC"


def test_plot_score_vs_variance(fitted_selector):
    """Test plot_score_vs_variance returns valid axes."""
    ax = fitted_selector.plot_score_vs_variance()

    assert ax is not None
    assert ax.get_xlabel() == "Variance"
    assert ax.get_ylabel() == "Mean Test Score"


# -- Test unfitted selector raises errors --------------------------------------


def test_unfitted_selector_raises_errors(unfitted_selector, dummy_data_loader):
    """Test that unfitted selector raises errors for methods requiring fit."""
    X, y = dummy_data_loader

    with pytest.raises(Exception):
        unfitted_selector.get_candidates()

    with pytest.raises(Exception):
        unfitted_selector.get_candidate(rank=1)

    with pytest.raises(Exception):
        unfitted_selector.filter_candidates()

    with pytest.raises(Exception):
        unfitted_selector.predict(X)

    with pytest.raises(Exception):
        unfitted_selector.score(X, y)

    with pytest.raises(Exception):
        len(unfitted_selector)

    with pytest.raises(Exception):
        list(unfitted_selector)

    with pytest.raises(Exception):
        unfitted_selector.summary()

    with pytest.raises(Exception):
        unfitted_selector.to_dataframe()

    with pytest.raises(Exception):
        unfitted_selector.plot_cv_metrics()

    with pytest.raises(Exception):
        unfitted_selector.plot_score_vs_variance()


# -- Test best_estimator_ usage ------------------------------------------------


def test_best_estimator_and_params(fitted_selector, dummy_data_loader):
    """Test that best_estimator_ is fitted and best_params_ matches best candidate."""
    X, _ = dummy_data_loader
    best_candidate = fitted_selector.get_candidate(rank=1)

    # Act
    predictions = fitted_selector.best_estimator_.predict(X)

    # Assert
    assert predictions.shape[0] == X.shape[0]
    assert fitted_selector.best_params_ == best_candidate.params

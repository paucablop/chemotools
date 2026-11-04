"""
The :mod:`chemotools.model_selection._fitted_model` module implements
a container for storing candidate model information during model selection.
"""

# Authors: Nusret Emirhan Salli <nusret.emirhan.salli@gmail.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Union

from sklearn.base import BaseEstimator, clone

__all__ = ["BaseFittedModel"]

ScoringType = Optional[Union[str, Callable[..., float]]]


@dataclass(slots=True)
class BaseFittedModel:
    """Container for a single candidate model's results.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator instance (scikit-learn compatible).
    params : dict
        Parameters used when fitting the estimator.
    rank : int, optional
        Rank assigned by model selection (1 = best).
    mean_test_score : float, optional
        Mean cross-validation score.
    std_test_score : float, optional
        Standard deviation of cross-validation score.
    mean_train_score : float, optional
        Mean training score.
    scoring : str or callable, optional
        Scoring function used.
    cv_results_index : int, optional
        Index in ``cv_results_``.
    rmsecv : float, optional
        Root-mean-square error from cross-validation.
    rmse_train : float, optional
        Training RMSE.
    rmse_ratio : float, optional
        Ratio of RMSECV to training RMSE (overfitting indicator).
    """

    estimator: BaseEstimator
    params: Dict[str, Any]
    rank: Optional[int] = None
    mean_test_score: Optional[float] = None
    std_test_score: Optional[float] = None
    mean_train_score: Optional[float] = None
    scoring: ScoringType = None
    cv_results_index: Optional[int] = None
    rmsecv: Optional[float] = None
    rmse_train: Optional[float] = None
    rmse_ratio: Optional[float] = None

    def __post_init__(self) -> None:
        if self.rank is not None and self.rank < 1:
            raise ValueError("rank must be a positive integer.")
        if not isinstance(self.params, dict):
            raise TypeError("params must be a dictionary.")

    @staticmethod
    def _to_native(value: Any) -> Any:
        """Convert numpy scalars to native Python types."""
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return value
        return value

    @classmethod
    def from_cv_results(
        cls,
        estimator: BaseEstimator,
        cv_results: Dict[str, Sequence[Any]],
        index: int,
        *,
        scoring: ScoringType = None,
    ) -> BaseFittedModel:
        """Create a ``BaseFittedModel`` from a ``GridSearchCV`` result entry."""
        params = cv_results["params"][index]
        estimator_clone = clone(estimator)
        estimator_clone.set_params(**params)

        rank_values = cv_results.get("rank_test_score")
        rank = (
            int(cls._to_native(rank_values[index])) if rank_values is not None else None
        )

        mean_test = cls._to_native(cv_results["mean_test_score"][index])
        std_test = cls._to_native(cv_results["std_test_score"][index])
        mean_train = (
            cls._to_native(cv_results["mean_train_score"][index])
            if "mean_train_score" in cv_results
            else None
        )

        # Calculate RMSE metrics if using neg_root_mean_squared_error
        rmsecv, rmse_train_val, rmse_ratio = None, None, None
        if isinstance(scoring, str) and "neg_root_mean_squared_error" in scoring:
            rmsecv = -mean_test
            if mean_train is not None:
                rmse_train_val = -mean_train
                if rmse_train_val != 0:
                    rmse_ratio = rmsecv / rmse_train_val

        return cls(
            estimator=estimator_clone,
            params=params,
            rank=rank,
            mean_test_score=mean_test,
            std_test_score=std_test,
            mean_train_score=mean_train,
            scoring=scoring,
            cv_results_index=index,
            rmsecv=rmsecv,
            rmse_train=rmse_train_val,
            rmse_ratio=rmse_ratio,
        )

    @property
    def variance(self) -> Optional[float]:
        """Variance of the test score (std_test_score²)."""
        return self.std_test_score**2 if self.std_test_score is not None else None

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of the candidate."""
        return {
            "rank": self.rank,
            "params": self.params,
            "mean_test_score": self.mean_test_score,
            "std_test_score": self.std_test_score,
            "variance": self.variance,
            "mean_train_score": self.mean_train_score,
            "rmsecv": self.rmsecv,
            "rmse_train": self.rmse_train,
            "rmse_ratio": self.rmse_ratio,
        }

    def clone_estimator(self) -> BaseEstimator:
        """Return a fresh estimator with the stored parameters."""
        cloned = clone(self.estimator)
        cloned.set_params(**self.params)
        return cloned

    def __repr__(self) -> str:
        """Return a concise string representation."""
        parts = [f"Rank {self.rank}"]
        if self.rmsecv is not None:
            parts.append(f"RMSECV={self.rmsecv:.4f}")
        if self.rmse_ratio is not None:
            parts.append(f"ratio={self.rmse_ratio:.3f}")
        if self.variance is not None:
            parts.append(f"var={self.variance:.2e}")
        return f"BaseFittedModel({', '.join(parts)})"

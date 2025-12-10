from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from .regression import RegressionMetrics


@dataclass
class InspectorSummary:
    """Base class for all inspector summaries."""

    model_type: str
    has_preprocessing: bool
    nr_features: int
    nr_samples: Dict[str, int]
    preprocessing_steps: List[Dict[str, Any]]

    def to_dict(self):
        return asdict(self)


@dataclass
class LatentSummary:
    nr_components: int
    hotelling_t2_limit: float
    q_residuals_limit: float


@dataclass
class RegressionSummary:
    regression: Dict[str, RegressionMetrics]


@dataclass
class PCASummary(InspectorSummary, LatentSummary):
    """Summary for PCA models."""

    explained_variance_ratio: List[float]
    cumulative_variance: List[float]
    pc_variances: Dict[str, float]
    total_variance: float
    variance_thresholds: Dict[str, Dict[str, Any]]


@dataclass
class PLSRegressionSummary(InspectorSummary, LatentSummary, RegressionSummary):
    """Summary for PLS Regression models."""

    explained_x_variance_ratio: Optional[List[float]] = None
    total_x_variance: Optional[float] = None
    explained_y_variance_ratio: Optional[List[float]] = None
    total_y_variance: Optional[float] = None

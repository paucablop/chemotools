from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from .latent import LatentSummary
from .regression import RegressionMetrics


@dataclass
class BaseSummary:
    model_type: str
    has_preprocessing: bool
    nr_features: int
    nr_samples: Dict[str, int]
    preprocessing_steps: List[Dict[str, Any]]


@dataclass
class PCASummary:
    explained_variance_ratio: Any
    cumulative_variance: Any
    pc_variances: Dict[str, float]
    total_variance: float
    variance_thresholds: Dict[str, Dict[str, Any]]


@dataclass
class PLSVarianceSummary:
    explained_x_variance_ratio: Optional[Any] = None
    total_x_variance: Optional[float] = None
    explained_y_variance_ratio: Optional[Any] = None
    total_y_variance: Optional[float] = None


@dataclass
class InspectorSummary:
    base: BaseSummary
    latent: Optional[LatentSummary] = None
    regression: Optional[Dict[str, RegressionMetrics]] = None
    pca: Optional[PCASummary] = None
    pls_variance: Optional[PLSVarianceSummary] = None

    def to_dict(self):
        return asdict(self)

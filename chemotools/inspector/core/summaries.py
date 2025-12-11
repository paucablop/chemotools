from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any


@dataclass
class RegressionMetrics:
    rmse: float
    r2: float
    bias: float


@dataclass(kw_only=True)
class InspectorSummary:
    """Base class for all inspector summaries."""

    model_type: str
    has_preprocessing: bool
    n_features: int
    n_samples: Dict[str, int]
    preprocessing_steps: List[Dict[str, Any]]

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(kw_only=True)
class LatentSummary:
    n_components: int
    hotelling_t2_limit: float
    q_residuals_limit: float


@dataclass(kw_only=True)
class RegressionSummary:
    train: RegressionMetrics
    test: Optional[RegressionMetrics] = None
    val: Optional[RegressionMetrics] = None

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}

    @property
    def metrics(self) -> Dict[str, Dict[str, float]]:
        """Get regression metrics as a dictionary suitable for pandas DataFrame.

        Returns a dictionary where keys are metric names (e.g. 'rmse', 'r2')
        and values are dictionaries mapping dataset names to metric values.
        This structure results in a DataFrame where:
        - Columns are metrics (RMSE, R2, Bias)
        - Rows are datasets (train, test, val)
        """
        # 1. Collect data by dataset
        by_dataset = {}
        for dataset in ["train", "test", "val"]:
            obj = getattr(self, dataset)
            if obj is not None:
                by_dataset[dataset] = asdict(obj)

        if not by_dataset:
            return {}

        # 2. Invert to be by metric (for DataFrame columns)
        # Assuming all datasets have the same metrics (defined by RegressionMetrics)
        metric_names = next(iter(by_dataset.values())).keys()

        return {
            metric: {ds: values[metric] for ds, values in by_dataset.items()}
            for metric in metric_names
        }


@dataclass(kw_only=True)
class PCASummary(InspectorSummary, LatentSummary):
    """Summary for PCA models."""

    explained_variance_ratio: List[float]
    cumulative_variance: List[float]
    pc_variances: Dict[str, float]
    total_variance: float
    variance_thresholds: Dict[str, Dict[str, Any]]


@dataclass(kw_only=True)
class PLSRegressionSummary(InspectorSummary, LatentSummary, RegressionSummary):
    """Summary for PLS Regression models."""

    explained_x_variance_ratio: Optional[List[float]] = None
    total_x_variance: Optional[float] = None
    explained_y_variance_ratio: Optional[List[float]] = None
    total_y_variance: Optional[float] = None

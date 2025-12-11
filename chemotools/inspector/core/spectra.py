"""Spectra visualization mixin for inspector implementations."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, TYPE_CHECKING, Union, Literal

import numpy as np

from chemotools.inspector.helpers._spectra import (
    create_spectra_plots_single_dataset,
    create_spectra_plots_multi_dataset,
)
from .utils import normalize_datasets, get_xlabel_for_features

if TYPE_CHECKING:  # pragma: no cover
    from typing import Protocol

    from matplotlib.figure import Figure
    from sklearn.pipeline import Pipeline

    class _SpectraInspectorProto(Protocol):
        @property
        def transformer(self) -> Optional[Pipeline]:  # pragma: no cover
            ...

        @property
        def x_axis(self) -> np.ndarray:  # pragma: no cover
            ...

        @property
        def feature_names(self) -> Optional[np.ndarray]:  # pragma: no cover
            ...

        def _get_raw_data(
            self, dataset: str
        ) -> Tuple[np.ndarray, Optional[np.ndarray]]:  # pragma: no cover
            ...

        def _get_preprocessed_data(
            self, dataset: str
        ) -> np.ndarray:  # pragma: no cover
            ...

        def _get_preprocessed_x_axis(self) -> np.ndarray:  # pragma: no cover
            ...


class SpectraMixin:
    """Mixin providing reusable spectra visualization methods.

    This mixin provides `inspect_spectra()` functionality that can be used
    by any inspector that has preprocessing pipelines. It eliminates code
    duplication between PCAInspector and PLSRegressionInspector.

    Requirements for the host class:
    - `transformer` property returning Optional[Pipeline]
    - `x_axis` property returning np.ndarray
    - `feature_names` property returning Optional[np.ndarray]
    - `_get_raw_data(dataset)` method
    - `_get_preprocessed_data(dataset)` method
    - `_get_preprocessed_x_axis()` method
    """

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------
    def _spectra_inspector(self) -> "_SpectraInspectorProto":
        return self  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def inspect_spectra(
        self,
        dataset: Union[str, Sequence[str]] = "train",
        color_by: Optional[Union[str, Dict[str, np.ndarray]]] = "y",
        xlim: Optional[Tuple[float, float]] = None,
        figsize: Tuple[float, float] = (12, 5),
        color_mode: Optional[Literal["continuous", "categorical"]] = None,
    ) -> Dict[str, "Figure"]:
        """Create independent plots comparing raw and preprocessed spectra.

        Only available if model is a Pipeline with preprocessing steps.
        Creates two separate figure windows: one for raw spectra and one
        for preprocessed spectra. When multiple datasets are provided,
        all spectra are plotted on the same figure with colors indicating
        the dataset.

        Parameters
        ----------
        dataset : Union[str, Sequence[str]], default='train'
            Dataset(s) to visualize. Can be a single dataset name or a sequence
            of dataset names (e.g., ["train", "test"]).
        color_by : str or dict, default='y'
            Coloring specification:
            - 'y': Color by y values (if available)
            - 'sample_index': Color by sample index
            - dict: Map dataset names to color arrays
            Ignored when multiple datasets are provided (colors by dataset instead).
        xlim : tuple of float, optional
            X-axis limits for zooming into spectral regions
        figsize : tuple of float, default=(12, 5)
            Figure size for each plot (width, height) in inches
        color_mode : Literal["continuous", "categorical"], default="continuous"
            Mode for coloring points.

        Returns
        -------
        figures : dict
            Dictionary containing both figures with keys:
            'raw_spectra', 'preprocessed_spectra'

        Raises
        ------
        ValueError
            If no preprocessing pipeline is available

        Examples
        --------
        >>> inspector = PCAInspector(pipeline, X_train, y_train)
        >>> # Single dataset
        >>> figs = inspector.inspect_spectra()  # Creates 2 separate plots
        >>> figs = inspector.inspect_spectra(xlim=(1000, 1800))  # Zoom in
        >>> # Multiple datasets comparison
        >>> inspector.X_test = X_test
        >>> figs = inspector.inspect_spectra(dataset=["train", "test"])
        >>> figs['raw_spectra'].savefig('raw_spectra_comparison.png')
        """
        inspector = self._spectra_inspector()

        if inspector.transformer is None:
            raise ValueError(
                "Spectra inspection requires a preprocessing pipeline. "
                "Model must be a Pipeline with preprocessing steps."
            )

        # Normalize dataset to always be a list
        datasets = normalize_datasets(dataset)
        is_multi_dataset = len(datasets) > 1

        # Determine xlabel based on feature_names
        xlabel = get_xlabel_for_features(inspector.feature_names is not None)

        # Get preprocessed x_axis (may be subset if feature selection)
        preprocessed_x_axis = inspector._get_preprocessed_x_axis()

        if is_multi_dataset:
            # Multiple datasets: plot all on same figure, color by dataset
            raw_data = {}
            preprocessed_data = {}
            for ds in datasets:
                X_raw, _ = inspector._get_raw_data(ds)
                X_preprocessed = inspector._get_preprocessed_data(ds)
                raw_data[ds] = X_raw
                preprocessed_data[ds] = X_preprocessed

            figures = create_spectra_plots_multi_dataset(
                raw_data=raw_data,
                preprocessed_data=preprocessed_data,
                x_axis=inspector.x_axis,
                preprocessed_x_axis=preprocessed_x_axis,
                xlabel=xlabel,
                xlim=xlim,
                figsize=figsize,
                color_mode=color_mode,
            )
        else:
            # Single dataset
            ds = datasets[0]
            X_raw, y = inspector._get_raw_data(ds)
            X_preprocessed = inspector._get_preprocessed_data(ds)

            figures = create_spectra_plots_single_dataset(
                X_raw=X_raw,
                X_preprocessed=X_preprocessed,
                y=y,
                x_axis=inspector.x_axis,
                preprocessed_x_axis=preprocessed_x_axis,
                dataset_name=ds,
                color_by=color_by,
                xlabel=xlabel,
                xlim=xlim,
                figsize=figsize,
                color_mode=color_mode,
            )

        return figures

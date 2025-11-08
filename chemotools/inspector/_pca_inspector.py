"""PCA Inspector for model diagnostics and visualization."""

from __future__ import annotations
from typing import Optional, Union, Sequence, Tuple, Dict, TYPE_CHECKING
import numpy as np
from sklearn.decomposition._base import _BasePCA
from sklearn.pipeline import Pipeline

if TYPE_CHECKING:
    import matplotlib.figure

from chemotools.plotting import (
    SpectrumPlot,
    ScoresPlot,
    LoadingsPlot,
    ExplainedVariancePlot,
)
from chemotools.plotting._utilities import annotate_points
from chemotools.plotting._styles import DATASET_COLORS
from ._validate import _validate_and_extract_model, _validate_datasets_consistency


class PCAInspector:
    """Inspector for PCA model diagnostics and visualization.

    This class provides a unified interface for inspecting PCA models by creating
    multiple independent diagnostic plots. Instead of complex dashboards with many
    subplots, each method produces several separate figure windows that are easier
    to customize, save, and interact with individually.

    The inspector provides convenience methods that create multiple independent plots:
    - inspect(): Creates all diagnostic plots (scores, loadings, explained variance)
    - inspect_spectra(): Creates raw and preprocessed spectra plots (if preprocessing exists)

    Parameters
    ----------
    model : _BasePCA or Pipeline
        Fitted PCA model or pipeline ending with PCA
    X_train : array-like of shape (n_samples, n_features)
        Training data
    y_train : array-like of shape (n_samples,), optional
        Training labels/targets (for coloring plots)
    X_test : array-like of shape (n_samples, n_features), optional
        Test data
    y_test : array-like of shape (n_samples,), optional
        Test labels/targets
    X_val : array-like of shape (n_samples, n_features), optional
        Validation data
    y_val : array-like of shape (n_samples,), optional
        Validation labels/targets
    wavenumbers : array-like of shape (n_features,), optional
        Feature names (e.g., wavenumbers for spectroscopy)
        If None, uses feature indices

    Attributes
    ----------
    estimator : _BasePCA
        The PCA estimator
    transformer : Pipeline or None
        Preprocessing pipeline before PCA (if model was a Pipeline)
    nr_components : int
        Number of principal components
    nr_features : int
        Number of features in original data
    nr_samples : dict
        Number of samples in each dataset
    wavenumbers : ndarray
        Feature names/indices

    Examples
    --------
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from chemotools.datasets import load_fermentation_train
    >>> from chemotools.inspector import PCAInspector
    >>>
    >>> # Load data
    >>> X, y = load_fermentation_train()
    >>>
    >>> # Create and fit pipeline
    >>> pipeline = make_pipeline(
    ...     StandardScaler(),
    ...     PCA(n_components=5)
    ... )
    >>> pipeline.fit(X)
    >>>
    >>> # Create inspector
    >>> inspector = PCAInspector(pipeline, X, y, wavenumbers=X.columns)
    >>>
    >>> # Print summary table
    >>> inspector.summary()
    >>>
    >>> # Create all diagnostic plots (multiple independent figures)
    >>> inspector.inspect()  # Creates scores, loadings, and variance plots
    >>>
    >>> # Compare preprocessing (creates 2 independent figures)
    >>> inspector.inspect_spectra()
    >>>
    >>> # Access underlying data for custom analysis
    >>> scores = inspector.get_scores('train')
    >>> loadings = inspector.get_loadings([0, 1, 2])
    """

    def __init__(
        self,
        model: Union[_BasePCA, Pipeline],
        X_train: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        wavenumbers: Optional[Sequence] = None,
    ):
        # Validate and extract model
        self._estimator, self._transformer = _validate_and_extract_model(model)

        # Convert to numpy arrays
        self._X_train = np.asarray(X_train)
        self._y_train = np.asarray(y_train) if y_train is not None else None

        self._X_test = np.asarray(X_test) if X_test is not None else None
        self._y_test = np.asarray(y_test) if y_test is not None else None

        self._X_val = np.asarray(X_val) if X_val is not None else None
        self._y_val = np.asarray(y_val) if y_val is not None else None

        # Validate datasets consistency
        _validate_datasets_consistency(
            self._X_train,
            self._y_train,
            self._X_test,
            self._y_test,
            self._X_val,
            self._y_val,
            supervised=False,  # PCA is unsupervised
        )

        # Set wavenumbers
        if wavenumbers is not None:
            self._wavenumbers = np.asarray(wavenumbers)
            if len(self._wavenumbers) != self.nr_features:
                raise ValueError(
                    f"wavenumbers length ({len(self._wavenumbers)}) must match "
                    f"number of features ({self.nr_features})"
                )
        else:
            self._wavenumbers = np.arange(self.nr_features)

        # Cache for computed values
        self._scores_cache: Dict[str, np.ndarray] = {}
        self._preprocessed_cache: Dict[str, np.ndarray] = {}

    @property
    def estimator(self) -> _BasePCA:
        """Return the PCA estimator."""
        return self._estimator

    @property
    def transformer(self) -> Optional[Pipeline]:
        """Return the preprocessing pipeline (if available)."""
        return self._transformer

    @property
    def nr_components(self) -> int:
        """Return the number of principal components."""
        return self.estimator.components_.shape[0]

    @property
    def nr_features(self) -> int:
        """Return the number of features in original data."""
        return self._X_train.shape[1]

    @property
    def nr_samples(self) -> Dict[str, int]:
        """Return the number of samples in each dataset."""
        samples = {"train": self._X_train.shape[0]}
        if self._X_test is not None:
            samples["test"] = self._X_test.shape[0]
        if self._X_val is not None:
            samples["val"] = self._X_val.shape[0]
        return samples

    @property
    def wavenumbers(self) -> np.ndarray:
        """Return the feature names/indices."""
        return self._wavenumbers

    def _get_raw_data(self, dataset: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Get raw X and y data for specified dataset."""
        if dataset == "train":
            return self._X_train, self._y_train
        elif dataset == "test":
            if self._X_test is None:
                raise ValueError("Test data not provided")
            return self._X_test, self._y_test
        elif dataset == "val":
            if self._X_val is None:
                raise ValueError("Validation data not provided")
            return self._X_val, self._y_val
        else:
            raise ValueError(
                f"Invalid dataset: {dataset}. Use 'train', 'test', or 'val'"
            )

    def _get_preprocessed_data(self, dataset: str) -> np.ndarray:
        """Get preprocessed X data for specified dataset."""
        if dataset not in self._preprocessed_cache:
            X, _ = self._get_raw_data(dataset)
            if self.transformer is not None:
                X_preprocessed = self.transformer.transform(X)
            else:
                X_preprocessed = X
            self._preprocessed_cache[dataset] = X_preprocessed
        return self._preprocessed_cache[dataset]

    def get_scores(self, dataset: str = "train") -> np.ndarray:
        """Get PCA scores for specified dataset.

        Parameters
        ----------
        dataset : {'train', 'test', 'val'}, default='train'
            Which dataset to get scores for

        Returns
        -------
        scores : ndarray of shape (n_samples, n_components)
            PCA scores
        """
        if dataset not in self._scores_cache:
            X_preprocessed = self._get_preprocessed_data(dataset)
            scores = self.estimator.transform(X_preprocessed)
            self._scores_cache[dataset] = scores
        return self._scores_cache[dataset]

    def get_loadings(
        self, components: Optional[Union[int, Sequence[int]]] = None
    ) -> np.ndarray:
        """Get PCA loadings.

        Parameters
        ----------
        components : int, list of int, or None, default=None
            Which components to return. If None, returns all components.

        Returns
        -------
        loadings : ndarray of shape (n_features, n_components_selected)
            PCA loadings (components transposed)
        """
        loadings = self.estimator.components_.T

        if components is not None:
            if isinstance(components, int):
                components = [components]
            loadings = loadings[:, components]

        return loadings

    def get_explained_variance_ratio(self) -> np.ndarray:
        """Get explained variance ratio for all components.

        Returns
        -------
        explained_variance_ratio : ndarray of shape (n_components,)
            Explained variance ratio
        """
        return self.estimator.explained_variance_ratio_

    def summary(self) -> None:
        """Display a formatted summary table of the PCA model.

        Prints a formatted summary directly to the console.

        Examples
        --------
        >>> inspector = PCAInspector(pca, X_train, y_train)
        >>> inspector.summary()
        """
        # Calculate cumulative variance
        cumsum = np.cumsum(self.get_explained_variance_ratio())

        # Find components for common variance thresholds
        n_90 = (
            np.argmax(cumsum >= 0.90) + 1
            if np.any(cumsum >= 0.90)
            else self.nr_components
        )
        n_95 = (
            np.argmax(cumsum >= 0.95) + 1
            if np.any(cumsum >= 0.95)
            else self.nr_components
        )
        n_99 = (
            np.argmax(cumsum >= 0.99) + 1
            if np.any(cumsum >= 0.99)
            else self.nr_components
        )

        # Build summary string
        summary_lines = [
            "=" * 70,
            "PCA Model Summary".center(70),
            "=" * 70,
            "",
            "Model Information:",
            "-" * 70,
            f"  Model Type          : {type(self.estimator).__name__}",
            f"  Has Preprocessing   : {'Yes' if self.transformer is not None else 'No'}",
            "",
            "Data Dimensions:",
            "-" * 70,
            f"  Number of Features  : {self.nr_features}",
            f"  Number of Components: {self.nr_components}",
        ]

        # Add sample counts
        for dataset, count in self.nr_samples.items():
            summary_lines.append(f"  Samples ({dataset:5s})    : {count}")

        # Add variance information
        summary_lines.extend(
            [
                "",
                "Explained Variance:",
                "-" * 70,
                f"  PC1 explains        : {self.get_explained_variance_ratio()[0] * 100:6.2f}%",
                f"  PC2 explains        : {self.get_explained_variance_ratio()[1] * 100:6.2f}%"
                if self.nr_components > 1
                else "",
                f"  PC3 explains        : {self.get_explained_variance_ratio()[2] * 100:6.2f}%"
                if self.nr_components > 2
                else "",
                "",
                f"  Total (all {self.nr_components} PCs)   : {cumsum[-1] * 100:6.2f}%",
                "",
                "Components for Variance Thresholds:",
                "-" * 70,
                f"  90% variance        : {n_90} component(s) (actual: {cumsum[n_90 - 1] * 100:.2f}%)",
                f"  95% variance        : {n_95} component(s) (actual: {cumsum[n_95 - 1] * 100:.2f}%)",
                f"  99% variance        : {n_99} component(s) (actual: {cumsum[n_99 - 1] * 100:.2f}%)",
            ]
        )

        # Add preprocessing info if available
        if self.transformer is not None:
            summary_lines.extend(
                [
                    "",
                    "Preprocessing Pipeline:",
                    "-" * 70,
                ]
            )
            for i, (name, transform) in enumerate(self.transformer.steps, 1):
                summary_lines.append(f"  {i}. {name:20s}: {type(transform).__name__}")

        summary_lines.extend(
            [
                "",
                "=" * 70,
            ]
        )

        # Filter out empty strings and print
        print("\n".join([line for line in summary_lines if line is not None]))

    def inspect(
        self,
        dataset: Union[str, Sequence[str]] = "train",
        components_scores: Union[Tuple[int, int], Sequence[Tuple[int, int]]] = (
            (0, 1),
            (1, 2),
        ),
        loadings_components: Union[int, Sequence[int]] = [0, 1, 2],
        variance_threshold: float = 0.95,
        color_by_y: bool = True,
        annotate_by: Optional[Union[str, Dict[str, np.ndarray]]] = None,
        confidence_ellipse: Union[bool, float, Sequence[str]] = True,
        include_spectra: bool = True,
        scores_figsize: Tuple[float, float] = (6, 6),
        loadings_figsize: Tuple[float, float] = (10, 5),
        variance_figsize: Tuple[float, float] = (10, 5),
        spectra_figsize: Tuple[float, float] = (12, 5),
    ) -> Dict[str, matplotlib.figure.Figure]:
        """Create multiple independent PCA diagnostic plots.

        This method creates separate figure windows for:
        - One or more scores plots (default: PC1 vs PC2 and PC2 vs PC3)
        - One loadings plot (overlaid or single component)
        - One explained variance plot
        - Raw and preprocessed spectra plots (if include_spectra=True and preprocessing exists)

        Each plot is a separate figure that can be independently manipulated,
        saved, or displayed.

        Parameters
        ----------
        dataset : Union[str, Sequence[str]], default='train'
            Dataset(s) to inspect. Can be a single dataset name ("train", "test", or "val")
            or a sequence of dataset names (e.g., ["train", "test"]).
        components_scores : int, tuple of two ints, or sequence, default=((0, 1), (1, 2))
            Component(s) for scores plots. Can be:
            - Single int: Creates one 1D plot (e.g., 0 for PC1 vs sample index/y-value)
            - Single tuple (x, y): Creates one 2D scatter plot (e.g., (0, 1) for PC1 vs PC2)
            - Sequence: Creates multiple plots (e.g., ((0, 1), (1, 2), 0) or [0, 1, (0, 1)])
        loadings_components : int or sequence of int, default=[0, 1, 2]
            Which components to show in loadings plot
        variance_threshold : float, default=0.95
            Threshold line for explained variance plot
        color_by_y : bool, default=True
            Whether to color scores by y values (if available)
        annotate_by : str or dict, optional
            Annotations for score plot points. Can be:
            - 'sample_index': Annotate with sample indices (0, 1, 2, ...)
            - 'y': Annotate with y values (only for single dataset)
            - dict: Dictionary mapping dataset names to annotation arrays
              e.g., {'train': ['A', 'B', 'C'], 'test': ['D', 'E']}
            If None (default), no annotations are added.
        confidence_ellipse : bool, float, or sequence of str, default=True
            Controls confidence ellipses on scores plots:
            - True: adds 95% ellipse for 'train' dataset only
            - False or None: no ellipses
            - float (0.90, 0.95, 0.99): confidence level for 'train' only
            - sequence of dataset names: ellipses for those datasets at 95%
              e.g., ['train'], ['train', 'test'], ['train', 'test', 'val']
        include_spectra : bool, default=True
            Whether to include raw and preprocessed spectra plots (only if preprocessing exists)
        scores_figsize : tuple of float, default=(6, 6)
            Figure size for each scores plot (width, height) in inches
        loadings_figsize : tuple of float, default=(10, 5)
            Figure size for loadings plot (width, height) in inches
        variance_figsize : tuple of float, default=(10, 5)
            Figure size for variance plot (width, height) in inches
        spectra_figsize : tuple of float, default=(12, 5)
            Figure size for spectra plots (width, height) in inches

        Returns
        -------
        figures : dict
            Dictionary containing all created figures with keys:
            - Single dataset: 'scores_1', 'scores_2', ..., 'loadings', 'variance', and optionally
              'raw_spectra', 'preprocessed_spectra'
            - Multiple datasets: 'scores_1', 'scores_2', ..., 'loadings', 'variance',
              and optionally 'raw_spectra', 'preprocessed_spectra'. All scores and spectra plots
              show all datasets together on the same figure, colored by dataset.
            Number of 'scores_N' entries depends on components_scores parameter

        Examples
        --------
        >>> inspector = PCAInspector(pca, X_train, y_train)
        >>> # Default: 2 scores plots + loadings + variance + spectra (if preprocessing exists)
        >>> figs = inspector.inspect()
        >>> # Multiple datasets for comparison
        >>> inspector.X_test = X_test
        >>> inspector.y_test = y_test
        >>> figs = inspector.inspect(dataset=["train", "test"])
        >>> # Access individual figures
        >>> figs["scores_1_train"].savefig("scores_1_train.png")
        >>> figs["scores_1_test"].savefig("scores_1_test.png")
        >>> # Without spectra plots
        >>> figs = inspector.inspect(include_spectra=False)
        >>> # Single 2D scores plot (PC1 vs PC2)
        >>> figs = inspector.inspect(components_scores=(0, 1))
        >>> # Single 1D scores plot (PC1 vs sample index or y)
        >>> figs = inspector.inspect(components_scores=0)
        >>> # Three plots: 2D, 2D, and 1D
        >>> figs = inspector.inspect(components_scores=((0, 1), (1, 2), 2))
        >>> # Mix of 1D and 2D plots
        >>> figs = inspector.inspect(components_scores=[0, 1, (0, 1)])
        >>> # Save individual plots
        >>> figs['scores_1'].savefig('scores_pc1_pc2.png')
        >>> figs['loadings'].savefig('loadings.png')
        """
        import matplotlib.pyplot as plt

        figures = {}

        # Normalize dataset to always be a list
        datasets = [dataset] if isinstance(dataset, str) else list(dataset)

        # Determine if we need to add dataset suffixes to figure names
        use_suffix = len(datasets) > 1

        # Variance plot is shared across all datasets (only created once)
        fig_variance, ax_variance = plt.subplots(figsize=variance_figsize)
        variance_plot = ExplainedVariancePlot(
            explained_variance_ratio=self.get_explained_variance_ratio(),
            threshold=variance_threshold,
        )
        variance_plot.render(ax=ax_variance)

        # Apply decorations
        ax_variance.set_title(
            "Explained Variance by Component", fontsize=12, fontweight="bold"
        )
        ax_variance.legend(loc="upper right")
        ax_variance.grid(alpha=0.3)
        plt.tight_layout()
        figures["variance"] = fig_variance

        # Loadings plot is shared across all datasets (only created once)
        fig_loadings, ax_loadings = plt.subplots(figsize=loadings_figsize)

        # Convert to list if it's a sequence for type compatibility
        loadings_comps = (
            loadings_components
            if isinstance(loadings_components, int)
            else list(loadings_components)
        )

        # Determine xlabel based on wavenumbers
        xlabel = (
            "Wavenumber (cm⁻¹)" if self._wavenumbers is not None else "Feature Index"
        )

        # Get preprocessed wavenumbers since loadings are on preprocessed features
        preprocessed_wavenumbers = self._get_preprocessed_wavenumbers()

        loadings = self.get_loadings()
        loadings_plot = LoadingsPlot(
            loadings=loadings,
            feature_names=preprocessed_wavenumbers,
            components=loadings_comps,
        )
        loadings_plot.render(ax=ax_loadings, linewidth=2, alpha=0.7)

        # Apply decorations
        ax_loadings.set_xlabel(xlabel, fontsize=10)
        ax_loadings.set_ylabel("Loading", fontsize=10)

        if isinstance(loadings_components, int):
            title = f"PC{loadings_components + 1} Loadings"
        else:
            comp_str = ", ".join([f"PC{c + 1}" for c in loadings_components])
            title = f"Loadings: {comp_str}"
        ax_loadings.set_title(title, fontsize=12, fontweight="bold")
        ax_loadings.grid(alpha=0.3)
        plt.tight_layout()
        figures["loadings"] = fig_loadings

        # Normalize components_scores to always be a sequence
        if isinstance(components_scores, int):
            # Single component like 0 (for PC1 vs index/y)
            components_list = [components_scores]
        elif (
            isinstance(components_scores, tuple)
            and len(components_scores) == 2
            and isinstance(components_scores[0], int)
        ):
            # Single pair like (0, 1)
            components_list = [components_scores]
        else:
            # Sequence of components/pairs like ((0, 1), (1, 2)) or [0, 1, (0, 1)]
            components_list = list(components_scores)

        # Create scores plots
        # When multiple datasets, combine them on the same plot with different colors
        if use_suffix:
            # Multiple datasets: combine on same plot
            # Use consistent dataset markers across the codebase
            dataset_markers = {
                "train": "o",
                "test": "s",
                "val": "^",
            }

            # Parse confidence_ellipse parameter
            if confidence_ellipse is True:
                ellipse_confidence = 0.95
                ellipse_datasets = ["train"]
            elif confidence_ellipse is False or confidence_ellipse is None:
                ellipse_confidence = None
                ellipse_datasets = []
            elif isinstance(confidence_ellipse, (list, tuple)):
                ellipse_confidence = 0.95
                ellipse_datasets = list(confidence_ellipse)
            elif isinstance(confidence_ellipse, (int, float)):
                # Must be numeric value at this point
                ellipse_confidence = float(confidence_ellipse)
                ellipse_datasets = ["train"]
            else:
                # Shouldn't reach here based on type hints, but handle gracefully
                ellipse_confidence = 0.95
                ellipse_datasets = ["train"]

            for i, component_spec in enumerate(components_list, start=1):
                fig, ax = plt.subplots(figsize=scores_figsize)
                explained_var = self.get_explained_variance_ratio()

                if isinstance(component_spec, int):
                    # 1D plot: Single component vs sample index or y-value
                    var_pct = explained_var[component_spec] * 100

                    for ds in datasets:
                        scores = self.get_scores(ds)
                        _, y = self._get_raw_data(ds)
                        pc_scores = scores[:, component_spec]
                        color = DATASET_COLORS.get(ds, "#7f7f7f")
                        marker = dataset_markers.get(ds, "o")

                        if color_by_y and y is not None:
                            # Plot PC score vs y-value
                            ax.scatter(
                                y,
                                pc_scores,
                                c=color,
                                marker=marker,
                                alpha=0.7,
                                s=50,
                                label=ds.capitalize(),
                            )
                            xlabel_text = "y-value"
                        else:
                            # Plot PC score vs sample index
                            ax.scatter(
                                range(len(pc_scores)),
                                pc_scores,
                                c=color,
                                marker=marker,
                                alpha=0.7,
                                s=50,
                                label=ds.capitalize(),
                            )
                            xlabel_text = "Sample Index"

                    # Apply decorations
                    ax.set_xlabel(xlabel_text, fontsize=10)
                    ax.set_ylabel(
                        f"PC{component_spec + 1} ({var_pct:.1f}%)", fontsize=10
                    )
                    ax.set_title(
                        f"Scores: PC{component_spec + 1}",
                        fontsize=12,
                        fontweight="bold",
                    )
                    ax.grid(alpha=0.3)
                    ax.legend(loc="best")
                else:
                    # 2D plot: Component pair scatter plot using composable ScoresPlot
                    components_pair = component_spec
                    var_x = explained_var[components_pair[0]] * 100
                    var_y = explained_var[components_pair[1]] * 100

                    # Determine which datasets should have ellipses
                    ellipse_datasets_set = (
                        set(ellipse_datasets)
                        if ellipse_confidence is not None
                        else set()
                    )

                    # Compose multiple datasets on same axes
                    for ds in datasets:
                        scores = self.get_scores(ds)
                        _, y = self._get_raw_data(ds)
                        color = DATASET_COLORS.get(ds, "#7f7f7f")

                        # Determine if this dataset should have ellipse
                        should_add_ellipse = ds in ellipse_datasets_set
                        ellipse_param = (
                            ellipse_confidence if should_add_ellipse else None
                        )

                        # Determine color_by parameter
                        color_by = y if (color_by_y and y is not None) else None

                        # Create and render ScoresPlot for this dataset
                        plot = ScoresPlot(
                            scores=scores,
                            components=components_pair,
                            color_by=color_by,
                            label=ds.capitalize(),
                            color=color if color_by is None else None,
                            colormap="viridis" if color_by is not None else None,
                            confidence_ellipse=ellipse_param,
                        )
                        plot.render(ax)

                        # Add annotations if requested
                        if annotate_by is not None:
                            # Get labels for this dataset
                            labels: np.ndarray | None
                            if isinstance(annotate_by, str):
                                if annotate_by == "sample_index":
                                    labels = np.arange(scores.shape[0])
                                elif annotate_by == "y":
                                    labels = y if y is not None else None
                                else:
                                    labels = None
                            elif isinstance(annotate_by, dict) and ds in annotate_by:
                                labels = np.asarray(annotate_by[ds])
                            else:
                                labels = None

                            if labels is not None:
                                annotate_points(
                                    ax,
                                    scores[:, components_pair[0]],
                                    scores[:, components_pair[1]],
                                    labels,
                                    fontsize=8,
                                    alpha=0.7,
                                    xytext=(3, 3),
                                    textcoords="offset points",
                                )

                    # Apply decorations with variance percentages
                    ax.set_xlabel(
                        f"PC{components_pair[0] + 1} ({var_x:.1f}%)", fontsize=10
                    )
                    ax.set_ylabel(
                        f"PC{components_pair[1] + 1} ({var_y:.1f}%)", fontsize=10
                    )
                    ax.set_title(
                        f"Scores: PC{components_pair[0] + 1} vs PC{components_pair[1] + 1}",
                        fontsize=12,
                        fontweight="bold",
                    )
                    ax.grid(alpha=0.3)
                    ax.legend(loc="best")

                plt.tight_layout()
                figures[f"scores_{i}"] = fig
        else:
            # Single dataset: use original logic
            ds = datasets[0]
            scores = self.get_scores(ds)
            _, y = self._get_raw_data(ds)
            explained_var = self.get_explained_variance_ratio()

            # Parse confidence_ellipse parameter for single dataset
            single_ellipse_param: bool | float | None
            if confidence_ellipse is True:
                single_ellipse_param = True  # Use ScoresPlot default (95%)
            elif confidence_ellipse is False or confidence_ellipse is None:
                single_ellipse_param = None
            elif isinstance(confidence_ellipse, (list, tuple)):
                # Check if current dataset should have ellipse
                single_ellipse_param = 0.95 if ds in confidence_ellipse else None
            elif isinstance(confidence_ellipse, (int, float)):
                # Numeric value - use it directly
                single_ellipse_param = float(confidence_ellipse)
            else:
                # Shouldn't reach here based on type hints
                single_ellipse_param = None

            # Prepare color_by parameter
            color_by = y if (color_by_y and y is not None) else None

            # Create scores plots
            for i, component_spec in enumerate(components_list, start=1):
                fig, ax = plt.subplots(figsize=scores_figsize)

                if isinstance(component_spec, int):
                    # 1D plot: Single component vs sample index or y-value
                    pc_scores = scores[:, component_spec]
                    var_pct = explained_var[component_spec] * 100

                    if color_by_y and y is not None:
                        # Plot PC score vs y-value
                        scatter = ax.scatter(
                            y, pc_scores, c=y, cmap="viridis", alpha=0.7, s=50
                        )
                        plt.colorbar(scatter, ax=ax, label="y-value")
                        xlabel_text = "y-value"
                    else:
                        # Plot PC score vs sample index
                        ax.scatter(range(len(pc_scores)), pc_scores, alpha=0.7, s=50)
                        xlabel_text = "Sample Index"

                    # Apply decorations
                    ax.set_xlabel(xlabel_text, fontsize=10)
                    ax.set_ylabel(
                        f"PC{component_spec + 1} ({var_pct:.1f}%)", fontsize=10
                    )
                    ax.set_title(
                        f"Scores: PC{component_spec + 1} ({ds.capitalize()})",
                        fontsize=12,
                        fontweight="bold",
                    )
                    ax.grid(alpha=0.3)
                else:
                    # 2D plot: Component pair scatter plot using composable ScoresPlot
                    components_pair = component_spec
                    var_x = explained_var[components_pair[0]] * 100
                    var_y = explained_var[components_pair[1]] * 100

                    # Create and render ScoresPlot
                    scores_plot = ScoresPlot(
                        scores=scores,
                        components=components_pair,
                        color_by=color_by,
                        label=ds.capitalize(),
                        colormap="viridis" if color_by is not None else None,
                        confidence_ellipse=single_ellipse_param,
                    )
                    scores_plot.render(ax=ax)

                    # Add annotations if requested
                    if annotate_by is not None:
                        # Get labels for this dataset
                        if isinstance(annotate_by, str):
                            if annotate_by == "sample_index":
                                labels = np.arange(scores.shape[0])
                            elif annotate_by == "y":
                                labels = y if y is not None else None
                            else:
                                labels = None
                        elif isinstance(annotate_by, dict) and ds in annotate_by:
                            labels = np.asarray(annotate_by[ds])
                        else:
                            labels = None

                        if labels is not None:
                            annotate_points(
                                ax,
                                scores[:, components_pair[0]],
                                scores[:, components_pair[1]],
                                labels,
                                fontsize=8,
                                alpha=0.7,
                                xytext=(3, 3),
                                textcoords="offset points",
                            )

                    # Apply decorations with variance percentages
                    ax.set_xlabel(
                        f"PC{components_pair[0] + 1} ({var_x:.1f}%)", fontsize=10
                    )
                    ax.set_ylabel(
                        f"PC{components_pair[1] + 1} ({var_y:.1f}%)", fontsize=10
                    )
                    ax.set_title(
                        f"Scores: PC{components_pair[0] + 1} vs PC{components_pair[1] + 1} ({ds.capitalize()})",
                        fontsize=12,
                        fontweight="bold",
                    )
                    ax.grid(alpha=0.3)

                plt.tight_layout()
                figures[f"scores_{i}"] = fig

        # Optionally add spectra plots if preprocessing exists
        # Note: We call inspect_spectra once with all datasets to plot them together
        if include_spectra and self.transformer is not None:
            spectra_figs = self.inspect_spectra(
                dataset=datasets if use_suffix else datasets[0],
                color_by_y=color_by_y,
                figsize=spectra_figsize,
            )
            figures.update(spectra_figs)

        return figures

    def inspect_spectra(
        self,
        dataset: Union[str, Sequence[str]] = "train",
        color_by_y: bool = True,
        xlim: Optional[Tuple[float, float]] = None,
        figsize: Tuple[float, float] = (12, 5),
    ) -> Dict[str, matplotlib.figure.Figure]:
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
        color_by_y : bool, default=True
            Whether to color by y values (if available). Only used for single dataset.
            Ignored when multiple datasets are provided (colors by dataset instead).
        xlim : tuple of float, optional
            X-axis limits for zooming into spectral regions
        figsize : tuple of float, default=(12, 5)
            Figure size for each plot (width, height) in inches

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
        if self.transformer is None:
            raise ValueError(
                "Spectra inspection requires a preprocessing pipeline. "
                "Model must be a Pipeline with preprocessing steps."
            )

        figures = {}

        # Normalize dataset to always be a list
        datasets = [dataset] if isinstance(dataset, str) else list(dataset)

        # Determine if multiple datasets (color by dataset) or single (color by y)
        is_multi_dataset = len(datasets) > 1

        # Determine xlabel based on wavenumbers
        xlabel = (
            "Wavenumber (cm⁻¹)" if self._wavenumbers is not None else "Feature Index"
        )

        if is_multi_dataset:
            # Multiple datasets: plot all on same figure, color by dataset
            # Collect all data
            raw_data = {}
            preprocessed_data = {}
            for ds in datasets:
                X_raw, _ = self._get_raw_data(ds)
                X_preprocessed = self._get_preprocessed_data(ds)
                raw_data[ds] = X_raw
                preprocessed_data[ds] = X_preprocessed

            # Create raw spectra plot with all datasets
            import matplotlib.pyplot as plt

            fig1, ax1 = plt.subplots(figsize=figsize)

            for ds in datasets:
                color = DATASET_COLORS.get(ds, "#7f7f7f")  # default gray if unknown
                for i in range(raw_data[ds].shape[0]):
                    ax1.plot(
                        self.wavenumbers
                        if self.wavenumbers is not None
                        else range(raw_data[ds].shape[1]),
                        raw_data[ds][i, :],
                        color=color,
                        alpha=0.6,
                        linewidth=1,
                        label=ds.capitalize()
                        if i == 0
                        else None,  # Label only first line
                    )

            ax1.set_xlabel(xlabel, fontsize=10)
            ax1.set_ylabel("Intensity", fontsize=10)
            ax1.set_title("Raw Spectra Comparison", fontsize=12, fontweight="bold")
            ax1.grid(alpha=0.3)
            if xlim:
                ax1.set_xlim(xlim)
            ax1.legend(loc="best")
            plt.tight_layout()
            figures["raw_spectra"] = fig1

            # Get wavenumbers for preprocessed data (may be subset if feature selection)
            preprocessed_wavenumbers = self._get_preprocessed_wavenumbers()

            # Create preprocessed spectra plot with all datasets
            fig2, ax2 = plt.subplots(figsize=figsize)

            for ds in datasets:
                color = DATASET_COLORS.get(ds, "#7f7f7f")
                for i in range(preprocessed_data[ds].shape[0]):
                    ax2.plot(
                        preprocessed_wavenumbers,
                        preprocessed_data[ds][i, :],
                        color=color,
                        alpha=0.6,
                        linewidth=1,
                        label=ds.capitalize() if i == 0 else None,
                    )

            ax2.set_xlabel(xlabel, fontsize=10)
            ax2.set_ylabel("Intensity", fontsize=10)
            ax2.set_title(
                "Preprocessed Spectra Comparison", fontsize=12, fontweight="bold"
            )
            ax2.grid(alpha=0.3)
            if xlim:
                ax2.set_xlim(xlim)
            ax2.legend(loc="best")
            plt.tight_layout()
            figures["preprocessed_spectra"] = fig2

        else:
            # Single dataset: use existing logic with color_by_y option
            ds = datasets[0]
            X_raw, y = self._get_raw_data(ds)
            X_preprocessed = self._get_preprocessed_data(ds)

            color_values = None
            if color_by_y and y is not None:
                color_values = y

            # Figure 1: Raw spectra - use .show() for complete figure
            plot_raw = SpectrumPlot(
                x=self.wavenumbers,
                y=X_raw,
                color_by=color_values,
                colormap="viridis",
                xlabel=xlabel,
                ylabel="Intensity",
            )
            fig1 = plot_raw.show(
                figsize=figsize,
                title=f"Raw Spectra ({ds.capitalize()})",
                xlim=xlim,
            )
            figures["raw_spectra"] = fig1

            # Figure 2: Preprocessed spectra - use .show() for complete figure
            # Get wavenumbers for preprocessed data (may be subset if feature selection)
            preprocessed_wavenumbers = self._get_preprocessed_wavenumbers()

            plot_preprocessed = SpectrumPlot(
                x=preprocessed_wavenumbers,
                y=X_preprocessed,
                color_by=color_values,
                colormap="viridis",
                xlabel=xlabel,
                ylabel="Intensity",
            )
            fig2 = plot_preprocessed.show(
                figsize=figsize,
                title=f"Preprocessed Spectra ({ds.capitalize()})",
                xlim=xlim,
            )
            figures["preprocessed_spectra"] = fig2

        return figures

    def _get_feature_mask(self) -> Optional[np.ndarray]:
        """Get the feature selection mask from preprocessing pipeline.

        Detects feature selectors by checking if they are instances of
        sklearn's SelectorMixin, which is the standard way feature selectors
        are identified in scikit-learn.

        Returns
        -------
        mask : np.ndarray or None
            Boolean array indicating which features are selected, or None if no
            feature selector is present in the pipeline.
        """
        from sklearn.feature_selection._base import SelectorMixin

        if self.transformer is None:
            return None

        # Check if transformer is a Pipeline
        if isinstance(self.transformer, Pipeline):
            # Look through pipeline steps for a feature selector (SelectorMixin)
            for _, step in self.transformer.steps:
                if isinstance(step, SelectorMixin):
                    # SelectorMixin provides get_support() method
                    return step.get_support()
        else:
            # Single transformer
            if isinstance(self.transformer, SelectorMixin):
                return self.transformer.get_support()

        return None

    def _get_preprocessed_wavenumbers(self) -> np.ndarray:
        """Get wavenumbers after feature selection.

        Returns
        -------
        wavenumbers : np.ndarray
            Wavenumbers/feature indices after feature selection. If no feature
            selector is present, returns the original wavenumbers.
        """
        feature_mask = self._get_feature_mask()

        if feature_mask is not None and self._wavenumbers is not None:
            # Apply feature mask to wavenumbers
            return self._wavenumbers[feature_mask]
        elif self._wavenumbers is not None:
            return self._wavenumbers
        else:
            # If no wavenumbers provided, use feature indices
            X_preprocessed = self._get_preprocessed_data("train")
            return np.arange(X_preprocessed.shape[1])

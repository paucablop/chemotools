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
        dataset: str = "train",
        components_scores: Union[Tuple[int, int], Sequence[Tuple[int, int]]] = (
            (0, 1),
            (1, 2),
        ),
        loadings_components: Union[int, Sequence[int]] = [0, 1, 2],
        variance_threshold: float = 0.95,
        color_by_y: bool = True,
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
        dataset : str, default='train'
            Which dataset to visualize
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
            'scores_1', 'scores_2', ..., 'loadings', 'variance', and optionally
            'raw_spectra', 'preprocessed_spectra'
            Number of 'scores_N' entries depends on components_scores parameter

        Examples
        --------
        >>> inspector = PCAInspector(pca, X_train, y_train)
        >>> # Default: 2 scores plots + loadings + variance + spectra (if preprocessing exists)
        >>> figs = inspector.inspect()
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

        # Get data
        scores = self.get_scores(dataset)
        loadings = self.get_loadings()
        _, y = self._get_raw_data(dataset)
        explained_var = self.get_explained_variance_ratio()

        # Prepare color_by_dict
        color_by_dict = None
        if color_by_y and y is not None:
            color_by_dict = {dataset: y}

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
                    ax.set_xlabel("y-value", fontsize=10)
                    plt.colorbar(scatter, ax=ax, label="y-value")
                else:
                    # Plot PC score vs sample index
                    ax.scatter(range(len(pc_scores)), pc_scores, alpha=0.7, s=50)
                    ax.set_xlabel("Sample Index", fontsize=10)

                ax.set_ylabel(f"PC{component_spec + 1} ({var_pct:.1f}%)", fontsize=10)
                ax.set_title(
                    f"Scores: PC{component_spec + 1} ({dataset.capitalize()})",
                    fontsize=12,
                    fontweight="bold",
                )
            else:
                # 2D plot: Component pair scatter plot
                components_pair = component_spec
                var_x = explained_var[components_pair[0]] * 100
                var_y = explained_var[components_pair[1]] * 100

                scores_plot = ScoresPlot(
                    scores_dict={dataset: scores},
                    components=components_pair,
                    color_by_dict=color_by_dict,
                    colormap="viridis",
                    confidence_ellipse=True if dataset == "train" else None,
                )
                scores_plot.render(ax=ax)

                # Set axis labels with variance percentages
                ax.set_xlabel(f"PC{components_pair[0] + 1} ({var_x:.1f}%)", fontsize=10)
                ax.set_ylabel(f"PC{components_pair[1] + 1} ({var_y:.1f}%)", fontsize=10)
                ax.set_title(
                    f"Scores: PC{components_pair[0] + 1} vs PC{components_pair[1] + 1} ({dataset.capitalize()})",
                    fontsize=12,
                    fontweight="bold",
                )

            ax.grid(alpha=0.3)
            plt.tight_layout()
            figures[f"scores_{i}"] = fig

        # Figure 3: Loadings plot
        fig3, ax3 = plt.subplots(figsize=loadings_figsize)
        # Convert to list if it's a sequence for type compatibility
        loadings_comps = (
            loadings_components
            if isinstance(loadings_components, int)
            else list(loadings_components)
        )
        loadings_plot = LoadingsPlot(
            loadings=loadings,
            feature_names=self.wavenumbers,
            components=loadings_comps,
        )
        loadings_plot.render(ax=ax3, linewidth=2, alpha=0.7)

        # Set axis labels
        xlabel = (
            "Wavenumber (cm⁻¹)" if self._wavenumbers is not None else "Feature Index"
        )
        ax3.set_xlabel(xlabel, fontsize=10)
        ax3.set_ylabel("Loading", fontsize=10)

        if isinstance(loadings_components, int):
            title = f"PC{loadings_components + 1} Loadings"
        else:
            comp_str = ", ".join([f"PC{c + 1}" for c in loadings_components])
            title = f"Loadings: {comp_str}"
        ax3.set_title(title, fontsize=12, fontweight="bold")
        ax3.grid(alpha=0.3)
        plt.tight_layout()
        figures["loadings"] = fig3

        # Figure 4: Explained variance
        fig4, ax4 = plt.subplots(figsize=variance_figsize)
        variance_plot = ExplainedVariancePlot(
            explained_variance_ratio=self.get_explained_variance_ratio(),
            threshold=variance_threshold,
        )
        variance_plot.render(ax=ax4)
        ax4.set_title("Explained Variance by Component", fontsize=12, fontweight="bold")
        ax4.legend(loc="upper right")
        ax4.grid(alpha=0.3)
        plt.tight_layout()
        figures["variance"] = fig4

        # Optionally add spectra plots if preprocessing exists
        if include_spectra and self.transformer is not None:
            spectra_figs = self.inspect_spectra(
                dataset=dataset,
                color_by_y=color_by_y,
                figsize=spectra_figsize,
            )
            figures.update(spectra_figs)

        return figures

    def inspect_spectra(
        self,
        dataset: str = "train",
        color_by_y: bool = True,
        xlim: Optional[Tuple[float, float]] = None,
        figsize: Tuple[float, float] = (12, 5),
    ) -> Dict[str, matplotlib.figure.Figure]:
        """Create independent plots comparing raw and preprocessed spectra.

        Only available if model is a Pipeline with preprocessing steps.
        Creates two separate figure windows: one for raw spectra and one
        for preprocessed spectra.

        Parameters
        ----------
        dataset : str, default='train'
            Which dataset to visualize
        color_by_y : bool, default=True
            Whether to color by y values (if available)
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
        >>> figs = inspector.inspect_spectra()  # Creates 2 separate plots
        >>> figs = inspector.inspect_spectra(xlim=(1000, 1800))  # Zoom in
        >>> # Save individual plots
        >>> figs['raw_spectra'].savefig('raw_spectra.png')
        >>> figs['preprocessed_spectra'].savefig('preprocessed_spectra.png')
        """
        if self.transformer is None:
            raise ValueError(
                "Spectra inspection requires a preprocessing pipeline. "
                "Model must be a Pipeline with preprocessing steps."
            )

        import matplotlib.pyplot as plt

        figures = {}

        # Get data
        X_raw, y = self._get_raw_data(dataset)
        X_preprocessed = self._get_preprocessed_data(dataset)

        color_values = None
        if color_by_y and y is not None:
            color_values = y

        # Figure 1: Raw spectra
        fig1, ax1 = plt.subplots(figsize=figsize)
        plot_raw = SpectrumPlot(
            x=self.wavenumbers, y=X_raw, color_by=color_values, colormap="viridis"
        )
        plot_raw.render(ax=ax1, xlim=xlim)
        ax1.set_title(
            f"Raw Spectra ({dataset.capitalize()})",
            fontsize=12,
            fontweight="bold",
        )
        ax1.grid(alpha=0.3)
        plt.tight_layout()
        figures["raw_spectra"] = fig1

        # Figure 2: Preprocessed spectra
        fig2, ax2 = plt.subplots(figsize=figsize)
        plot_preprocessed = SpectrumPlot(
            x=self.wavenumbers,
            y=X_preprocessed,
            color_by=color_values,
            colormap="viridis",
        )
        plot_preprocessed.render(ax=ax2, xlim=xlim)
        ax2.set_title(
            f"Preprocessed Spectra ({dataset.capitalize()})",
            fontsize=12,
            fontweight="bold",
        )
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        figures["preprocessed_spectra"] = fig2

        return figures

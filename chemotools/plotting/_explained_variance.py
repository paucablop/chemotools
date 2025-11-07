"""Explained Variance plot for PCA/PLS model diagnostics."""

from typing import Optional, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from chemotools.plotting._utilities import setup_figure


class ExplainedVariancePlot:
    """Visualize explained variance by component with cumulative variance.

    Shows both individual and cumulative explained variance ratios across
    components. Works with any decomposition method (PCA, PLS, ICA, etc.)
    to help determine the optimal number of components.

    **Works with:**
    - PCA: Use `pca.explained_variance_ratio_` directly
    - PLS: Calculate variance explained in X-space or Y-space (see examples)
    - Any method: Just provide an array of variance ratios per component

    Parameters
    ----------
    explained_variance_ratio : np.ndarray
        Array of explained variance ratios for each component.
        Should be 1D array with values between 0 and 1.
        For PCA, use `model.explained_variance_ratio_` directly.
        For PLS, calculate manually for X-space or Y-space.
    xlabel : str, optional
        Label for x-axis. Default is "Component".
    ylabel : str, optional
        Label for y-axis. Default is "Explained Variance Ratio".
    threshold : float or None, optional
        If provided, draws a horizontal dashed line at this variance level.
        Common values are 0.90, 0.95, 0.99. Default is 0.95.

    Attributes
    ----------
    cumulative_variance : np.ndarray
        Cumulative sum of explained variance ratios.

    Examples
    --------
    **Example 1: PCA (simplest case)**

    >>> from sklearn.decomposition import PCA
    >>> pca = PCA(n_components=10)
    >>> pca.fit(X)
    >>> plot = ExplainedVariancePlot(pca.explained_variance_ratio_)
    >>> plot.show(title="PCA Explained Variance")

    **Example 2: PLS - X-space variance**

    >>> from sklearn.cross_decomposition import PLSRegression
    >>> pls = PLSRegression(n_components=5)
    >>> pls.fit(X, y)
    >>>
    >>> # Calculate variance explained in X
    >>> X_centered = X - X.mean(axis=0)
    >>> total_var = np.sum(X_centered**2)
    >>> var_ratios = []
    >>> for i in range(pls.n_components):
    ...     X_recon = pls.x_scores_[:, :i+1] @ pls.x_loadings_[:, :i+1].T
    ...     var_ratios.append(np.sum(X_recon**2) / total_var)
    >>> var_individual = np.diff([0] + var_ratios)
    >>>
    >>> plot = ExplainedVariancePlot(var_individual)
    >>> plot.show(title="PLS Explained Variance in X")

    **Example 3: Custom threshold**

    >>> plot = ExplainedVariancePlot(
    ...     pca.explained_variance_ratio_,
    ...     threshold=0.90
    ... )
    >>> plot.show()

    **Example 4: No threshold line**

    >>> plot = ExplainedVariancePlot(
    ...     pca.explained_variance_ratio_,
    ...     threshold=None
    ... )
    >>> plot.show()
    """

    def __init__(
        self,
        explained_variance_ratio: np.ndarray,
        xlabel: str = "Component",
        ylabel: str = "Explained Variance Ratio",
        threshold: Optional[float] = 0.95,
    ):
        # Validate input
        if not isinstance(explained_variance_ratio, np.ndarray):
            explained_variance_ratio = np.asarray(explained_variance_ratio)

        if explained_variance_ratio.ndim != 1:
            raise ValueError(
                f"explained_variance_ratio must be 1D, got shape {explained_variance_ratio.shape}"
            )

        if len(explained_variance_ratio) == 0:
            raise ValueError("explained_variance_ratio cannot be empty")

        # Validate threshold if provided
        if threshold is not None and not (0 < threshold <= 1):
            raise ValueError(f"threshold must be between 0 and 1, got {threshold}")

        self.explained_variance_ratio = explained_variance_ratio
        self.cumulative_variance = np.cumsum(explained_variance_ratio)
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.threshold = threshold

    def show(
        self,
        figsize: Optional[tuple[float, float]] = None,
        title: Optional[str] = None,
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        **kwargs: Any,
    ) -> Figure:
        """Create and display a complete figure with the variance plot.

        Parameters
        ----------
        figsize : tuple[float, float], optional
            Figure size (width, height) in inches.
        title : str, optional
            Plot title.
        xlim : tuple[float, float], optional
            X-axis limits (min, max).
        ylim : tuple[float, float], optional
            Y-axis limits (min, max).
        **kwargs : Any
            Additional keyword arguments passed to the plotting functions.
            Can include bar plot kwargs (alpha, color, edgecolor) and
            line plot kwargs (linewidth, markersize).

        Returns
        -------
        Figure
            The matplotlib Figure object.
        """
        # Extract figure setup kwargs
        subplot_kw = kwargs.pop("subplot_kw", None)
        gridspec_kw = kwargs.pop("gridspec_kw", None)

        # Use setup_figure utility for consistent styling
        fig, ax = setup_figure(
            figsize=figsize or (10, 6),
            title=title,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            subplot_kw=subplot_kw,
            gridspec_kw=gridspec_kw,
        )

        self.render(ax=ax, xlim=xlim, ylim=ylim, **kwargs)
        ax.legend()
        plt.tight_layout()
        return fig

    def render(
        self,
        ax: Optional[Axes] = None,
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        **kwargs: Any,
    ) -> Axes:
        """Render the plot on the given axes or create new ones.

        Parameters
        ----------
        ax : Axes, optional
            Matplotlib axes to render on. If None, current axes are used.
        xlim : tuple[float, float], optional
            X-axis limits (min, max).
        ylim : tuple[float, float], optional
            Y-axis limits (min, max).
        **kwargs : Any
            Additional keyword arguments for plot customization.

        Returns
        -------
        Axes
            The matplotlib Axes object.
        """
        if ax is None:
            ax = plt.gca()

        self._render_plot(ax, **kwargs)

        # Apply axis limits if provided
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        # Set labels if not already set
        if not ax.get_xlabel():
            ax.set_xlabel(self.xlabel)
        if not ax.get_ylabel():
            ax.set_ylabel(self.ylabel)

        return ax

    def _render_plot(self, ax: Axes, **kwargs: Any) -> None:
        """Internal method to render the variance plot.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes to render on.
        **kwargs : Any
            Additional keyword arguments for customization.
            - bar_kwargs: dict for bar plot styling
            - line_kwargs: dict for line plot styling
            - threshold_kwargs: dict for threshold line styling
        """
        n_components = len(self.explained_variance_ratio)
        components = np.arange(1, n_components + 1)

        # Extract specific kwargs for different plot elements
        bar_kwargs = kwargs.pop("bar_kwargs", {})
        line_kwargs = kwargs.pop("line_kwargs", {})
        threshold_kwargs = kwargs.pop("threshold_kwargs", {})

        # Default bar plot settings
        bar_defaults: dict[str, Any] = {
            "alpha": 0.6,
            "color": "steelblue",
            "edgecolor": "black",
        }
        bar_defaults.update(bar_kwargs)  # type: ignore[arg-type]

        # Bar plot for individual variance
        ax.bar(
            components,
            self.explained_variance_ratio,
            label="Individual",
            **bar_defaults,  # type: ignore[arg-type]
        )

        # Default line plot settings
        line_defaults: dict[str, Any] = {
            "color": "red",
            "marker": "o",
            "linestyle": "-",
            "linewidth": 2,
            "markersize": 6,
        }
        line_defaults.update(line_kwargs)  # type: ignore[arg-type]

        # Line plot for cumulative variance
        ax.plot(
            components,
            self.cumulative_variance,
            label="Cumulative",
            **line_defaults,  # type: ignore[arg-type]
        )

        # Add threshold line if specified
        if self.threshold is not None:
            threshold_defaults: dict[str, Any] = {
                "color": "green",
                "linestyle": "--",
                "alpha": 0.5,
            }
            threshold_defaults.update(threshold_kwargs)  # type: ignore[arg-type]
            ax.axhline(
                y=self.threshold,
                label=f"{self.threshold * 100:.0f}% Threshold",
                **threshold_defaults,  # type: ignore[arg-type]
            )

        # Grid for better readability
        ax.grid(alpha=0.3, axis="y")


def calculate_pls_variance_ratio(
    pls_model, X: np.ndarray, y: Optional[np.ndarray] = None, space: str = "X"
) -> np.ndarray:
    """Calculate explained variance ratio for PLS models.

    Helper function to compute explained variance ratios for PLS regression,
    which can then be used with ExplainedVariancePlot.

    Parameters
    ----------
    pls_model : PLSRegression
        Fitted PLS model from sklearn.
    X : np.ndarray
        Original X data used to fit the model.
    y : np.ndarray, optional
        Original y data used to fit the model. Required if space='Y'.
    space : str, optional
        Which space to calculate variance for: 'X' (predictors) or 'Y' (response).
        Default is 'X'.

    Returns
    -------
    np.ndarray
        Array of explained variance ratios (individual per component).

    Examples
    --------
    >>> from sklearn.cross_decomposition import PLSRegression
    >>> from chemotools.plotting import ExplainedVariancePlot, calculate_pls_variance_ratio
    >>>
    >>> pls = PLSRegression(n_components=5)
    >>> pls.fit(X_train, y_train)
    >>>
    >>> # Get variance ratios for X-space
    >>> var_ratios_x = calculate_pls_variance_ratio(pls, X_train, space='X')
    >>> plot_x = ExplainedVariancePlot(var_ratios_x)
    >>> plot_x.show(title='PLS Explained Variance in X')
    >>>
    >>> # Get variance ratios for Y-space
    >>> var_ratios_y = calculate_pls_variance_ratio(pls, X_train, y_train, space='Y')
    >>> plot_y = ExplainedVariancePlot(var_ratios_y)
    >>> plot_y.show(title='PLS Explained Variance in Y')
    """
    space = space.upper()
    if space not in ("X", "Y"):
        raise ValueError(f"space must be 'X' or 'Y', got '{space}'")

    if space == "Y" and y is None:
        raise ValueError("y data is required when space='Y'")

    # Import PLSRegression for refitting with different n_components (Y-space only)
    from sklearn.cross_decomposition import PLSRegression

    if space == "X":
        # Calculate variance explained in X-space using score variances
        # This measures how much variance the latent variables capture
        X_array = np.asarray(X)  # Convert to numpy array if DataFrame
        score_variances = np.var(pls_model.x_scores_, axis=0)

        # Total variance in X
        total_variance = np.var(X_array, axis=0).sum()

        # Explained variance ratio for each component
        var_individual = score_variances / total_variance

    else:
        # Calculate variance explained in Y-space using R² from predictions
        # For PLS, Y variance is fundamentally about prediction quality
        # We need to refit models with different n_components to get cumulative R²
        y_array = np.asarray(y)  # Convert to numpy array if Series/DataFrame
        y_2d = y_array.reshape(-1, 1) if y_array.ndim == 1 else y_array
        y_centered = y_2d - y_2d.mean(axis=0)
        total_variance = np.sum(y_centered**2)

        # Get model parameters to maintain consistency with original model
        scale = getattr(pls_model, "scale", True)

        # Calculate cumulative R² for each number of components
        var_ratios = []
        for i in range(1, pls_model.n_components + 1):
            # Fit PLS with i components using same parameters as original
            pls_temp = PLSRegression(n_components=i, scale=scale)
            pls_temp.fit(X, y)
            y_pred = pls_temp.predict(X)

            # Calculate R² (cumulative explained variance)
            ss_res = np.sum((y_2d - y_pred) ** 2)
            r2 = 1.0 - (ss_res / total_variance)
            var_ratios.append(float(r2))

        # Convert cumulative to individual variance per component
        var_cumulative = np.array([0.0] + var_ratios)
        var_individual = np.diff(var_cumulative)

    return var_individual

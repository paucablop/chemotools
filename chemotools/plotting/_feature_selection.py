"""
The :mod:`chemotools.plotting._feature_selection` module implements the FeatureSelectionPlot class.
"""

from typing import Any
import numpy as np
from matplotlib.axes import Axes

from chemotools.plotting._spectrum import SpectrumPlot
from chemotools.plotting._utils import validate_data


class FeatureSelectionPlot(SpectrumPlot):
    """Plot class for visualizing feature selection on spectral data.

    This class extends SpectrumPlot to highlight excluded features using
    colored vertical spans.

    Parameters
    ----------
    x : np.ndarray
        X-axis data (e.g., wavelengths, wavenumbers).
    y : np.ndarray
        Y-axis data (e.g., spectra intensities).
    support : np.ndarray
        Boolean mask indicating selected features (True means selected).
        Must have same length as x.
    selection_color : str, optional
        Color to use for highlighting excluded features. Default is "red".
    selection_alpha : float, optional
        Transparency of the selection highlight. Default is 0.2.
    **kwargs : Any
        Additional arguments passed to SpectrumPlot.__init__.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        support: np.ndarray,
        *,
        selection_color: str = "red",
        selection_alpha: float = 0.2,
        **kwargs: Any,
    ):
        super().__init__(x, y, **kwargs)

        self.support = validate_data(
            support, name="support", ensure_2d=False, numeric=False
        ).astype(bool)

        if len(self.support) != len(self.x):
            raise ValueError(
                f"Support mask length ({len(self.support)}) must match x length ({len(self.x)})"
            )

        self.selection_color = selection_color
        self.selection_alpha = selection_alpha

    def _render_plot(self, ax: Axes, **kwargs: Any) -> None:
        """Render the plot with feature selection highlights.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes to plot on.
        **kwargs : Any
            Additional keyword arguments passed to the plot function.
        """
        # 1. Render the standard spectrum plot
        super()._render_plot(ax, **kwargs)

        # 2. Overlay the selection regions (highlight excluded features)
        regions = self._get_continuous_regions(~self.support)

        # Add label only once for the legend
        label_added = False

        for start_idx, end_idx in regions:
            # Get x coordinates for the span
            # We extend slightly to cover the full point width if needed,
            # but for now using exact point coordinates is standard
            x_start = self.x[start_idx]
            x_end = self.x[end_idx]

            # Handle case where start > end (e.g. wavenumbers in descending order)
            if x_start > x_end:
                x_start, x_end = x_end, x_start

            label = "Excluded Features" if not label_added else None

            ax.axvspan(
                x_start,
                x_end,
                color=self.selection_color,
                alpha=self.selection_alpha,
                label=label,
                zorder=-1,  # Put behind spectra
            )
            label_added = True

    def _get_continuous_regions(self, mask: np.ndarray) -> list[tuple[int, int]]:
        """Convert boolean mask to list of (start, end) indices."""
        # Pad with False to detect edges
        padded = np.concatenate(([False], mask, [False]))

        # Find where values change
        # diff = 1 means False -> True (Start)
        # diff = -1 means True -> False (End)
        changes = np.diff(padded.astype(int))

        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0] - 1

        return list(zip(starts, ends))

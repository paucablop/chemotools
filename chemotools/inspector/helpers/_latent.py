"""Thin re-export module — keeps existing ``import _latent`` working.

The actual implementations now live in:
- ``_scores.py``           — scores rendering and ``create_*scores*`` functions
- ``_distances.py``        — distance / residual plots
- ``_loadings_variance.py`` — loadings and explained-variance plots
"""

from chemotools.inspector.helpers._distances import (  # noqa: F401
    create_model_distances_plot,
    create_q_vs_y_residuals_plot,
)
from chemotools.inspector.helpers._loadings_variance import (  # noqa: F401
    create_loadings_plot,
    create_variance_plot,
)
from chemotools.inspector.helpers._scores import (  # noqa: F401
    create_scores_plot_multi_dataset,
    create_scores_plot_single_dataset,
    create_x_vs_y_scores_plots,
)

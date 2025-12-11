"""Shared utilities for inspector modules.

This module provides common helper functions, type aliases, and constants
used across all inspector types (PCA, PLS, ICA, etc.).
"""

from __future__ import annotations
from typing import List, Union, Tuple, Sequence, Optional, Dict
import numpy as np

# ==============================================================================
# Type Aliases for clarity across all inspectors
# ==============================================================================

ComponentSpec = Union[int, Tuple[int, int]]
"""Component specification: either a single int or pair of ints."""

ComponentsInput = Union[ComponentSpec, Sequence[ComponentSpec]]
"""Input for components: single spec or sequence of specs."""

DatasetInput = Union[str, Sequence[str]]
"""Dataset input: single name or sequence of names."""


# ==============================================================================
# Array manipulation utilities
# ==============================================================================


def select_components(
    matrix: np.ndarray, components: Optional[Union[int, Sequence[int]]]
) -> np.ndarray:
    """Select specific components from a matrix.

    This utility function handles the common pattern of extracting specific
    columns (components) from loadings, weights, or rotation matrices.

    Parameters
    ----------
    matrix : np.ndarray
        Matrix of shape (n_features, n_components)
    components : int, sequence of int, or None
        Which components to return. If None, returns all components.
        If int, converts to a single-element list.

    Returns
    -------
    np.ndarray
        Matrix with selected components, shape (n_features, n_selected)

    Raises
    ------
    IndexError
        If any component index is out of bounds.

    Examples
    --------
    >>> matrix = np.array([[1, 2, 3], [4, 5, 6]])
    >>> select_components(matrix, None)  # Returns full matrix
    array([[1, 2, 3],
            [4, 5, 6]])
    >>> select_components(matrix, 0)  # Single component
    array([[1],
            [4]])
    >>> select_components(matrix, [0, 2])  # Multiple components
    array([[1, 3],
            [4, 6]])
    """
    if components is None:
        return matrix
    if isinstance(components, int):
        components = [components]

    # Validate component indices
    n_components = matrix.shape[1]
    for comp in components:
        if comp < 0 or comp >= n_components:
            raise IndexError(
                f"Component index {comp} is out of bounds. "
                f"Valid range: 0 to {n_components - 1} "
                f"(model has {n_components} components)."
            )

    return matrix[:, components]


# ==============================================================================
# Input normalization functions
# ==============================================================================


def normalize_datasets(dataset: DatasetInput) -> List[str]:
    """Normalize dataset input to always be a list.

    Parameters
    ----------
    dataset : DatasetInput
        Single dataset name or sequence of dataset names

    Returns
    -------
    List[str]
        List of dataset names

    Examples
    --------
    >>> normalize_datasets("train")
    ['train']
    >>> normalize_datasets(["train", "test"])
    ['train', 'test']
    """
    return [dataset] if isinstance(dataset, str) else list(dataset)


def normalize_components(components_input: ComponentsInput) -> List[ComponentSpec]:
    """Normalize components input to always be a list.

    Parameters
    ----------
    components_input : ComponentsInput
        Can be:
        - Single int: e.g., 0 (for PC1 vs index/y)
        - Single tuple: e.g., (0, 1) (for PC1 vs PC2)
        - Sequence: e.g., ((0, 1), (1, 2), 0) or [0, 1, (0, 1)]

    Returns
    -------
    List[ComponentSpec]
        List of component specifications

    Examples
    --------
    >>> normalize_components(0)
    [0]
    >>> normalize_components((0, 1))
    [(0, 1)]
    >>> normalize_components(((0, 1), (1, 2)))
    [(0, 1), (1, 2)]
    >>> normalize_components([0, 1, (0, 1)])
    [0, 1, (0, 1)]
    """
    if isinstance(components_input, int):
        # Single component like 0 (for PC1 vs index/y)
        return [components_input]
    elif (
        isinstance(components_input, tuple)
        and len(components_input) == 2
        and isinstance(components_input[0], int)
        and isinstance(components_input[1], int)
    ):
        # Single pair like (0, 1) - cast to ensure type checker understands
        pair: Tuple[int, int] = (components_input[0], components_input[1])
        return [pair]
    else:
        # Sequence of components/pairs like ((0, 1), (1, 2)) or [0, 1, (0, 1)]
        return list(components_input)  # type: ignore[return-value]


def get_default_scores_components(
    n_components: int,
) -> Union[int, Tuple[int, int], Tuple[Tuple[int, int], ...]]:
    """Generate sensible default component pairs for scores plots based on available components.

    Parameters
    ----------
    n_components : int
        Number of components in the model

    Returns
    -------
    Union[int, Tuple[int, int], Tuple[Tuple[int, int], ...]]
        Default component specification:
        - For 1 component: 0 (single 1D plot)
        - For 2 components: (0, 1) (single 2D plot)
        - For 3+ components: ((0, 1), (1, 2)) (two 2D plots)

    Examples
    --------
    >>> get_default_scores_components(1)
    0
    >>> get_default_scores_components(2)
    (0, 1)
    >>> get_default_scores_components(3)
    ((0, 1), (1, 2))
    >>> get_default_scores_components(5)
    ((0, 1), (1, 2))
    """
    if n_components == 1:
        return 0
    elif n_components == 2:
        return (0, 1)
    else:  # 3 or more
        return ((0, 1), (1, 2))


def get_default_loadings_components(n_components: int) -> Union[int, List[int]]:
    """Generate sensible default components for loadings plots based on available components.

    Parameters
    ----------
    n_components : int
        Number of components in the model

    Returns
    -------
    Union[int, List[int]]
        Default component indices:
        - For 1 component: 0
        - For 2+ components: [0, 1, ..., n_components-1] (all components)

    Examples
    --------
    >>> get_default_loadings_components(1)
    0
    >>> get_default_loadings_components(2)
    [0, 1]
    >>> get_default_loadings_components(3)
    [0, 1, 2]
    >>> get_default_loadings_components(5)
    [0, 1, 2, 3, 4]
    """
    if n_components == 1:
        return 0
    else:  # 2 or more
        return list(range(n_components))


# ==============================================================================
# Label and text helpers
# ==============================================================================


def select_primary_target(y: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Return a 1D target vector, falling back to the first column when needed.

    Parameters
    ----------
    y : Optional[np.ndarray]
        Target array that may be 1D or 2D.

    Returns
    -------
    Optional[np.ndarray]
        A flattened 1D view using the first column when ``y`` is 2D, or ``None``
        when no targets are provided.
    """

    if y is None:
        return None

    y_arr = np.asarray(y)

    if y_arr.ndim == 0:
        return y_arr.reshape(1)

    if y_arr.ndim > 1:
        y_arr = y_arr[:, 0]

    return y_arr.ravel()


def get_xlabel_for_features(wavenumbers_provided: bool) -> str:
    """Get appropriate xlabel for feature plots.

    Parameters
    ----------
    wavenumbers_provided : bool
        Whether wavenumbers were provided

    Returns
    -------
    str
        Label for x-axis: either "Wavenumber (cm⁻¹)" or "Feature Index"

    Examples
    --------
    >>> get_xlabel_for_features(True)
    'Wavenumber (cm⁻¹)'
    >>> get_xlabel_for_features(False)
    'Feature Index'
    """
    return "Wavenumber (cm⁻¹)" if wavenumbers_provided else "Feature Index"


def prepare_annotations(
    annotate_by: Optional[Union[str, Dict[str, np.ndarray]]],
    dataset_name: str,
    scores: Optional[np.ndarray],
    y: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Prepare annotation labels for a dataset.

    Parameters
    ----------
    annotate_by : Optional[Union[str, Dict]]
        Annotation specification:
        - 'sample_index': Use sample indices (0, 1, 2, ...)
        - 'y': Use y values
        - dict: Map dataset names to annotation arrays
    dataset_name : str
        Name of current dataset
    scores : Optional[np.ndarray]
        Scores array (used to get sample count). Can be None if not available.
    y : Optional[np.ndarray]
        Target values

    Returns
    -------
    Optional[np.ndarray]
        Labels array or None if no annotations

    Examples
    --------
    >>> scores = np.random.rand(10, 3)
    >>> labels = prepare_annotations('sample_index', 'train', scores, None)
    >>> list(labels)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    if annotate_by is None:
        return None

    if isinstance(annotate_by, str):
        if annotate_by == "sample_index":
            if scores is None:
                return None
            return np.arange(scores.shape[0])
        elif annotate_by == "y":
            return select_primary_target(y)
        else:
            return None
    elif isinstance(annotate_by, dict) and dataset_name in annotate_by:
        return select_primary_target(np.asarray(annotate_by[dataset_name]))
    else:
        return None


def prepare_color_values(
    color_by: Optional[Union[str, Dict[str, np.ndarray]]],
    dataset_name: str,
    y: Optional[np.ndarray],
    n_samples: int,
) -> Optional[np.ndarray]:
    """Prepare color values for a dataset based on color_by specification.

    Parameters
    ----------
    color_by : Optional[Union[str, Dict]]
        Color specification:
        - 'y': Use y values (default)
        - 'sample_index': Use sample indices
        - dict: Map dataset names to color arrays
    dataset_name : str
        Name of current dataset
    y : Optional[np.ndarray]
        Target values
    n_samples : int
        Number of samples in the dataset

    Returns
    -------
    Optional[np.ndarray]
        Color values array or None if no coloring
    """
    if color_by is None:
        return None

    if isinstance(color_by, str):
        if color_by == "y":
            return select_primary_target(y)
        elif color_by == "sample_index":
            return np.arange(n_samples)
        else:
            return None
    elif isinstance(color_by, dict) and dataset_name in color_by:
        return select_primary_target(np.asarray(color_by[dataset_name]))
    else:
        return None

from matplotlib.colors import LinearSegmentedColormap


DATASET_COLORS = {
    "train": "#008BFB",  # Blue
    "test": "#E69F00",  # Orange
    "val": "#FF0051",  # Red
}

# Default marker styles for datasets
DATASET_MARKERS = {
    "train": "o",
    "test": "s",
    "val": "^",
}

# Custom colormaps
CUSTOM_COLORS = ["#008BFB", "#FF0051"]
CUSTOM_CMAP = LinearSegmentedColormap.from_list("shap", CUSTOM_COLORS)

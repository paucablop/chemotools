# Authors: Pau Cabaneros
# License: MIT


class DocLinkMixin:
    """Mixin that adds a documentation link to the sklearn HTML representation.

    When a chemotools estimator is displayed in a Jupyter notebook the sklearn
    HTML widget shows a ``?`` button.  Without this mixin the button is absent
    because sklearn's default implementation only generates a URL for its own
    classes.  This mixin overrides :meth:`_get_doc_link` so that the button
    links to the corresponding page in the chemotools documentation.
    """

    def _get_doc_link(self) -> str:
        """Return the URL of the API documentation page for this estimator."""
        module_name = type(self).__module__
        if not module_name.startswith("chemotools"):
            return ""
        class_name = type(self).__name__
        parts = module_name.split(".")
        # Strip the private submodule leaf (e.g. _linear_correction) so the
        # resulting path matches the public package used by autosummary:
        #   chemotools.baseline._linear_correction  →  chemotools.baseline
        if parts[-1].startswith("_"):
            parts = parts[:-1]
        public_module = ".".join(parts)
        return (
            f"https://paucablop.github.io/chemotools/"
            f"methods/generated/{public_module}.{class_name}.html"
        )

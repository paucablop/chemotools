# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "chemotools"
copyright = "2025, chemotools"
author = "Pau Cabaneros"
# release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_design",
    "nbsphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
]

html_theme = "pydata_sphinx_theme"
html_logo = "_static/images/main/logo-light-3.svg"
html_favicon = "_static/images/main/favicon2.svg"  # or favicon.png if using PNG

# Optional theme customization
html_theme_options = {
    "show_prev_next": True,
    "logo": {
        "image_light": "_static/images/main/logo-light-3.svg",
        "image_dark": "_static/images/main/logo-dark-3.svg",
    },
    "navbar_align": "left",
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links", "theme-switcher", "version-switcher"],
    "show_nav_level": 4,
    "switcher": {
        "json_url": "_static/language-switcher.json",
        "version_match": "en",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/paucablop/chemotools",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },
        {
            "name": "LinkedIn",
            "url": "https://www.linkedin.com/company/chemotools",
            "icon": "fab fa-linkedin",
            "type": "fontawesome",
        },
    ],
    "use_edit_page_button": False,
    "navigation_with_keys": False,
}


# Only include public members unless explicitly told otherwise
autodoc_default_options = {
    "members": True,
    "undoc-members": False,  # don't include things without docstrings
    "private-members": False,  # skip _private methods
    "special-members": False,  # skip __special__ methods
}


# optional but nice:
autodoc_member_order = "bysource"
autoclass_content = "both"
autosummary_generate = True  # generate stub pages automatically
# NOTE: imported members can include objects without a proper __file__ leading to
# None source paths in gettext aggregation (Sphinx relpath NoneType error). Disable for now.
# autosummary_imported_members = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]
html_js_files = [
    "language-switcher.js",
]


# Tell autosummary where to find templates
autosummary_generate = True
templates_path = ["_templates"]
# autosummary_imported_members intentionally disabled above (see note) to avoid
# gettext builder relpath(None) TypeError.

# --- Internationalization (i18n) -----------------------------------------
# Default (source) language
language = "en"
# Where translation catalogs (.po/.mo) will live
locale_dirs = ["../locale"]  # relative to this conf.py directory
gettext_compact = False  # keep one .po per source file
# If a translation is missing, fall back to English text
html_language = "en"
# Optionally, keep untranslated strings marked (set to True only locally)
# gettext_uuid = False


# Napoleon settings (for NumPy-style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True

napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False

napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False

napoleon_use_ivar = True  # <-- KEY: ensures Attributes become proper fields
napoleon_use_param = True
napoleon_use_rtype = True

# Intersphinx to resolve external refs from sklearn docstrings
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# Be tolerant to external labels missing in intersphinx
nitpicky = False

# Exclude autosummary per-method/attribute pages (keep class pages)
exclude_patterns = [
    "_methods/generated/*.*.*.*.rst",
    "api/generated/chemotools.outliers.rst",  # keep excluded due to earlier duplicate method issues
]


# ---------------------------------------------------------------------------
# Workaround: Prevent gettext builder crash when a message location source is None.
# This happens when Sphinx tries to compute a relative path for objects lacking a
# concrete filesystem origin (e.g., dynamically created or C-level objects). We
# monkeypatch GettextRenderer._relpath to tolerate None values.
# Remove this once upstream Sphinx handles None safely.
def _patch_gettext_relpath(app):  # pragma: no cover - build system hook
    try:
        from sphinx.builders.gettext import GettextRenderer, SphinxRenderer
        from sphinx.util.osutil import canon_path, relpath as sphinx_relpath
    except Exception:
        return
    if getattr(GettextRenderer, "_chemotools_safe", False):
        return

    def safe_render(self, filename, context):  # noqa: D401
        # Re-implement original render but guard relpath against None
        def _relpath(s):
            if s is None:
                return "UNKNOWN"
            try:
                return canon_path(sphinx_relpath(s, self.outdir))
            except Exception:
                return "UNKNOWN"

        context["relpath"] = _relpath
        return SphinxRenderer.render(self, filename, context)

    GettextRenderer.render = safe_render  # type: ignore[assignment]
    GettextRenderer._chemotools_safe = True


def setup(app):  # pragma: no cover - Sphinx hook
    app.connect("builder-inited", _patch_gettext_relpath)

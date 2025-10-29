"""
The :mod:`chemotools.utils.discovery` module includes utilities to discover
objects (i.e. estimators, displays, functions) from the `chemotools` package.
"""

# Adapted from scikit-learn

import inspect
import pkgutil
from importlib import import_module
from operator import itemgetter
from pathlib import Path
import sys

from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    ClusterMixin,
    RegressorMixin,
    TransformerMixin,
    OutlierMixin,
)
from sklearn.feature_selection import SelectorMixin
from sklearn.utils._testing import ignore_warnings

_MODULE_TO_IGNORE = {"tests", "utils", "datasets"}


def all_estimators(type_filter=None):
    """Get a list of all estimators from `chemotools`.

    This function crawls the module and gets all classes that inherit
    from `BaseEstimator`. Classes that are defined in test-modules are not
    included.

    Parameters
    ----------
    type_filter : {"classifier", "regressor", "cluster", "transformer", "selector", "outlier"} \
            or list of such str, default=None
        Which kind of estimators should be returned. If None, no filter is
        applied and all estimators are returned.  Possible values are
        'classifier', 'regressor', 'cluster', 'transformer', 'selector', and 'outlier'
        to get estimators only of these specific types, or a list of these to
        get the estimators that fit at least one of the types.

    Returns
    -------
    estimators : list of tuples
        List of (name, class), where ``name`` is the class name as string
        and ``class`` is the actual type of the class.

    Examples
    --------
    >>> from chemotools.utils.discovery import all_estimators
    >>> estimators = all_estimators()
    >>> type(estimators)
    <class 'list'>
    """

    def is_abstract(c):
        return bool(getattr(c, "__abstractmethods__", False))

    all_classes = []
    root = str(Path(__file__).parent.parent)  # chemotools package root
    # Ensure chemotools is importable
    if root not in sys.path:
        sys.path.insert(0, root)

    with ignore_warnings(category=FutureWarning):
        for _, module_name, _ in pkgutil.walk_packages(
            path=[root], prefix="chemotools."
        ):
            module_parts = module_name.split(".")
            # Skip ignored modules and any submodules of datasets

            if any(part in _MODULE_TO_IGNORE for part in module_parts):
                continue

            module = import_module(module_name)
            classes = inspect.getmembers(module, inspect.isclass)
            # Only classes defined in this module
            classes = [
                (name, cls)
                for name, cls in classes
                if not name.startswith("_") and cls.__module__ == module.__name__
            ]
            all_classes.extend(classes)

    all_classes = set(all_classes)

    estimators = [
        (name, cls)
        for name, cls in all_classes
        if issubclass(cls, BaseEstimator)
        and name != "BaseEstimator"
        and not is_abstract(cls)
    ]

    if type_filter is not None:
        if not isinstance(type_filter, list):
            type_filter = [type_filter]
        else:
            type_filter = list(type_filter)  # copy
        filtered_estimators = []
        filters = {
            "classifier": ClassifierMixin,
            "regressor": RegressorMixin,
            "transformer": TransformerMixin,
            "cluster": ClusterMixin,
            "selector": SelectorMixin,
            "outlier": OutlierMixin,
        }
        for name, mixin in filters.items():
            if name in type_filter:
                type_filter.remove(name)
                filtered = [est for est in estimators if issubclass(est[1], mixin)]
                filtered_estimators.extend(filtered)
        estimators = filtered_estimators
        if type_filter:
            raise ValueError(
                "Parameter type_filter must be 'classifier', "
                "'regressor', 'transformer', 'cluster', 'selector', 'outlier' or "
                "None, got"
                f" {repr(type_filter)}."
            )

    return sorted(set(estimators), key=itemgetter(0))


def all_displays():
    """Get a list of all displays from `chemotools`.

    Returns
    -------
    displays : list of tuples
        List of (name, class), where ``name`` is the display class name as
        string and ``class`` is the actual type of the class.

    Examples
    --------
    >>> from chemotools.utils.discovery import all_displays
    >>> displays = all_displays()
    """
    all_classes = []
    root = str(Path(__file__).parent.parent)  # chemotools package
    # Ignore deprecation warnings triggered at import time and from walking
    # packages
    with ignore_warnings(category=FutureWarning):
        for _, module_name, _ in pkgutil.walk_packages(
            path=[root], prefix="chemotools."
        ):
            module_parts = module_name.split(".")
            if any(part in _MODULE_TO_IGNORE for part in module_parts):
                continue
            module = import_module(module_name)
            classes = inspect.getmembers(module, inspect.isclass)
            classes = [
                (name, display_class)
                for name, display_class in classes
                if not name.startswith("_") and name.endswith("Display")
            ]
            all_classes.extend(classes)

    return sorted(set(all_classes), key=itemgetter(0))


def _is_checked_function(item):
    if not inspect.isfunction(item):
        return False

    if item.__name__.startswith("_"):
        return False

    mod = item.__module__
    if not mod.startswith("chemotools.") or mod.endswith("estimator_checks"):
        return False

    return True


def all_functions():
    """Get a list of all functions from `chemotools`.

    Returns
    -------
    functions : list of tuples
        List of (name, function), where ``name`` is the function name as
        string and ``function`` is the actual function.

    Examples
    --------
    >>> from chemotools.utils.discovery import all_functions
    >>> functions = all_functions()
    """
    all_functions = []
    root = str(Path(__file__).parent.parent)  # chemotools package
    # Ignore deprecation warnings triggered at import time and from walking
    # packages
    with ignore_warnings(category=FutureWarning):
        for _, module_name, _ in pkgutil.walk_packages(
            path=[root], prefix="chemotools."
        ):
            module_parts = module_name.split(".")
            if any(part in _MODULE_TO_IGNORE for part in module_parts):
                continue

            module = import_module(module_name)
            functions = inspect.getmembers(module, _is_checked_function)
            functions = [(func.__name__, func) for name, func in functions]
            all_functions.extend(functions)

    # drop duplicates, sort for reproducibility
    # itemgetter is used to ensure the sort does not extend to the 2nd item of
    # the tuple
    return sorted(set(all_functions), key=itemgetter(0))

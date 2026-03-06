"""Helpers for optional third-party dependencies.

Provides a single entry-point — :func:`import_optional_dependency` — for
checking and importing packages that are not listed in the core
``dependencies`` of the project.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


def import_optional_dependency(
    package_name: str,
    *,
    caller_name: str,
    extra_name: str | None = None,
) -> ModuleType:
    """Import an optional dependency, raising a helpful error if missing.

    Parameters
    ----------
    package_name : str
        Top-level package name to import (e.g. ``"matplotlib"``, ``"pandas"``).
    caller_name : str
        Human-readable name of the feature or function that needs the
        package (used in the error message).
    extra_name : str or None, optional
        If the package is provided through a *pip extra*, pass its name
        here (e.g. ``"viz"``).  The error message will then suggest
        ``pip install chemotools[<extra_name>]`` instead of a plain
        ``pip install <package_name>``.

    Returns
    -------
    module : ModuleType
        The imported module.

    Raises
    ------
    ImportError
        When *package_name* cannot be imported.
    """
    try:
        return import_module(package_name)
    except (ModuleNotFoundError, ImportError) as exc:
        # Only intercept errors that originate from *this* package.
        if (
            isinstance(exc, ModuleNotFoundError)
            and exc.name
            not in {
                package_name,
                None,
            }
            and not (exc.name or "").startswith(f"{package_name}.")
        ):
            raise

        if extra_name is not None:
            install_hint = f"pip install chemotools[{extra_name}]"
        else:
            install_hint = f"pip install {package_name}"

        raise ImportError(
            f"'{caller_name}' requires the optional dependency '{package_name}'. "
            f"Install it with: {install_hint}"
        ) from exc

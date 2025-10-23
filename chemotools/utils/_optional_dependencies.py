def check_optional_dependency(package_name: str, caller_name: str) -> object:
    """
    Check if an optional dependency is installed and import it.
    Raise ImportError with detailed error message if a package is not installed.

    Parameters
    ----------
    package_name : str
        The name of the package to check (e.g. "pandas", "polars").
    caller_name : str
        The name of the function or module that requires the package.

    Returns
    -------
    module
        The imported package module.
    """
    try:
        return __import__(package_name)
    except ImportError as e:
        raise ImportError(
            f"The optional dependency '{package_name}' is required for '{caller_name}' "
            f"but is not installed. Install it via: `pip install {package_name}`"
        ) from e

"""Nox sessions for local quality checks and compatibility testing."""

from __future__ import annotations

import nox

SUPPORTED_PYTHONS = ("3.10", "3.11", "3.12", "3.13", "3.14")
MINIMUM_SKLEARN_PYTHONS = ("3.10", "3.11", "3.12")
MINIMUM_SKLEARN = "1.4.*"
LATEST_SKLEARN = "scikit-learn>=1.4.0,<2"

TEST_DEPENDENCIES = ("pytest>=8.3.0", "pytest-cov>=6.3.0")

LINT_DEPENDENCIES = ("ruff>=0.10.0",)

TYPECHECK_DEPENDENCIES = (
    "ty>=0.0.15",
    "pandas-stubs>=2.2.3.241126,<3",
    "scipy-stubs>=1.15.1.0,<2",
    "typing-extensions>=4.12.0",
)

nox.options.sessions = ("lint", "type-check", "tests")
nox.options.reuse_existing_virtualenvs = True


def install_project(session: nox.Session, *extra_dependencies: str) -> None:
    """Install the project in editable mode plus any extra tooling."""
    session.install("-e", ".", *extra_dependencies)


def install_test_dependencies(
    session: nox.Session, *, sklearn_requirement: str | None = None
) -> None:
    """Install the project test dependencies with an optional sklearn override."""
    install_project(session, *TEST_DEPENDENCIES)
    if sklearn_requirement is not None:
        session.install("--upgrade", sklearn_requirement)


@nox.session
def lint(session: nox.Session) -> None:
    """Run formatting and lint checks."""
    install_project(session, *LINT_DEPENDENCIES)
    session.run("ruff", "format", "--check", ".")
    session.run("ruff", "check", ".")


@nox.session(name="type-check")
def type_check(session: nox.Session) -> None:
    """Run static type checks."""
    install_project(session, *TYPECHECK_DEPENDENCIES)
    session.run("ty", "check", "./chemotools")


@nox.session(python=SUPPORTED_PYTHONS)
def tests(session: nox.Session) -> None:
    """Run the test suite with the default dependency set."""
    install_test_dependencies(session, sklearn_requirement=LATEST_SKLEARN)
    session.run("pytest", "-rs", *(session.posargs or ["tests/"]))


@nox.session(name="tests-min-sklearn", python=MINIMUM_SKLEARN_PYTHONS)
def tests_min_sklearn(session: nox.Session) -> None:
    """Run tests against the minimum supported scikit-learn version."""
    install_test_dependencies(
        session, sklearn_requirement=f"scikit-learn=={MINIMUM_SKLEARN}"
    )
    session.run("pytest", "-rs", *(session.posargs or ["tests/"]))

import pytest
from chemotools.utils._linear_algebra import whittaker_solver_dispatch


def test_whittaker_solver_dispatch_invalid_solver():
    # Arrange, Act & Assert
    with pytest.raises(ValueError, match="Unknown solver_type:"):
        whittaker_solver_dispatch("invalid_solver")

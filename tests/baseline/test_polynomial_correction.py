import numpy as np
from sklearn.utils.estimator_checks import check_estimator

from chemotools.baseline import PolynomialCorrection


# Test compliance with scikit-learn
def test_compliance_polynomial_correction():
    # Arrange
    transformer = PolynomialCorrection()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_polynomial_correction_order1_removes_linear_trend():
    # A spectrum that is a pure linear ramp: after order-1 correction the result
    # should be all zeros (the polynomial fits the ramp exactly).
    # Arrange
    spectrum = np.arange(10, dtype=float).reshape(1, -1)
    transformer = PolynomialCorrection(order=1)

    # Act
    result = transformer.fit_transform(spectrum)

    # Assert
    assert np.allclose(result[0], np.zeros(10), atol=1e-8)


def test_polynomial_correction_order2_with_explicit_indices():
    # A flat spectrum (all ones) with a quadratic fit through three explicit
    # anchor points.  The polynomial fitted to three identical y-values is the
    # constant 1, so every point is corrected to zero.
    # Arrange
    spectrum = np.ones((1, 10))
    transformer = PolynomialCorrection(order=2, indices=[0, 4, 9])

    # Act
    result = transformer.fit_transform(spectrum)

    # Assert
    assert np.allclose(result[0], np.zeros(10), atol=1e-8)


def test_polynomial_correction_multi_row_consistent():
    # Two identical spectra must produce identical corrected spectra.
    # Arrange
    row = np.array([1.0, 2.0, 4.0, 3.0, 1.0])
    X = np.vstack([row, row])
    transformer = PolynomialCorrection(order=2, indices=[0, 2, 4])

    # Act
    result = transformer.fit_transform(X)

    # Assert
    assert np.allclose(result[0], result[1], atol=1e-8)


def test_polynomial_correction_snapshot():
    # Snapshot of the exact floating-point output produced by the row-by-row
    # reference implementation.  Any vectorized rewrite must match these values
    # to full double precision — if it doesn't, something changed numerically.
    # Arrange
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 10))
    transformer = PolynomialCorrection(order=2, indices=[1, 4, 7])

    # Act
    result = transformer.fit_transform(X)

    # Assert
    expected = np.array(
        [
            [
                -2.9586749582972027e-01,
                5.5511151231257827e-16,
                1.1166395554867603e00,
                7.1563852548645368e-01,
                -1.2212453270876722e-15,
                6.1260485443604984e-01,
                1.0607597325599178e00,
                -1.7763568394002505e-15,
                -2.5642473879575012e00,
                -4.2489553506800659e00,
            ],
            [
                -1.5863122435256347e00,
                7.9797279894933126e-16,
                -1.6909613653988393e00,
                8.4435672084088553e-01,
                0.0000000000000000e00,
                4.5008974197544538e-01,
                3.2822785019361744e-01,
                1.4432899320127035e-15,
                -7.4572396992285239e-02,
                -4.9250906671311667e-01,
            ],
            [
                -1.3466176144569362e00,
                -8.8817841970012523e-16,
                -2.0286828554050462e00,
                -8.5764701552612033e-01,
                4.4408920985006262e-16,
                -3.5241517225556290e-01,
                -5.8152820006683648e-01,
                1.1102230246251565e-15,
                1.3751096852305684e00,
                3.1154965766754823e00,
            ],
        ]
    )
    assert np.allclose(result, expected, atol=1e-12)

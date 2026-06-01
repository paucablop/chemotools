"""Tests for DirectOrthogonalization."""

import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from chemotools.projection import DirectOrthogonalization


# Test compliance with scikit-learn
def test_compliance_direct_orthogonalization():
    """
    Check sklearn estimator compliance for the DirectOrthogonalization transformer.
    """
    # Arrange
    do = DirectOrthogonalization()

    # Act & Assert
    check_estimator(do)


# Test functionality
def test_direct_orthogonalization_correctness():
    """
    Test the correctness of the DirectOrthogonalization implementation against the
    example provided in the original paper by Trygg and Wold (2002) [1].
    """
    # Arrange
    X = np.array([[-2.18, 1.84, -0.48, 0.83], [-2.18, -0.16, 1.52, 0.83]]).T
    y = np.array([2, 2, 0, -4])

    # Values calculated for numerical stability
    x_weights_orth_ref = np.array([0.85718287, 0.51501217])
    x_loadings_orth_ref = np.array([0.85718287, 0.51501217])
    x_scores_orth_ref = np.array([-2.99481566, 1.49138404, 0.36794023, 1.13549139])
    x_transformed_ref = np.array(
        [
            [0.387105, -0.637633],
            [0.561611, -0.928081],
            [-0.795392, 1.330506],
            [-0.143324, 0.245208],
        ]
    )

    # Act
    do = DirectOrthogonalization(n_components=1).fit(X, y)
    transformed = do.transform(X)

    # Assert
    # Calculated value used to assess numerical stability
    np.testing.assert_allclose(
        do.x_weights_orth_.flatten(), x_weights_orth_ref, atol=1e-8
    )

    np.testing.assert_allclose(
        do.x_loadings_orth_.flatten(), x_loadings_orth_ref, atol=1e-8
    )

    np.testing.assert_allclose(
        do.x_scores_orth_.flatten(), x_scores_orth_ref, atol=1e-8
    )

    np.testing.assert_allclose(
        do.removed_variance_ratio_, 0.7495221388680522, atol=1e-8
    )

    np.testing.assert_allclose(transformed, x_transformed_ref, atol=1e-6)


def test_direct_orthogonalization_raises_error_many_components():
    """
    Test that DirectOrthogonalization raises an error when the number of components
    requested is greater than the number of features.
    """
    # Arrange
    X = np.array([[-2.18, 1.84, -0.48, 0.83], [-2.18, -0.16, 1.52, 0.83]]).T
    y = np.array([2, 2, 0, -4])

    # Act / Assert
    with pytest.raises(
        ValueError,
        match="Number of components must be less than or"
        " equal to the number of features",
    ):
        DirectOrthogonalization(n_components=3).fit(X, y)


def test_fit_rejects_single_sample():
    """Reject datasets with fewer than two samples."""
    # Arrange
    X = np.array([[1.0, 2.0, 3.0]])
    y = np.array([1.0])
    do = DirectOrthogonalization()

    # Act / Assert
    with pytest.raises(ValueError, match="At least 2 samples are required"):
        do.fit(X, y)


def test_fit_rejects_zero_variance_X():
    """Reject X with zero variance after mean-centering (all-constant matrix)."""
    # Arrange
    X = np.ones((3, 2))
    y = np.array([1.0, 2.0, 3.0])
    transformer = DirectOrthogonalization(n_components=1)

    # Act / Assert
    with pytest.raises(ValueError, match="X has zero variance after mean-centering"):
        transformer.fit(X, y)


def test_direct_orthogonalization_snapshot_n_components_1():
    # Snapshot of exact output for n_components=1.
    # Arrange
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 50))
    y = rng.normal(size=3)
    do = DirectOrthogonalization(n_components=1)

    # Act
    result = do.fit_transform(X, y)

    # Assert
    expected = np.array(
        [
            [
                0.31454854103745733,
                -0.26206294996582247,
                0.1784163511680797,
                0.01432773550374984,
                -0.46423125426205414,
                -0.05233398092510261,
                0.4368323468462908,
                0.7639010209190837,
                -1.2941884887520376,
                -0.5992270126804423,
                -0.5392296904678169,
                -0.41560155787042946,
                -0.14729370457869062,
                -0.21791159500333712,
                -0.1241675297373771,
                -0.5534307403013454,
                0.8006723215684187,
                0.5961719074209311,
                0.6109022636052568,
                -0.19767545766591335,
                0.18049802975540027,
                0.9630374335243147,
                0.18156613578808403,
                -0.4908893510556539,
                0.9933077902904092,
                -0.6848580259632089,
                -0.8880425081704829,
                0.11283879528236047,
                -0.2879592885439221,
                0.5934663611163016,
                -0.5487896334003602,
                -0.25111693853604466,
                -0.11008424672963396,
                0.10461151275150599,
                -0.4295902851089998,
                0.6416956424186953,
                0.36489779892110846,
                0.7613328845132249,
                -0.6380450054450978,
                1.5012518168196587,
                -0.4624922025544967,
                1.2540864436111197,
                0.3953782531286795,
                0.08326826459601819,
                0.27300108739224355,
                0.20158469845269023,
                0.07594657054940061,
                0.4244728883936997,
                -0.10758126360936908,
                0.18944103170686438,
            ],
            [
                0.23135979220914377,
                -1.1215823534077007,
                0.303896876356211,
                0.716924511967483,
                -1.336040502135328,
                0.6713854709869674,
                1.0086264423660285,
                0.8183001914149866,
                -0.7900392347576609,
                -1.106332296929833,
                -0.4925281842947326,
                -0.8648405520951996,
                0.2859084221823194,
                -0.4964981016947589,
                -0.419701364218857,
                -0.37793120068863234,
                0.685842422147001,
                0.7113613209950522,
                0.5003552192219817,
                -1.3757863682614575,
                -0.15422479956211607,
                0.9529397181871961,
                0.43881873962017226,
                -0.0556750744921648,
                1.7620521866761716,
                -0.8005994202092716,
                -0.5650573463308446,
                0.24456393378301372,
                -0.06425051465329068,
                1.7532648918426657,
                -0.11904576678265133,
                -0.6052016973504022,
                -0.4103609020610627,
                -0.7999959392343525,
                -0.8476969486158856,
                0.4393146038165491,
                -0.09874984433133277,
                0.6999256754575901,
                0.19447531392469186,
                1.6838878007071363,
                -0.8190345504466466,
                1.7478282159290415,
                0.20159243409885458,
                -0.26959720807574894,
                0.24408198544482448,
                0.6873944383268142,
                1.083429787390687,
                0.4394809816370102,
                -0.06697578495889538,
                -0.6502334021544711,
            ],
            [
                0.4398851483806139,
                1.0329348410858548,
                -0.01063930282638254,
                -1.0442420534702919,
                0.8492830721084236,
                -1.1427283196849451,
                -0.42466314431922436,
                0.6819403282981389,
                -2.053766652629075,
                0.16480486409149386,
                -0.6095926713275271,
                0.2612458790178225,
                -0.7999791413331521,
                0.20182169936735828,
                0.32109951287037947,
                -0.817847710085807,
                0.9736811745413092,
                0.42262139124446646,
                0.7774583386750777,
                1.5773292958344256,
                0.6848093028284375,
                0.9782511901933015,
                -0.206024362309815,
                -1.1466064000564948,
                -0.16492353202346277,
                -0.5104758662566442,
                -1.3746691851831137,
                -0.08562532629678386,
                -0.6250108627924573,
                -1.1539479838551263,
                -1.1962646709148244,
                0.2823660505314217,
                0.3423285828137551,
                1.467541363316292,
                0.20035152132052453,
                0.946613712785731,
                1.0634540751772132,
                0.8538522619272249,
                -1.8923645359551702,
                1.2260826993410598,
                0.07469352133699958,
                0.5101887468968356,
                0.6873463071084323,
                0.6149142121086439,
                0.31657214930328326,
                -0.5303621749850802,
                -1.4419813960637788,
                0.4018608943884947,
                -0.16875964416295625,
                1.4545393328024336,
            ],
        ]
    )
    assert np.allclose(result, expected, atol=1e-12)


def test_direct_orthogonalization_snapshot_n_components_2():
    # With n_samples=3, column-wise mean-centering reduces rank(X_centered) to ≤ 2,
    # and y-deflation reduces it further to rank ≤ 1.  The second orthogonal
    # component therefore corresponds to a near-zero singular value (pure numerical
    # noise), so the second right singular vector is arbitrary and differs between
    # LAPACK backends (OpenBLAS vs. Accelerate etc.).  An exact snapshot is not
    # reproducible across platforms in this case, so we verify platform-invariant
    # algebraic properties instead.
    # Arrange
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 50))
    y = rng.normal(size=10)
    do = DirectOrthogonalization(n_components=2)

    # Act
    result = do.fit_transform(X, y)

    # Assert — platform-invariant algebraic properties
    X_centered = X - do.mean_X_
    result_centered = result - do.mean_X_

    # 1. Corrected data is orthogonal to the orthogonal loadings.
    assert np.allclose(result_centered @ do.x_loadings_orth_, 0, atol=1e-10)

    # 2. Corrected + removed variation reconstructs the original (identity check).
    removed = X_centered @ do.x_loadings_orth_ @ do.x_loadings_orth_.T
    assert np.allclose(result_centered + removed, X_centered, atol=1e-12)

    # 3. Variance partition sums to 1.
    assert 0.0 <= do.retained_variance_ratio_ <= 1.0
    assert np.isclose(do.retained_variance_ratio_ + do.removed_variance_ratio_, 1.0)

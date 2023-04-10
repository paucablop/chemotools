import numpy as np

from chemotools.baseline import AirPls
from chemotools.smoothing import WhittakerSmooth
from tests.fixtures import spectrum, reference_airpls, reference_whitakker


def test_air_pls(spectrum, reference_airpls):
    from chemotools.baseline import AirPls

    air_pls = AirPls()
    spectrum_corrected = air_pls.fit_transform(spectrum)

    assert np.allclose(spectrum_corrected[0], reference_airpls[0], atol=1e-8)


def test_max_norm(spectrum):
    from chemotools.normalize import MaxNorm

    max_norm = MaxNorm()
    spectrum_corrected = max_norm.fit_transform(spectrum)

    assert np.allclose(spectrum_corrected[0], spectrum[0] / np.max(spectrum[0]), atol=1e-8)


def test_whitakker_smooth(spectrum, reference_whitakker):
    from chemotools.baseline import AirPls

    whitakker_smooth = WhittakerSmooth()
    spectrum_corrected = whitakker_smooth.fit_transform(spectrum)

    assert np.allclose(spectrum_corrected[0], reference_whitakker[0], atol=1e-2)

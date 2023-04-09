import numpy as np

from chemotools.baseline import AirPls
from tests.fixtures import spectrum, reference_airpls


def test_air_pls(spectrum, reference_airpls):
    from chemotools.baseline import AirPls

    air_pls = AirPls()
    spectrum_corrected = air_pls.fit_transform(spectrum)

    assert np.allclose(spectrum_corrected[0], reference_airpls[0], atol=1e-8)

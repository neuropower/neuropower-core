import numpy as np

from neuropower import peakdistribution


def test_peakdistribution():
    x = peakdistribution.peakdens3D(2, 1)
    assert np.around(x, decimals=2) == 0.48

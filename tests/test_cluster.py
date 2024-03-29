import numpy as np
import pytest

from neuropower import cluster


@pytest.mark.xfail(reason="AttributeError: 'DataFrame' object has no attribute 'append'")
def test_cluster_output():
    np.random.seed(seed=100)
    testSPM = np.random.rand(4, 4, 4)
    mask = np.zeros((4, 4, 4)) + 1
    tab = cluster.PeakTable(testSPM, 0.5, mask)
    assert np.around(tab.peak[1], decimals=2) == 0.98
    assert len(tab) == 6

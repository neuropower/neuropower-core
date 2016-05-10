from unittest import TestCase
from neuropower import peakdistribution
import numpy as np

class TestPeakDistribution(TestCase):
    def test_peakdistribution(self):
        x = peakdistribution.peakdens3D(2,1)
        self.assertEqual(np.around(x,decimals=2),0.48)

from unittest import TestCase
from neuropower import cluster
import numpy as np

class TestCluster(TestCase):
    def test_cluster_output(self):
        np.random.seed(seed=100)
        testSPM = np.random.rand(4,4,4)
        tab = cluster.cluster(testSPM,0.5)
        self.assertEqual(np.around(tab.peak[1],decimals=2),0.98)
        self.assertEqual(len(tab),6)

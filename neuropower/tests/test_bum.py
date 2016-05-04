from unittest import TestCase
from neuropower import BUM
import numpy as np

class TestBUM(TestCase):
    def test_fpLL(self):
        np.random.seed(seed=100)
        testpeaks = np.vstack((np.random.uniform(0,1,10),np.random.uniform(0,0.2,10))).flatten()
        x = np.sum(BUM.fpLL([0.5,0.5],testpeaks))
        self.assertEqual(np.around(x,decimals=2),9.57)

    def test_fbumnLL(self):
        np.random.seed(seed=100)
        testpeaks = np.vstack((np.random.uniform(0,1,10),np.random.uniform(0,0.2,10))).flatten()
        x = BUM.fbumnLL([0.5,0.5],testpeaks)
        self.assertEqual(np.around(x,decimals=2)[0],-4.42)

    def test_fbumnLL(self):
        np.random.seed(seed=100)
        testpeaks = np.vstack((np.random.uniform(0,1,10),np.random.uniform(0,0.2,10))).flatten()
        x = BUM.bumOptim(testpeaks,starts=1,seed=100)
        self.assertEqual(np.around(x['pi1'],decimals=2),0.29)

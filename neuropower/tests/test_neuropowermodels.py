from unittest import TestCase
from neuropower import neuropowermodels
import numpy as np

class TestAltPDF(TestCase):
    def test_AltPDF_RFT(self):
        x = neuropowermodels.altPDF(peaks=[4.],mu=3.,sigma=1.,exc=2.,method="RFT")
        self.assertEqual(np.around(x,decimals=2)[0],0.29)

    def test_AltPDF_CS(self):
        x = neuropowermodels.altPDF(peaks=[4.],mu=3.,method="CS")
        self.assertEqual(np.around(x,decimals=2)[0],0.31)

class TestNulPDF(TestCase):
    def test_NulPDF_RFT(self):
        x = neuropowermodels.nulPDF(peaks=[4.],exc=2.,method="RFT")
        self.assertEqual(np.around(x,decimals=2)[0],0.04)

    def test_NulPDF_CS(self):
        x = neuropowermodels.nulPDF(peaks=[4.],method="CS")
        self.assertEqual(np.around(x,decimals=2)[0],0.01)

class TestAltCDF(TestCase):
    def test_AltCDF_RFT(self):
        x = neuropowermodels.altCDF(peaks=[4.],mu=3.,sigma=1.,exc=2.,method="RFT")
        self.assertEqual(np.around(x,decimals=2)[0],0.81)

    def test_AltCDF_CS(self):
        x = neuropowermodels.altCDF(peaks=[4.],mu=3.,method="CS")
        self.assertEqual(np.around(x,decimals=2)[0],0.16)

class TestNulCDF(TestCase):
    def test_NulCDF_RFT(self):
        x = neuropowermodels.nulCDF(peaks=[3.],exc=2.,method="RFT")
        self.assertEqual(np.around(x,decimals=2)[0],0.86)

    def test_NulCDF_CS(self):
        x = neuropowermodels.nulCDF(peaks=[3.],method="CS")
        self.assertEqual(np.around(x,decimals=2)[0],0.93)

class TestTruncTau(TestCase):
    def test_trunctau(self):
        x = neuropowermodels.TruncTau(4,1,3)
        self.assertEqual(np.around(x,decimals=2),0.21)

class TestMixPDF(TestCase):
    def test_MixPDF_RFT(self):
        x = neuropowermodels.mixPDF(peaks=[4.],pi1=0.5,mu=3.,sigma=1.,exc=2.,method="RFT")
        self.assertEqual(np.around(x,decimals=2),0.16)

    def test_MixPDF_CS(self):
        x = neuropowermodels.mixPDF(peaks=[4.],pi1=0.5,mu=3.,method="CS")
        self.assertEqual(np.around(x,decimals=2),0.16)

class TestMixPDF_SLL(TestCase):
    def test_MixPDF_SLL_RFT(self):
        np.random.seed(seed=100)
        testpeaks = np.random.uniform(2,10,30)
        x = neuropowermodels.mixPDF_SLL(pars=[4,1],peaks=testpeaks,pi1=0.5,exc=2.,method="RFT")
        self.assertEqual(np.around(x,decimals=2),156.03)

    def test_MixPDF_SLL_CS(self):
        np.random.seed(seed=100)
        testpeaks = np.random.uniform(-5,5,30)
        x = neuropowermodels.mixPDF_SLL(pars=[3],peaks=testpeaks,pi1=0.5,method="CS")
        self.assertEqual(np.around(x,decimals=2),451.62)

class TestModelfit(TestCase):
    def test_Modelfit_RFT(self):
        np.random.seed(seed=100)
        testpeaks = np.random.uniform(2,10,20)
        x = neuropowermodels.modelfit(peaks=testpeaks,pi1=0.5,exc=2.,starts=1,seed=20,method="RFT")
        self.assertEqual(np.around(x['mu'],decimals=2),6.10)

    def test_MixPDF_SLL_CS(self):
        np.random.seed(seed=100)
        testpeaks = np.random.uniform(-5,5,30)
        x = neuropowermodels.modelfit(peaks=testpeaks,pi1=0.5,starts=1,seed=20,method="CS")
        self.assertEqual(np.around(x['maxloglikelihood'],decimals=2),448.15)

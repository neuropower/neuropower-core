import numpy as np

from neuropower import neuropowermodels


def test_AltPDF_RFT():
    x = neuropowermodels.altPDF(peaks=[4.0], mu=3.0, sigma=1.0, exc=2.0, method="RFT")
    assert np.around(x, decimals=2)[0] == 0.29


def test_AltPDF_CS():
    x = neuropowermodels.altPDF(peaks=[4.0], mu=3.0, method="CS")
    assert np.around(x, decimals=2)[0] == 0.31


def test_NulPDF_RFT():
    x = neuropowermodels.nulPDF(peaks=[4.0], exc=2.0, method="RFT")
    assert np.around(x, decimals=2)[0] == 0.04


def test_NulPDF_CS():
    x = neuropowermodels.nulPDF(peaks=[4.0], method="CS")
    assert np.around(x, decimals=2)[0] == 0.01


def test_AltCDF_RFT():
    x = neuropowermodels.altCDF(peaks=[4.0], mu=3.0, sigma=1.0, exc=2.0, method="RFT")
    assert np.around(x, decimals=2)[0] == 0.81


def test_AltCDF_CS():
    x = neuropowermodels.altCDF(peaks=[4.0], mu=3.0, method="CS")
    assert np.around(x, decimals=2)[0] == 0.16


def test_NulCDF_RFT():
    x = neuropowermodels._nulCDF(peaks=[3.0], exc=2.0, method="RFT")
    assert np.around(x, decimals=2)[0] == 0.86


def test_NulCDF_CS():
    x = neuropowermodels._nulCDF(peaks=[3.0], method="CS")
    assert np.around(x, decimals=2)[0] == 0.93


def test_trunctau():
    x = neuropowermodels.TruncTau(4, 1, 3)
    assert np.around(x, decimals=2) == 0.21


def test_MixPDF_RFT():
    x = neuropowermodels.mixPDF(
        peaks=[4.0], pi1=0.5, mu=3.0, sigma=1.0, exc=2.0, method="RFT"
    )
    assert np.around(x, decimals=2) == 0.16


def test_MixPDF_CS():
    x = neuropowermodels.mixPDF(peaks=[4.0], pi1=0.5, mu=3.0, method="CS")
    assert np.around(x, decimals=2) == 0.16


def test_MixPDF_SLL_RFT():
    np.random.seed(seed=100)
    testpeaks = np.random.uniform(2, 10, 30)
    x = neuropowermodels._mixPDF_SLL(
        pars=[4, 1], peaks=testpeaks, pi1=0.5, exc=2.0, method="RFT"
    )
    assert np.around(x, decimals=2) == 156.03


def test_Modelfit_RFT():
    np.random.seed(seed=100)
    testpeaks = np.random.uniform(2, 10, 20)
    x = neuropowermodels.modelfit(
        peaks=testpeaks, pi1=0.5, exc=2.0, seed=20, method="RFT"
    )
    assert np.around(x["mu"], decimals=2) == 6.10


def test_MixPDF_SLL_CS():
    np.random.seed(seed=100)
    testpeaks = np.random.uniform(-5, 5, 30)
    x = neuropowermodels.modelfit(peaks=testpeaks, pi1=0.5, seed=20, method="CS")
    assert np.around(x["maxloglikelihood"], decimals=2) == 448.15

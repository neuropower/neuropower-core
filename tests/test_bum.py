import numpy as np
import pytest

from neuropower import BUM


def test_fpLL():
    np.random.seed(seed=100)
    testpeaks = np.vstack(
        (np.random.uniform(0, 1, 10), np.random.uniform(0, 0.2, 10))
    ).flatten()
    x = np.sum(BUM._fpLL([0.5, 0.5], testpeaks))
    assert np.around(x, decimals=2) == 9.57


@pytest.mark.xfail(reason="IndexError: invalid index to scalar variable.")
def test_fbumnLL():
    np.random.seed(seed=100)
    testpeaks = np.vstack(
        (np.random.uniform(0, 1, 10), np.random.uniform(0, 0.2, 10))
    ).flatten()
    x = BUM._fbumnLL([0.5, 0.5], testpeaks)
    assert np.around(x, decimals=2)[0] == -4.42


@pytest.mark.xfail(reason="module 'neuropower.BUM' has no attribute 'bumOptim'")
def test_bumOptim():
    np.random.seed(seed=100)
    testpeaks = np.vstack(
        (np.random.uniform(0, 1, 10), np.random.uniform(0, 0.2, 10))
    ).flatten()
    x = BUM.bumOptim(testpeaks, starts=1, seed=100)
    assert np.around(x["pi1"], decimals=2) == 0.29

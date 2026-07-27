import inspect

import numpy as np
import scipy.stats as stats

# SciPy >= 1.16 evaluates multivariate_normal.cdf with an unseeded randomized
# quasi-Monte-Carlo integration, so repeated calls with identical inputs return
# slightly different values. That noise is larger than the step size used by
# scipy.optimize.minimize's finite-difference gradient in neuropowermodels.modelfit,
# which makes the CS-method fit non-reproducible and prone to converging to a bad
# optimum. Pinning a fresh, fixed-seed generator on each call restores determinism.
_MVN_CDF_SUPPORTS_RNG = (
    "rng" in inspect.signature(stats.multivariate_normal.cdf).parameters
)


def _mvn_cdf(x, mean, cov, lower_limit):
    kwargs = {"rng": np.random.default_rng(0)} if _MVN_CDF_SUPPORTS_RNG else {}
    return stats.multivariate_normal.cdf(
        x, mean=mean, cov=cov, lower_limit=lower_limit, **kwargs
    )


def peakdens3D(x, k):
    """Return the PDF of a peak.

    Parameters
    ----------
    x
    k

    Returns
    -------
    out
    """
    fd1 = 144 * stats.norm.pdf(x) / (29 * 6 ** (0.5) - 36)
    fd211 = (
        k**2.0
        * (
            (1.0 - k**2.0) ** 3.0
            + 6.0 * (1.0 - k**2.0) ** 2.0
            + 12.0 * (1.0 - k**2.0)
            + 24.0
        )
        * x**2.0
        / (4.0 * (3.0 - k**2.0) ** 2.0)
    )
    fd212 = (
        2.0 * (1.0 - k**2.0) ** 3.0 + 3.0 * (1.0 - k**2.0) ** 2.0 + 6.0 * (1.0 - k**2.0)
    ) / (4.0 * (3.0 - k**2.0))
    fd213 = 3.0 / 2.0
    fd21 = fd211 + fd212 + fd213
    fd22 = np.exp(-(k**2.0) * x**2.0 / (2.0 * (3.0 - k**2.0))) / (
        2.0 * (3.0 - k**2.0)
    ) ** (0.5)
    fd23 = stats.norm.cdf(2.0 * k * x / ((3.0 - k**2.0) * (5.0 - 3.0 * k**2.0)) ** (0.5))
    fd2 = fd21 * fd22 * fd23
    fd31 = (k**2.0 * (2.0 - k**2.0)) / 4.0 * x**2.0 - k**2.0 * (1.0 - k**2.0) / 2.0 - 1.0
    fd32 = np.exp(-(k**2.0) * x**2.0 / (2.0 * (2.0 - k**2.0))) / (
        2.0 * (2.0 - k**2.0)
    ) ** (0.5)
    fd33 = stats.norm.cdf(k * x / ((2.0 - k**2.0) * (5.0 - 3.0 * k**2.0)) ** (0.5))
    fd3 = fd31 * fd32 * fd33
    fd41 = (7.0 - k**2.0) + (1 - k**2) * (
        3.0 * (1.0 - k**2.0) ** 2.0 + 12.0 * (1.0 - k**2.0) + 28.0
    ) / (2.0 * (3.0 - k**2.0))
    fd42 = k * x / (4.0 * np.pi ** (0.5) * (3.0 - k**2.0) * (5.0 - 3.0 * k**2) ** 0.5)
    fd43 = np.exp(-3.0 * k**2.0 * x**2 / (2.0 * (5 - 3.0 * k**2.0)))
    fd4 = fd41 * fd42 * fd43
    fd51 = np.pi**0.5 * k**3.0 / 4.0 * x * (x**2.0 - 3.0)
    f521low = np.array([-10.0, -10.0])
    f521up = np.array([0.0, k * x / 2.0 ** (0.5)])
    f521mu = np.array([0.0, 0.0])
    f521sigma = np.array([[3.0 / 2.0, -1.0], [-1.0, (3.0 - k**2.0) / 2.0]])
    fd521 = _mvn_cdf(f521up, mean=f521mu, cov=f521sigma, lower_limit=f521low)
    f522low = np.array([-10.0, -10.0])
    f522up = np.array([0.0, k * x / 2.0 ** (0.5)])
    f522mu = np.array([0.0, 0.0])
    f522sigma = np.array([[3.0 / 2.0, -1.0 / 2.0], [-1.0 / 2.0, (2.0 - k**2.0) / 2.0]])
    fd522 = _mvn_cdf(f522up, mean=f522mu, cov=f522sigma, lower_limit=f522low)
    fd5 = fd51 * (fd521 + fd522)
    out = fd1 * (fd2 + fd3 + fd4 + fd5)
    return out

#!/usr/bin/env python

"""
Fit a beta-uniform mixture model to a list of p-values.
The BUM model is introduced in Pounds & Morris, 2003.
"""

import numpy as np
from scipy.optimize import minimize


def _fpLL(pars, p_values):
    """
    Returns the gradient function of the BUM model.

    Parameters
    ----------
    pars : :obj:`list`

    p_values : :obj:`float`
    """
    a, l = pars
    dl = -sum((1 - a * p_values ** (a - 1)) / (a * (1 - l) * p_values ** (a - 1) + l))
    da = -sum((a * (1 - l) * p_values ** (a - 1) * np.log(p_values) + (1 - l) * p_values ** (a - 1)) \
              / (a * (1 - l) * p_values ** (a - 1) + l))
    return np.asarray([dl, da])


def _fbumnLL(pars, p_values):
    """
    Returns the negative sum of the loglikelihood

    Parameters
    ----------
    pars : :obj:`list`
    p_values :
    """
    a = pars[0]
    l = pars[1]
    L = l + (1 - l) * a * p_values ** (a - 1)
    negsumlog = -sum(np.log(L))
    return negsumlog


def EstimatePi1(p_values, n_iters=10, seed=None):
    """
    Returns the MLE estimator for pi1, with the shaping parameters and the value
    of the negative sum of the loglikelihood. Searches the maximum likelihood
    estimator for the shape parameters of the BUM-model given a list of p-values.

    Parameters
    ----------
    p_values : :obj:`numpy.ndarray`
        Array of p-values associated with local maxima in input image.
    n_iters : :obj:`int`, optional
        Number of iterations to run.
    seed : :obj:`int`, optional
        Random seed.

    Returns
    -------
    out : :obj:`dict`
        Dictionary of parameters from fitted beta-uniform mixture model,
        including ``maxloglikelihood``, ``pi1``, ``a``, and ``lambda``.
    """
    if seed is None:
        seed = np.random.uniform(0, 1000, 1)
    a = np.random.uniform(0.05, 0.95, (n_iters,))
    l = np.random.uniform(0.05, 0.95, (n_iters,))
    best = []
    par = []
    p_values = np.asarray(p_values)
    p_values[p_values < 10**(-6)] = 10**(-6)  # optimiser is stuck when p-values == 0
    for i in range(n_iters):
        pars = np.array((a[i], l[i]))
        opt = minimize(_fbumnLL, pars, method='L-BFGS-B', args=(p_values,),
                       jac=_fpLL, bounds=((0.00001, 1), (0.00001, 1)))
        best.append(opt.fun)
        par.append(opt.x)
    minind = best.index(np.nanmin(best))
    a, lambda_ = par[minind]
    pi1 = 1 - (lambda_ + (1 - lambda_) * a)
    out = {'maxloglikelihood': best[minind],
           'pi1': pi1,
           'a': a,
           'lambda': lambda_}
    return out

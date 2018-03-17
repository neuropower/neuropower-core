#!/usr/bin/env python

"""
Fit a mixture model to a list of peak height T-values.
The model is introduced in the HBM poster:
http://www2.warwick.ac.uk/fac/sci/statistics/staff/academic-research/nichols/presentations/ohbm2015/Durnez-PeakPower-OHBM2015.pdf
"""
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy import integrate
from scipy.stats import norm, beta
from scipy.stats import t as tdist
from scipy.optimize import minimize

from neuropower import cluster, peakdistribution, BUM


def altPDF(peaks, mu, sigma=None, thresh=None, method="RFT"):
    """
    Returns probability density using a truncated normal
    distribution that we define as the distribution of local maxima in a
    GRF under the alternative hypothesis of activation

    Parameters
    ----------
    peaks : :obj:`float` or :obj:`list` of :obj:`float`
        List of peak heights.
    mu : :obj:`float`
        Stuff
    sigma : :obj:`float`, optional
        Stuff
    thresh : :obj:
        Stuff
    method : {'RFT', 'CS'}
        Stuff

    Returns
    -------
    fa : :obj:`float` or :obj:`list`
        Probability density of the peaks heights under Ha.
    """
    # Returns probability density of the alternative peak distribution
    peaks = np.asarray(peaks)
    if method == "RFT":
        # assert type(sigma) is in [float, int]
        # assert sigma is not None
        ksi = (peaks - mu) / sigma
        alpha = (thresh - mu) / sigma
        num = 1. / sigma * norm.pdf(ksi)
        den = 1. - norm.cdf(alpha)
        fa = num / den
    elif method == "CS":
        fa = [peakdistribution.peakdens3D(y - mu, 1) for y in peaks]
    else:
        raise ValueError('Argument `method` must be either "RFT" or "CS"')
    return fa


def nulPDF(peaks, thresh=None, method='RFT'):
    """
    Returns probability density of the null peak distribution.

    Parameters
    ----------
    peaks : :obj:`numpy.ndarray`
        Stuff
    thresh : :obj:`float`, optional
        Stuff
    method : {'RFT', 'CS'}
        Stuff
    """
    peaks = np.asarray(peaks)
    if method == 'RFT':
        f0 = thresh * np.exp(-thresh * (peaks - thresh))
    elif method == 'CS':
        f0 = [peakdistribution.peakdens3D(x, 1) for x in peaks]
    else:
        raise ValueError('Argument `method` must be either "RFT" or "CS"')
    return f0


def altCDF(peaks, mu, sigma=None, thresh=None, method="RFT"):
    # Returns the CDF of the alternative peak distribution
    peaks = np.asarray(peaks)
    if method == "RFT":
        ksi = (peaks - mu) / sigma
        alpha = (thresh - mu) / sigma
        Fa = (norm.cdf(ksi) - norm.cdf(alpha)) / (1 - norm.cdf(alpha))
    elif method == "CS":
        Fa = [integrate.quad(lambda x:peakdistribution.peakdens3D(x, 1), -20, y)[0] for y in peaks-mu]
    else:
        raise ValueError('Argument `method` must be either "RFT" or "CS"')
    return Fa


def TruncTau(mu, sigma, thresh):
    num = norm.cdf((thresh - mu) / sigma)
    den = 1 - norm.pdf((thresh - mu) / sigma)
    tau = num / den
    return tau


def _nulCDF(peaks, thresh=None, method="RFT"):
    # Returns the CDF of the null peak distribution
    peaks = np.asarray(peaks)
    if method == "RFT":
        F0 = 1 - np.exp(-thresh * (peaks - thresh))
    elif method == "CS":
        F0 = [integrate.quad(lambda x:peakdistribution.peakdens3D(x, 1), -20, y)[0] for y in peaks]
    else:
        raise ValueError('Argument `method` must be either "RFT" or "CS"')
    return F0


def mixPDF(peaks, pi1, mu, sigma=None, thresh=None, method="RFT"):
    """
    Returns the PDF of the mixture of null and alternative distribution.

    Parameters
    ----------
    peaks : :obj:`numpy.ndarray`
        A list of p-values associated with local maxima in the input image.
    pi1 : :obj:`float`

    """
    peaks = np.array(peaks)
    if method == "RFT":
        f0 = nulPDF(peaks, thresh=thresh, method="RFT")
        fa = altPDF(peaks, mu, sigma=sigma, thresh=thresh, method="RFT")
    elif method == "CS":
        f0 = nulPDF(peaks, method="CS")
        fa = altPDF(peaks, mu, method="CS")
    else:
        raise ValueError('Argument `method` must be either "RFT" or "CS"')
    f = [(1-pi1) * x + pi1 * y for x, y in zip(f0, fa)]
    return f


def _mixPDF_SLL(pars, peaks, pi1, thresh=None, method="RFT"):
    # Returns the negative sum of the loglikelihood of the PDF with RFT
    mu = pars[0]
    if method == "RFT":
        sigma = pars[1]
        f = mixPDF(peaks, pi1=pi1, mu=mu, sigma=sigma, thresh=thresh, method="RFT")
    elif method == "CS":
        f = mixPDF(peaks, pi1=pi1, mu=mu, method="CS")
    else:
        raise ValueError('Argument `method` must be either "RFT" or "CS"')
    LL = -sum(np.log(f))
    return LL


def modelfit(peaks, pi1, thresh=None, n_iters=1, seed=None, method="RFT"):
    """
    Searches the maximum likelihood estimator for the mixture distribution of
    null and alternative.

    Parameters
    ----------
    peaks : :obj:`numpy.ndarray`
        1D array of z-values from peaks in statistical map.
    pi1 : :obj:`float` in (0, 1)

    thresh : :obj:`float`
        Voxel-level z-threshold.
    """
    peaks = np.asarray(peaks)
    if seed is None:
        seed = np.random.uniform(0, 1000, 1)
    mus = np.random.uniform(thresh+(1./thresh),10,(n_iters,)) if method == "RFT" else np.random.uniform(0,10,(n_iters,))
    sigmas = np.random.uniform(0.1, 10, (n_iters,)) if method=="RFT" else np.repeat(None, n_iters)
    best = []
    par = []
    for i in range(n_iters):
        if method == "RFT":
            opt = minimize(_mixPDF_SLL, [mus[i], sigmas[i]], method='L-BFGS-B',
                           args=(peaks, pi1, thresh, method),
                           bounds=((thresh + (1. / thresh), 50), (0.1, 50)))
        elif method == "CS":
            opt = minimize(_mixPDF_SLL, [mus[i]], method='L-BFGS-B',
                           args=(peaks, pi1, thresh, method), bounds=((0, 50),))
        else:
            raise ValueError('Argument `method` must be either "RFT" or "CS"')
        best.append(opt.fun)
        par.append(opt.x)
    minind = best.index(np.nanmin(best))
    out = {'maxloglikelihood': best[minind],
           'mu': par[minind][0],
           'sigma': par[minind][1] if method == 'RFT' else 'nan'}
    return out


def threshold(pvalues, fwhm, voxsize, n_voxels, alpha=0.05, thresh=None):
    """Threshold p-values from peaks.

    thresh : :obj:`float`
        Cluster defining threshold in Z.
    """
    # only RFT
    peakrange = np.arange(thresh, 15, 0.001)
    pN = 1-_nulCDF(np.array(peakrange), thresh=thresh)
    # smoothness
    FWHM_vox = np.asarray(fwhm)/np.asarray(voxsize)
    resels = n_voxels/np.product(FWHM_vox)
    pN_RFT = resels*np.exp(-peakrange**2/2)*peakrange**2
    cutoff_UN = np.min(peakrange[pN<alpha])
    cutoff_BF = np.min(peakrange[pN<alpha/len(pvalues)])
    cutoff_RFT = np.min(peakrange[pN_RFT<alpha])
    #Benjamini-Hochberg
    pvals_sortind = np.argsort(pvalues)
    pvals_order = pvals_sortind.argsort()
    FDRqval = pvals_order/float(len(pvalues))*0.05
    reject = pvalues < FDRqval
    if reject.any():
        FDRc = np.max(pvalues[reject])
    else:
        FDRc = 0
    cutoff_BH = 'nan' if FDRc == 0 else min(peakrange[pN < FDRc])
    out = {'UN': cutoff_UN,
           'BF': cutoff_BF,
           'RFT': cutoff_RFT,
           'BH': cutoff_BH}
    return out


def BH(pvals, alpha):
    pvals_sortind = np.argsort(pvals)
    pvals_order = pvals_sortind.argsort()
    FDRqval = pvals_order / float(len(pvals)) * alpha
    reject = pvals < FDRqval
    if np.sum(reject) == 0:
        FDRc = 0
    else:
        FDRc = np.max(pvals[reject])
    return FDRc


def run_power_analysis(input_img, mask_img=None, dtype='t', n=0, design='one-sample',
                       cdt=0.001, alpha=0.05, correction='RFT', n_iters=1000,
                       seed=None, fwhm=[8, 8, 8]):
    spm = input_img.get_data()
    voxel_size = input_img.header.get_zooms()
    if mask_img is not None:
        mask = mask_img.get_data()
    else:
        mask = (spm != 0).astype(int)
    n_voxels = np.sum(mask)

    z_u = norm.ppf(1 - cdt)
    if dtype == 'z':
        spm_z = spm.copy()
    elif dtype == 't':
        spm_z = -norm.ppf(tdist.cdf(-spm, df=float(n-1)))
        spm_p = tdist.sf(spm, n-1)
        spm_z = -norm.ppf(spm_p)
    peak_df = cluster.PeakTable(spm_z, z_u, mask)
    z_values = peak_df['peak'].values
    p_values = norm.sf(abs(z_values))
    p_values[p_values<10**-6] = 10**-6
    peak_df['pval'] = p_values
    p_values2 = np.exp(-float(z_u)*(np.array(z_values)-float(z_u)))
    p_values2 = np.array([max(10**(-6), p) for p in p_values2])
    p_values = p_values2[:]
    #peak_df['pval2'] = p_values2

    out1 = BUM.EstimatePi1(p_values, n_iters=n_iters)
    out2 = modelfit(z_values, pi1=out1['pi1'], thresh=z_u,
                    n_iters=n_iters, seed=seed, method=correction)
    params = {}
    params['p_values'] = p_values
    params['z_values'] = z_values
    params['z_u'] = z_u
    params['a'] = out1['a']
    params['pi1'] = out1['pi1']
    params['lambda'] = out1['lambda']
    params['mu'] = out2['mu']
    params['sigma'] = out2['sigma']
    params['mu_s'] = params['mu'] / np.sqrt(n)

    thresholds = threshold(p_values, fwhm, voxel_size, n_voxels, alpha, z_u)
    powerpred_all = []
    test_ns = range(n, n+600)
    for s in test_ns:
        projected_effect = params['mu_s'] * np.sqrt(s)

        powerpred_s = {}
        for k, v in thresholds.items():
            if not v == 'nan':
                powerpred_s[k] = 1 - altCDF([v], projected_effect, params['sigma'],
                                            params['z_u'], correction)[0]
        powerpred_s['sample size'] = s
        powerpred_all.append(powerpred_s)
    power_df = pd.DataFrame(powerpred_all)
    power_df = power_df.set_index('sample size', drop=True)
    power_df = power_df.loc[(power_df[power_df.columns]<1).all(axis=1)]
    return params, power_df


def plot_stuff(d):
    p_values = d['p_values']
    z_values = d['z_values']
    z_u = d['z_u']
    a = d['a']
    pi1 = d['pi1']
    mu = d['mu']
    sigma = d['sigma']
    mu_s = d['mu_s']
    fig, axes = plt.subplots(ncols=2, figsize=(16, 7))

    # p-values
    axes[0].hist(p_values, bins=np.arange(0,1.1,0.1), normed=True,
                 alpha=0.6, label='observed distribution')
    axes[0].axhline(1-pi1, color='g', lw=5, alpha=0.6, label='null distribution')

    x_min, x_max = np.floor(np.min(p_values)), np.ceil(np.max(p_values))
    x = np.linspace(x_min, x_max, 100)
    y_a = (pi1 * beta.pdf(x, a=a, b=1)) + 1 - pi1
    axes[0].plot(x, y_a, 'r-', lw=5, alpha=0.6, label='alternative distribution')

    axes[0].set_xlabel('Peak p-values', fontsize=16)
    axes[0].set_title('Distribution of {0} peak p-values'
                      '\n$\pi_1$={1:0.03f}'.format(len(p_values), pi1),
                      fontsize=20)
    legend = axes[0].legend(frameon=True, fontsize=14)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    axes[0].set_xlim((0, 1))

    # Z-values
    y, _, _ = axes[1].hist(z_values, bins=np.arange(min(z_values),30,0.3),
                           normed=True, alpha=0.6, label='observed distribution')
    x_min, x_max = np.floor(np.min(z_values)), np.ceil(np.max(z_values))
    y_max = np.ceil(y.max())

    x = np.linspace(x_min, x_max, 100)
    y_0 = (1-pi1)*nulPDF(x, thresh=z_u, method="RFT")
    y_a = pi1*altPDF(x, mu=mu, sigma=sigma, thresh=z_u, method="RFT")
    y_m = mixPDF(x, pi1=pi1, mu=mu, sigma=sigma, thresh=z_u, method="RFT")

    axes[1].plot(x, y_a, 'r-', lw=5, alpha=0.6, label='alternative distribution')
    axes[1].plot(x, y_0, 'g-', lw=5, alpha=0.6, label='null distribution')
    axes[1].plot(x, y_m, 'b-', lw=5, alpha=0.6, label='total distribution')

    axes[1].set_title('Distribution of peak heights\n$\delta_1$ '
                      '= {0:0.03f}'.format(mu_s),
                      fontsize=20)
    axes[1].set_xlabel('Peak heights (z-values)', fontsize=16)
    axes[1].set_ylabel('Density', fontsize=16)
    legend = axes[1].legend(frameon=True, fontsize=14)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    axes[1].set_xlim((min(z_values), x_max))
    axes[1].set_ylim((0, y_max))

    return fig, axes

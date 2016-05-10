import scipy.stats
import numpy as np
from neuropower import peakdistribution
import scipy.integrate as integrate

"""
Fit a mixture model to a list of peak height T-values.
The model is introduced in the HBM poster:
http://www2.warwick.ac.uk/fac/sci/statistics/staff/academic-research/nichols/presentations/ohbm2015/Durnez-PeakPower-OHBM2015.pdf
"""

def altPDF(peaks,mu,sigma=None,exc=None,method="RFT"):
	"""
	altPDF: Returns probability density using a truncated normal
	distribution that we define as the distribution of local maxima in a
	GRF under the alternative hypothesis of activation
	parameters
	----------
	peaks: float or list of floats
		list of peak heigths
	mu:
	sigma:

	returns
	-------
	fa: float or list
		probability density of the peaks heights under Ha
	"""
	#Returns probability density of the alternative peak distribution
	peaks = np.asarray(peaks)
	if method == "RFT":
		# assert type(sigma) is in [float, int]
		# assert sigma is not None
		ksi = (peaks-mu)/sigma
		alpha = (exc-mu)/sigma
		num = 1/sigma * scipy.stats.norm.pdf(ksi)
		den = 1. - scipy.stats.norm.cdf(alpha)
		fa = num/den
	elif method == "CS":
		fa = [peakdistribution.peakdens3D(y-mu,1) for y in peaks]
	return fa

def nulPDF(peaks,exc=None,method="RFT"):
	#Returns probability density of the null peak distribution
	peaks = np.asarray(peaks)
	if method == "RFT":
		f0 = exc*np.exp(-exc*(peaks-exc))
	elif method == "CS":
		f0 = [peakdistribution.peakdens3D(x,1) for x in peaks]
	return f0

def altCDF(peaks,mu,sigma=None,exc=None,method="RFT"):
	# Returns the CDF of the alternative peak distribution
	peaks = np.asarray(peaks)
	if method == "RFT":
		ksi = (peaks-mu)/sigma
		alpha = (exc-mu)/sigma
		Fa = (scipy.stats.norm.cdf(ksi) - scipy.stats.norm.cdf(alpha))/(1-scipy.stats.norm.cdf(alpha))
	elif method == "CS":
		Fa = [integrate.quad(lambda x:peakdistribution.peakdens3D(x,1),-20,y)[0] for y in peaks-mu]
	return Fa

def TruncTau(mu,sigma,exc):
	num = scipy.stats.norm.cdf((exc-mu)/sigma)
	den = 1-scipy.stats.norm.pdf((exc-mu)/sigma)
	tau = num/den
	return tau

def nulCDF(peaks,exc=None,method="RFT"):
	# Returns the CDF of the null peak distribution
	peaks = np.asarray(peaks)
	if method == "RFT":
		F0 = 1-np.exp(-exc*(peaks-exc))
	elif method == "CS":
		F0 = [integrate.quad(lambda x:peakdistribution.peakdens3D(x,1),-20,y)[0] for y in peaks]
	return F0

def mixPDF(peaks,pi1,mu,sigma=None,exc=None,method="RFT"):
	# returns the PDF of the mixture of null and alternative distribution
	peaks = np.array(peaks)
	if method == "RFT":
		f0=nulPDF(peaks,exc=exc,method="RFT")
		fa=altPDF(peaks,mu,sigma=sigma,exc=exc,method="RFT")
	elif method == "CS":
		f0 = nulPDF(peaks,method="CS")
		fa = altPDF(peaks,mu,method="CS")
	f=[(1-pi1)*x + pi1*y for x, y in zip(f0, fa)]
	return(f)

def mixPDF_SLL(pars,peaks,pi1,exc=None,method="RFT"):
	# Returns the negative sum of the loglikelihood of the PDF with RFT
	mu = pars[0]
	if method == "RFT":
		sigma = pars[1]
		f = mixPDF(peaks,pi1=pi1,mu=mu,sigma=sigma,exc=exc,method="RFT")
	elif method == "CS":
		f = mixPDF(peaks,pi1=pi1,mu=mu,method="CS")
	LL = -sum(np.log(f))
	return(LL)

def modelfit(peaks,pi1,exc=None,starts=1,seed=None,method="RFT"):
	# Searches the maximum likelihood estimator for the mixture distribution of null and alternative
	peaks = np.asarray(peaks)
	seed = np.random.uniform(0,1000,1) if not 'seed' in locals() else seed
	mus = np.random.uniform(exc+(1./exc),10,(starts,)) if method == "RFT" else np.random.uniform(0,10,(starts,))
	seed = np.random.uniform(0,1000,1) if not 'seed' in locals() else seed
	sigmas = np.random.uniform(0.1,10,(starts,)) if method=="RFT" else np.repeat(None,starts)
	best = []
	par = []
	for i in range(0,starts):
		if method == "RFT":
			opt = scipy.optimize.minimize(mixPDF_SLL,[mus[i],sigmas[i]],method='L-BFGS-B',args=(peaks,pi1,exc,method),bounds=((exc+(1./exc),50),(0.1,50)))
		elif method == "CS":
			opt = scipy.optimize.minimize(mixPDF_SLL,[mus[i]],method='L-BFGS-B',args=(peaks,pi1,exc,method),bounds=((0,50),))
		best.append(opt.fun)
		par.append(opt.x)
	minind=best.index(np.nanmin(best))
	out={'maxloglikelihood': best[minind],
			'mu': par[minind][0],
			'sigma': par[minind][1] if method == "RFT" else 'nan'}
	return out

def threshold(peaks,pvalues,FWHM,voxsize,nvox,alpha=0.05,exc=None,method="RFT"):
	# only RFT
	peakrange = np.arange(exc,15,0.001)
	pN = 1-nulCDF(np.array(peakrange),exc=exc)
	# smoothness
	FWHM_vox = np.asarray(FWHM)/np.asarray(voxsize)
	resels = nvox/np.product(FWHM_vox)
	pN_RFT = resels*np.exp(-peakrange**2/2)*peakrange**2
	cutoff_UN = np.min(peakrange[pN<alpha])
	cutoff_BF = np.min(peakrange[pN<alpha/len(peaks)])
	cutoff_RFT = np.min(peakrange[pN_RFT<alpha])
	#Benjamini-Hochberg
	pvals_sortind = np.argsort(pvalues)
	pvals_order = pvals_sortind.argsort()
	FDRqval = pvals_order/float(len(pvalues))*0.05
	reject = pvalues<FDRqval
	if reject.any():
		FDRc = np.max(pvalues[reject])
	else:
		FDRc = 0
	cutoff_BH = 'nan' if FDRc==0 else min(peakrange[pN<FDRc])
	out = {'UN':cutoff_UN,'BF':cutoff_BF,'RFT':cutoff_RFT,'BH':cutoff_BH}
	return out

def BH(pvals,alpha):
	pvals_sortind = np.argsort(pvals)
	pvals_order = pvals_sortind.argsort()
	FDRqval = pvals_order/float(len(pvals))*alpha
	reject = pvals<FDRqval
	if np.sum(reject)==0:
		FDRc=0
	else:
		FDRc = np.max(pvals[reject])
	return FDRc

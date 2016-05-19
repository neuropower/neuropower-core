#!/usr/bin/env python

"""
Fit a beta-uniform mixture model to a list of p-values.
The BUM model is introduced in Pounds & Morris, 2003.
"""

import numpy as np
import scipy

def fpLL(pars,x):
	# Returns the gradient function of the BUM model
	a = pars[0]
	l = pars[1]
	dl = -sum((1-a*x**(a-1))/(a*(1-l)*x**(a-1)+l))
	da = -sum((a*(1-l)*x**(a-1)*np.log(x)+(1-l)*x**(a-1))/(a*(1-l)*x**(a-1)+l))
	return np.asarray([dl,da])

def fbumnLL(pars,x):
	#Returns the negative sum of the loglikelihood
	a = pars[0]
	l = pars[1]
	L = l+(1-l)*a*x**(a-1)
	negsumlog = -sum(np.log(L))
	return(negsumlog)

def EstimatePi1(x,starts=10,seed=None):
	# Returns the MLE estimator for pi1, with the shaping parameters and the value of the negative sum of the loglikelihood
	"""Searches the maximum likelihood estimator for the shape parameters of the BUM-model given a list of p-values"""
	seed = np.random.uniform(0,1000,1) if not 'seed' in locals() else seed
	a = np.random.uniform(0.05,0.95,(starts,))
	l = np.random.uniform(0.05,0.95,(starts,))
	best = []
	par = []
	x = np.asarray(x)
	x = [10**(-6) if y<= 10**(-6) else y for y in x] #optimiser is stuck when p-values == 0
	for i in range(0,starts):
		pars = np.array((a[i],l[i]))
		opt = scipy.optimize.minimize(fbumnLL,[pars[0],pars[1]],method='L-BFGS-B',args=(x,),jac=fpLL,bounds=((0.00001,1),(0.00001,1)))
		best.append(opt.fun)
		par.append(opt.x)
	minind=best.index(np.nanmin(best))
	bestpar=par[minind]
	pi1=1-(bestpar[1] + (1-bestpar[1])*bestpar[0])
	out={'maxloglikelihood': best[minind],
		'pi1': pi1,
		'a': bestpar[0],
		'lambda': bestpar[1]}
	return(out)

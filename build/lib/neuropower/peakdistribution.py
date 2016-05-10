import scipy.stats as stats
import numpy as np

def peakdens3D(x,k):
    # returns the PDF of a peak
    fd1 = 144*stats.norm.pdf(x)/(29*6**(0.5)-36)
    fd211 = k**2.*((1.-k**2.)**3. + 6.*(1.-k**2.)**2. + 12.*(1.-k**2.)+24.)*x**2. / (4.*(3.-k**2.)**2.)
    fd212 = (2.*(1.-k**2.)**3. + 3.*(1.-k**2.)**2.+6.*(1.-k**2.)) / (4.*(3.-k**2.))
    fd213 = 3./2.
    fd21 = (fd211 + fd212 + fd213)
    fd22 = np.exp(-k**2.*x**2./(2.*(3.-k**2.))) / (2.*(3.-k**2.))**(0.5)
    fd23 = stats.norm.cdf(2.*k*x / ((3.-k**2.)*(5.-3.*k**2.))**(0.5))
    fd2 = fd21*fd22*fd23
    fd31 = (k**2.*(2.-k**2.))/4.*x**2. - k**2.*(1.-k**2.)/2. - 1.
    fd32 = np.exp(-k**2.*x**2./(2.*(2.-k**2.))) / (2.*(2.-k**2.))**(0.5)
    fd33 = stats.norm.cdf(k*x / ((2.-k**2.)*(5.-3.*k**2.))**(0.5))
    fd3 = fd31 * fd32 * fd33
    fd41 = (7.-k**2.) + (1-k**2)*(3.*(1.-k**2.)**2. + 12.*(1.-k**2.) + 28.)/(2.*(3.-k**2.))
    fd42 = k*x / (4.*np.pi**(0.5)*(3.-k**2.)*(5.-3.*k**2)**0.5)
    fd43 = np.exp(-3.*k**2.*x**2/(2.*(5-3.*k**2.)))
    fd4 = fd41*fd42 * fd43
    fd51 = np.pi**0.5*k**3./4.*x*(x**2.-3.)
    f521low = np.array([-10.,-10.])
    f521up = np.array([0.,k*x/2.**(0.5)])
    f521mu = np.array([0.,0.])
    f521sigma = np.array([[3./2., -1.],[-1.,(3.-k**2.)/2.]])
    fd521,i = stats.mvn.mvnun(f521low,f521up,f521mu,f521sigma)
    f522low = np.array([-10.,-10.])
    f522up = np.array([0.,k*x/2.**(0.5)])
    f522mu = np.array([0.,0.])
    f522sigma = np.array([[3./2., -1./2.],[-1./2.,(2.-k**2.)/2.]])
    fd522,i = stats.mvn.mvnun(f522low,f522up,f522mu,f522sigma)
    fd5 = fd51*(fd521+fd522)
    out = fd1*(fd2+fd3+fd4+fd5)
    return out

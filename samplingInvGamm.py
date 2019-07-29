import numpy as np
import config
from scipy.stats import invgamma
import healpy as hp
#import utils


def sampling(s):
    #Input alm for m >= 0 in m major
    #Output: sampled Cls, from the conditional
    cls = []
    sigmas = hp.sphtfunc.alm2cl(s, lmax=config.L_MAX_SCALARS)
    for l, sigma in enumerate(sigmas):
        if l >= 2:
            beta = (2*l+1)*sigma/2
            alpha = (2*l-1)/2
            sampled_cl = beta*invgamma.rvs(a=alpha)
            cls.append(sampled_cl)

    return np.concatenate((np.zeros(2), np.array(cls)))

import numpy as np
import config
import healpy as hp

#N_iter = 100000
N_iter = 1000000


def compute_log_likelihood(x_pix, observations):
    return -(1/2)*np.sum(((observations - x_pix)**2)*(1/config.noise_covar))


def compute_CN_ratio(x_pix, y_pix, observations):
    return compute_log_likelihood(y_pix, observations) - compute_log_likelihood(x_pix, observations)


def crankNicolson(cls_, observations):
    accept = 0
    history = []
    s = np.zeros(config.dimension_sph).astype(complex)
    s_pixel = hp.sphtfunc.alm2map(s, nside=config.NSIDE)
    for i in range(N_iter):
        history.append(s)
        prop = np.sqrt(1-config.beta_CN**2)*s + config.beta_CN*hp.sphtfunc.synalm(cls_, lmax=config.L_MAX_SCALARS)
        prop_pix = hp.sphtfunc.alm2map(prop, nside=config.NSIDE)
        r = compute_CN_ratio(s_pixel, prop_pix, observations)
        #r = 1
        if np.log(np.random.uniform()) < r:
            s = prop
            s_pixel = prop_pix
            accept += 1


    print(accept/N_iter)
    return s, history

import numpy as np
import config
import healpy as hp
import matplotlib.pyplot as plt
import gradientDescent


def compute_log_likelihood(x_pix, observations):
    return -(1/2)*np.sum(((observations - x_pix)**2)*(1/config.noise_covar))


def compute_CN_ratio(x_pix, y_pix, observations):
    return compute_log_likelihood(y_pix, observations) - compute_log_likelihood(x_pix, observations)


def crankNicolson(cls_, observations):
    accept = 0
    history = []
    #s = hp.synalm(cls_, lmax=config.L_MAX_SCALARS)
    h, s = gradientDescent.gradient_ascent(observations, cls_)
    s_pixel = hp.sphtfunc.alm2map(s, nside=config.NSIDE)
    for i in range(config.N_CN):
        history.append(s)
        prop = np.sqrt(1-config.beta_CN**2)*s + config.beta_CN*hp.sphtfunc.synalm(cls_, lmax=config.L_MAX_SCALARS)
        prop_pix = hp.sphtfunc.alm2map(prop, nside=config.NSIDE)
        r = compute_CN_ratio(s_pixel, prop_pix, observations)
        if np.log(np.random.uniform()) < r:
            s = prop
            s_pixel = prop_pix
            accept += 1

    print(accept/config.N_CN)
    return history, s

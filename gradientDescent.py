import numpy as np
import healpy as hp
import config

N_grad = 5
step_size = 0.00005

def compute_gradient_log_constant_part(observations):
    temp = (1/config.noise_covar)*observations
    return np.real(hp.sphtfunc.map2alm(temp, lmax=config.L_MAX_SCALARS))


def compute_gradient_log(s, s_pix, grad_constant_part, extended_cls):
    intermediate = (1/config.noise_covar)*s_pix
    first_part = hp.sphtfunc.map2alm(intermediate, lmax=config.L_MAX_SCALARS)
    denom = np.concatenate([np.zeros(2), 1/extended_cls[2:(config.L_MAX_SCALARS+1)], np.zeros(1), 1/extended_cls[(config.L_MAX_SCALARS+2):]])
    second_part = denom*s
    grad_variable = first_part + second_part
    return grad_constant_part - grad_variable


def gradient_ascent(observations, cls):
    history = []
    extended_cls = np.array([cl for l in range(config.L_MAX_SCALARS + 1) for cl in cls[l:]])
    grad_constant_part = compute_gradient_log_constant_part(observations)
    s = np.zeros(config.dimension_sph)
    s_pix = hp.sphtfunc.alm2map(s.astype(complex), nside=config.NSIDE)
    for i in range(N_grad):
        history.append(s)
        grad_log = compute_gradient_log(s, s_pix, grad_constant_part, extended_cls)
        s = s + step_size*grad_log
        s_pix = hp.alm2map(s, nside=config.NSIDE)

    return history, s


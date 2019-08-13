import numpy as np
import healpy as hp
import config
import matplotlib.pyplot as plt
import conjugateGradient
import utils

N_grad = 100
step_size = 0.00005

def compute_gradient_log_constant_part(observations):
    temp = (1/config.noise_covar)*observations
    return hp.sphtfunc.map2alm(temp, lmax=config.L_MAX_SCALARS)


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
    #s = np.zeros(config.dimension_sph)
    s = hp.synalm(cls, lmax=config.L_MAX_SCALARS)
    s_pix = hp.sphtfunc.alm2map(s, nside=config.NSIDE)
    for i in range(N_grad):
        history.append(s)
        grad_log = compute_gradient_log(s, s_pix, grad_constant_part, extended_cls)

        s = s + step_size*grad_log
        s_pix = hp.alm2map(s, nside=config.NSIDE)

    return history, s


###### Second gradient ascent


def flatten_map(s):
    s_real = s.real[[i for i in range(len(s)) if i != 0 and i != 1 and i != (config.L_MAX_SCALARS+1)]]
    s_imag = s.imag[[i for i in range((config.L_MAX_SCALARS+2),len(s))]]
    s_flatten = np.concatenate((s_real, s_imag))
    return s_flatten


def compute_gradient_log_constant_part2(observations):
    temp = (1/config.noise_covar)*observations
    return flatten_map(hp.sphtfunc.map2alm(temp, lmax=config.L_MAX_SCALARS))


def compute_gradient_log2(s, s_pix, grad_constant_part, extended_cls):
    intermediate = (1/config.noise_covar)*s_pix
    first_part = flatten_map(hp.sphtfunc.map2alm(intermediate, lmax=config.L_MAX_SCALARS))
    second_part = (1/np.array(extended_cls))*s
    grad_variable = first_part + second_part
    return grad_constant_part - grad_variable


def extend_cls(cls):
    extended_cls = [cl for l in range(config.L_MAX_SCALARS+1) for cl in cls[l:]]
    extended_cls_real = (extended_cls[:(config.L_MAX_SCALARS+1)] + extended_cls[(config.L_MAX_SCALARS+2):])[2:]
    extended_cls_imag = extended_cls[(config.L_MAX_SCALARS+2):]
    extended_cls = extended_cls_real + extended_cls_imag
    return np.array(extended_cls)


def unflat_map_to_pix(s):
    real_part = np.concatenate((np.zeros(2), s[:(config.L_MAX_SCALARS-1)] , np.zeros(1),
                                 s[(config.L_MAX_SCALARS-1):(config.dimension_sph-3)]))

    imag_part = np.concatenate((np.zeros(config.L_MAX_SCALARS+2), s[(config.dimension_sph-3):]))
    return real_part + 1j*imag_part


def gradient_ascent2(observations, cls):
    history = []
    extended_cls = extend_cls(cls)
    grad_constant_part = compute_gradient_log_constant_part2(observations)
    #s = np.zeros(config.dimension_sph)
    s = hp.synalm(cls, lmax=config.L_MAX_SCALARS)
    s_pix = hp.sphtfunc.alm2map(s, nside=config.NSIDE)
    s = flatten_map(s)
    for i in range(N_grad):
        history.append(conjugateGradient.flatten_map3(conjugateGradient.unflat_map_to_pix2(s)))
        grad_log = compute_gradient_log2(s, s_pix, grad_constant_part, extended_cls)

        s = s + step_size*grad_log
        s_pix = hp.alm2map(unflat_map_to_pix(s), nside=config.NSIDE)

    return history, s
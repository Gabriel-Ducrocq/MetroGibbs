import numpy as np
import healpy as hp
import config
import gradientDescent


def compute_gradient_log_constant_part(observations):
    temp = (1/config.noise_covar)*observations
    return np.real(hp.sphtfunc.map2alm(temp, lmax=config.L_MAX_SCALARS))


def compute_log_proposal(x,y, grad_log_x):
    a = (y - (x + config.step_size_mala*grad_log_x))**2
    b = (1 / (2 * config.step_size_mala * config.var_mala))
    prod = -(1/2)*np.sum(a*b)
    prod = prod.real
    return prod


def compute_log_density(x, x_pix, observations, extended_cls):
    denom = np.concatenate([np.zeros(2), 1/extended_cls[2:(config.L_MAX_SCALARS+1)], np.zeros(1), 1/extended_cls[(config.L_MAX_SCALARS+2):]])
    intermediate = -(1 / 2)*np.sum(denom*(x**2))
    intermediate = intermediate.real
    return -(1/2)*np.sum(((observations - x_pix)**2)*(1/config.noise_covar)) + intermediate


def compute_gradient_log(s, s_pix, grad_constant_part, extended_cls):
    intermediate = (1/config.noise_covar)*s_pix
    first_part = hp.sphtfunc.map2alm(intermediate, lmax=config.L_MAX_SCALARS)
    denom = np.concatenate([np.zeros(2), 1/extended_cls[2:(config.L_MAX_SCALARS+1)], np.zeros(1), 1/extended_cls[(config.L_MAX_SCALARS+2):]])
    second_part = denom*s
    grad_variable = first_part + second_part
    return grad_constant_part - grad_variable


def proposal(gradient_log, s):
    return s + config.step_size_mala*gradient_log + np.sqrt(config.var_mala)*np.sqrt(2*config.step_size_mala)*np.random.normal(size=config.dimension_sph)


def compute_MH_ratio(grad_log, grad_log_prop, s, s_pix, prop, prop_pix, observations, cls):
    return compute_log_density(prop, prop_pix, observations, cls) + compute_log_proposal(prop, s, grad_log_prop) - compute_log_density(s, s_pix, observations, cls) - compute_log_proposal(s, prop, grad_log)


def mala(cls, observations, grad_constant_part):
    extended_cls = np.array([cl for l in range(config.L_MAX_SCALARS+1) for cl in cls[l:]])
    acceptance = 0
    #s = hp.synalm(cls, lmax=config.L_MAX_SCALARS)
    h, s = gradientDescent.gradient_ascent(observations, cls)
    s_pixel = hp.sphtfunc.alm2map(s, nside=config.NSIDE)
    grad_log_s = compute_gradient_log(s, s_pixel, grad_constant_part, extended_cls)
    history = []
    for i in range(config.N_mala):
        history.append(s)
        s_prop = proposal(grad_log_s, s)
        s_prop_pix = hp.sphtfunc.alm2map(s, nside=config.NSIDE)
        grad_log_prop = compute_gradient_log(s_prop, s_prop_pix, grad_constant_part, extended_cls)
        r = compute_MH_ratio(grad_log_s, grad_log_prop, s, s_pixel, s_prop, s_prop_pix, observations, extended_cls)
        if np.log(np.random.uniform()) < r:
            s = s_prop
            s_pixel = s_prop_pix
            grad_log_s = grad_log_prop
            acceptance += 1

    print("Acceptance rate:")
    print(acceptance/config.N_mala)
    return history, s









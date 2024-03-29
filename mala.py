import numpy as np
import healpy as hp
import config
import gradientDescent
import utils
from conjugateGradient import CG


def compute_gradient_log_constant_part(observations):
    temp = (1/config.noise_covar)*observations
    return hp.sphtfunc.map2alm(temp, lmax=config.L_MAX_SCALARS)


def compute_log_proposal(x,y, grad_log_x):
    a = np.sqrt((1 / (2 * config.step_size_mala * config.var_mala)))*(y - (x + config.step_size_mala*grad_log_x))
    log_prop = -(1/2)*np.sum((np.concatenate((a.real, a.imag))**2))
    return log_prop


def compute_log_density(x, x_pix, observations, extended_cls):
    denom = np.concatenate([np.zeros(2), 1/extended_cls[2:(config.L_MAX_SCALARS+1)], np.zeros(1), 1/extended_cls[(config.L_MAX_SCALARS+2):]])
    x = np.sqrt(denom)*x
    intermediate = -(1/2)*np.sum((np.concatenate((x.real, x.imag))**2))
    return -(1/2)*np.sum(((observations - x_pix)**2)*(1/config.noise_covar)) + intermediate


def compute_gradient_log(s, s_pix, grad_constant_part, extended_cls):
    intermediate = (1/config.noise_covar)*s_pix
    first_part = hp.sphtfunc.map2alm(intermediate, lmax=config.L_MAX_SCALARS)
    denom = np.concatenate([np.zeros(2), 1/extended_cls[2:(config.L_MAX_SCALARS+1)], np.zeros(1), 1/extended_cls[(config.L_MAX_SCALARS+2):]])
    second_part = denom*s
    grad_variable = first_part + second_part
    return grad_constant_part - grad_variable


def proposal(gradient_log, s):
    ## Il faut aussi proposer sur la partie imaginaire !!
    return s + config.step_size_mala*gradient_log \
           + np.sqrt(config.var_mala)*np.sqrt(2*config.step_size_mala)*np.random.normal(size=config.dimension_sph) \
           + 1j*np.sqrt(config.var_mala)*np.sqrt(2*config.step_size_mala)*np.random.normal(size=config.dimension_sph)


def compute_MH_ratio(grad_log, grad_log_prop, s, s_pix, prop, prop_pix, observations, cls):
    num = compute_log_density(prop, prop_pix, observations, cls) + compute_log_proposal(prop, s, grad_log_prop)
    denom = compute_log_density(s, s_pix, observations, cls) + compute_log_proposal(s, prop, grad_log)
    return num - denom


def mala(cls, observations, grad_constant_part):
    conjgrad = CG()
    extended_cls = np.array([cl for l in range(config.L_MAX_SCALARS+1) for cl in cls[l:]])
    acceptance = 0
    s = hp.synalm(cls, lmax=config.L_MAX_SCALARS)
    #h_g, s = gradientDescent.gradient_ascent(observations, cls)
    warm_start = s
    s_pixel = hp.sphtfunc.alm2map(s, nside=config.NSIDE)
    grad_log_s = compute_gradient_log(s, s_pixel, grad_constant_part, extended_cls)
    history = []
    history_ratio = []
    for i in range(config.N_mala):
        history.append(s)
        s_prop = proposal(grad_log_s, s)
        s_prop_pix = hp.sphtfunc.alm2map(s, nside=config.NSIDE)
        grad_log_prop = compute_gradient_log(s_prop, s_prop_pix, grad_constant_part, extended_cls)
        r = compute_MH_ratio(grad_log_s, grad_log_prop, s, s_pixel, s_prop, s_prop_pix, observations, extended_cls)
        history_ratio.append(r)
        r = 1
        if np.log(np.random.uniform()) < r:
            s = s_prop
            s_pixel = s_prop_pix
            grad_log_s = grad_log_prop
            acceptance += 1

    print("Acceptance rate:")
    print(acceptance/config.N_mala)
    return history, s, warm_start, history_ratio









################# Second implementation of mala


def compute_gradient_log_constant_part2(observations):
    temp = (1/config.noise_covar)*observations
    return flatten_map(hp.sphtfunc.map2alm(temp, lmax=config.L_MAX_SCALARS))


def compute_log_proposal2(x,y, grad_log_x):
    log_prop = (1 / (2 * config.step_size_mala))*np.sum((y - (x + config.step_size_mala*grad_log_x))**2)
    return log_prop


def compute_log_density2(x, x_pix, observations, extended_cls):
    intermediate = -(1/2)*np.sum((1/np.array(extended_cls))*(x**2))
    return -(1/2)*np.sum(((observations - x_pix)**2)*(1/config.noise_covar)) + intermediate


def compute_gradient_log2(s, s_pix, grad_constant_part, extended_cls):
    intermediate = (1/config.noise_covar)*s_pix
    first_part = flatten_map(hp.sphtfunc.map2alm(intermediate, lmax=config.L_MAX_SCALARS))
    second_part = (1/np.array(extended_cls))*s
    grad_variable = first_part + second_part
    return grad_constant_part - grad_variable


def proposal2(gradient_log, s):
    ## Il faut aussi proposer sur la partie imaginaire !!
    return s + config.step_size_mala*gradient_log \
           + np.concatenate((np.sqrt(2*config.step_size_mala)*np.random.normal(size=(config.dimension_sph-3)),
           np.sqrt(2*config.step_size_mala)*np.random.normal(size=config.dimension_sph - (config.L_MAX_SCALARS + 2))))


def compute_MH_ratio2(grad_log, grad_log_prop, s, s_pix, prop, prop_pix, observations, cls):
    num = compute_log_density2(prop, prop_pix, observations, cls) + compute_log_proposal2(prop, s, grad_log_prop)
    denom = compute_log_density2(s, s_pix, observations, cls) + compute_log_proposal2(s, prop, grad_log)
    return num - denom


def flatten_map(s):
    s_real = s.real[[i for i in range(len(s)) if i != 0 and i != 1 and i != (config.L_MAX_SCALARS+1)]]
    s_imag = s.imag[[i for i in range((config.L_MAX_SCALARS+2),len(s))]]
    s_flatten = np.concatenate((s_real, s_imag))
    return s_flatten


def unflat_map_to_pix(s):
    real_part = np.concatenate((np.zeros(2), s[:(config.L_MAX_SCALARS-1)] , np.zeros(1),
                                 s[(config.L_MAX_SCALARS-1):(config.dimension_sph-3)]))

    imag_part = np.concatenate((np.zeros(config.L_MAX_SCALARS+2), s[(config.dimension_sph-3):]))
    return real_part + 1j*imag_part


def extend_cls(cls):
    extended_cls = [cl for l in range(config.L_MAX_SCALARS+1) for cl in cls[l:]]
    extended_cls_real = (extended_cls[:(config.L_MAX_SCALARS+1)] + extended_cls[(config.L_MAX_SCALARS+2):])[2:]
    extended_cls_imag = extended_cls[(config.L_MAX_SCALARS+2):]
    extended_cls = extended_cls_real + extended_cls_imag
    return np.array(extended_cls)


def mala2(cls, observations, grad_constant_part, unadjusted=True):
    extended_cls = extend_cls(cls)
    acceptance = 0
    #s = hp.synalm(cls, lmax=config.L_MAX_SCALARS)
    #s = np.zeros(config.dimension_sph)
    h_g, s = gradientDescent.gradient_ascent2(observations, cls)
    #s = flatten_map(s)
    warm_start = s
    s_pixel = hp.sphtfunc.alm2map(unflat_map_to_pix(s), nside=config.NSIDE)
    grad_log_s = compute_gradient_log2(s, s_pixel, grad_constant_part, extended_cls)
    history = []
    history_ratio = []
    history.append(s)
    for i in range(config.N_mala):
        s_prime = flatten_map4(unflat_map_to_pix(s))
        history.append(s_prime)
        s_prop = proposal2(grad_log_s, s)
        imag_map = unflat_map_to_pix(s_prop)
        s_prop_pix = hp.sphtfunc.alm2map(imag_map, nside=config.NSIDE)
        grad_log_prop = compute_gradient_log2(s_prop, s_prop_pix, grad_constant_part, extended_cls)
        r = compute_MH_ratio2(grad_log_s, grad_log_prop, s, s_pixel, s_prop, s_prop_pix, observations, extended_cls)
        history_ratio.append(r)
        if unadjusted:
            r = 1

        if np.log(np.random.uniform()) < r:
            s = s_prop
            s_pixel = s_prop_pix
            grad_log_s = grad_log_prop
            acceptance += 1

    print("Acceptance rate:")
    print(acceptance/config.N_mala)
    return history, s, warm_start, history_ratio


##### MALA 3

def compute_gradient_log_constant_part3(observations):
    temp = (1/config.noise_covar)*observations
    return flatten_map(hp.sphtfunc.map2alm(temp, lmax=config.L_MAX_SCALARS))


def compute_gradient_log3(s, s_pix, grad_constant_part, extended_cls):
    intermediate = (1/config.noise_covar)*s_pix
    first_part = flatten_map4(hp.sphtfunc.map2alm(intermediate, lmax=config.L_MAX_SCALARS))
    second_part = (1/np.array(extended_cls))*s
    grad_variable = first_part + second_part
    return grad_constant_part - grad_variable


def proposal3(s, grad_log_s, step_size):
    return s + step_size*grad_log_s + np.sqrt(2*step_size)*np.random.normal(size = len(s))


def compute_log_density3(s, s_pix, extended_cls, d):
    return -(1/2)*np.sum(((d - s_pix)**2)/config.noise_covar) - (1/2)*np.sum((s**2)/extended_cls)


def compute_log_proposal3(s, grad_log_s, s_prop, step_size):
    return -(1/2)*np.sum((s_prop - (s + step_size*grad_log_s)**2)/(2*step_size))


def compute_log_MH_ratio3(s, s_pix, grad_log_s, s_prop, s_prop_pix, grad_log_s_prop, d, extended_cls, step_size):
    return compute_log_density3(s_prop, s_prop_pix, extended_cls, d) + compute_log_proposal3(s_prop, grad_log_s_prop, s, step_size) \
    - (compute_log_density3(s, s_pix, extended_cls, d) + compute_log_proposal3(s, grad_log_s, s_prop, step_size))


def mala3(cls, d, grad_constant_part):
    extended_cls = extend_cls4(cls)
    s = hp.synalm(cls, lmax=config.L_MAX_SCALARS)
    s_pix = hp.sphtfunc.alm2map(s, nside=config.NSIDE)
    s = flatten_map4(s)
    grad_log_s = compute_gradient_log3(s, s_pix, grad_constant_part, extended_cls)
    history = []
    accepted = 0
    history.append(s)
    for i in range(config.N_mala):
        print(i)
        s_prop = proposal3(s, grad_log_s, step_size=config.step_size_mala)
        s_prop_pix = hp.alm2map(unflat_map_to_pix4(s_prop), nside =config.NSIDE)
        grad_log_s_prop = compute_gradient_log3(s_prop, s_prop_pix, grad_constant_part, extended_cls)
        r = compute_log_MH_ratio3(s, s_pix, grad_log_s, s_prop, s_prop_pix, grad_log_s_prop, d, extended_cls, config.step_size_mala)
        if np.log(np.random.uniform()) < r:
            s = s_prop
            s_pix = s_prop_pix
            grad_log_s = grad_log_s_prop
            accepted += 1

        history.append(s)

    print(accepted/config.N_mala)
    return history, s


def unflat_map_to_pix4(s):
    #real_part = np.concatenate((np.zeros(2), s[:(config.L_MAX_SCALARS-1)] , np.zeros(1),
    #                             s[(config.L_MAX_SCALARS-1):(config.dimension_sph-3)]))

    #imag_part = np.concatenate((np.zeros(config.L_MAX_SCALARS+2), s[(config.dimension_sph-3):]))
    imag_part = np.concatenate((np.zeros(config.L_MAX_SCALARS+2), s[:config.N-config.L_MAX_SCALARS-1-1]))
    real_part = s[config.N-config.L_MAX_SCALARS-1-1:]
    real_part = np.insert(real_part, [0, 0, config.L_MAX_SCALARS - 1], 0)
    return real_part + 1j*imag_part

def extend_cls4(cls):
    extended_cls = [cl for l in range(config.L_MAX_SCALARS+1) for cl in cls[l:]]
    extended_cls_real = (extended_cls[:(config.L_MAX_SCALARS+1)] + extended_cls[(config.L_MAX_SCALARS+2):])[2:]
    extended_cls_imag = extended_cls[(config.L_MAX_SCALARS+2):]
    extended_cls = extended_cls_imag+extended_cls_real
    return np.array(extended_cls)

def flatten_map4(s):
    s_real = s.real[[i for i in range(len(s)) if i != 0 and i != 1 and i != (config.L_MAX_SCALARS+1)]]
    s_imag = s.imag[[i for i in range((config.L_MAX_SCALARS+2),len(s))]]
    s_flatten = np.concatenate((s_imag, s_real))
    return s_flatten


#####  Final version

def proposal_good(s, grad_log_s):
    return s + config.step_size_mala*grad_log_s + np.sqrt(2*config.step_size_mala)*np.random.normal(size=len(s))


def compute_log_proposal_good(x,y, grad_log_x):
    log_prop = (1 / (2 * config.step_size_mala))*np.sum((y - (x + config.step_size_mala*grad_log_x))**2)
    return log_prop


def compute_log_density_good(x, x_pix, observations, extended_cls):
    intermediate = -(1/2)*np.sum((1/np.array(extended_cls))*(x**2))
    return -(1/2)*np.sum(((observations - x_pix)**2)*(1/config.noise_covar)) + intermediate


def compute_log_MH_ratio_good(grad_log, grad_log_prop, s, s_pix, prop, prop_pix, observations, cls):
    num = compute_log_density_good(prop, prop_pix, observations, cls) + compute_log_proposal2(prop, s, grad_log_prop)
    denom = compute_log_density_good(s, s_pix, observations, cls) + compute_log_proposal2(s, prop, grad_log)
    return num - denom


def mala_good(cls_, d, grad_constant_part, unadjusted=False):
    extended_cls = utils.extend_cls(cls_)
    #s = hp.synalm(cls_, lmax=config.L_MAX_SCALARS)
    _, s = gradientDescent.gradient_ascent_good(d, cls_)
    print("AAAAAAAAAAAAAAAAAAAAAAAA")
    print(s[30])
    s_pix = hp.sphtfunc.alm2map(utils.unflat_map_to_pix(s), nside=config.NSIDE)
    #s = utils.flatten_map(s)
    grad_log_s = utils.compute_gradient_log(s, s_pix, grad_constant_part, extended_cls)
    history = []
    accepted = 0
    for i in range(config.N_mala):
        history.append(s)
        s_prop = proposal_good(s, grad_log_s)
        s_prop_pix = hp.alm2map(utils.unflat_map_to_pix(s_prop), nside =config.NSIDE)
        grad_log_s_prop = utils.compute_gradient_log(s_prop, s_prop_pix, grad_constant_part, extended_cls)
        r = compute_log_MH_ratio_good(grad_log_s, grad_log_s_prop, s, s_pix, s_prop, s_prop_pix, d, extended_cls)
        if unadjusted:
            r = 0

        if np.log(np.random.uniform()) < r:
            s = s_prop
            s_pix = s_prop_pix
            grad_log_s = grad_log_s_prop
            accepted += 1


    print("Acceptance rate:")
    print(accepted / config.N_mala)
    return history, s
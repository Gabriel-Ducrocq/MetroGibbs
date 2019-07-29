import numpy as np
import healpy as hp
import config

NSIDE= 16
L_MAX_SCALARS=50
#N_iter = 100000
N_iter = 100000
dimension_sph = int((L_MAX_SCALARS*(L_MAX_SCALARS + 1)/2)+L_MAX_SCALARS+1)
step_size = 0.00000000000000008
#step_size = 0.000000000000000000001
#step_size = 0.0000000000001955

def compute_gradient_log_constant_part(observations):
    ### Attention: j'ai pris la partie réelle !!
    temp = (1/config.noise_covar)*observations
    return np.real(hp.sphtfunc.map2alm(temp.astype(complex), lmax=L_MAX_SCALARS))


def compute_log_proposal(x,y, grad_log_x):
    a = ((y - (x + step_size*grad_log_x))**2)
    b = (1 / (2 * step_size * config.var_mala))
    return -(1/2)*np.sum(a*b)


def compute_log_density(x, x_pix, observations, cls):
    #Be careful, the first 3 coefs of the extended cls are 0
    return -(1/2)*np.sum(((observations - x_pix)**2)*(1/config.noise_covar)) -(1/2)*np.sum((x[3:]**2)*(1/cls[3:]))


def compute_gradient_log(s, s_pix, grad_constant_part, cls):
    #### ATTENTION, ici j'ai pris la partie réelle !
    intermediate = (1/config.noise_covar)*s_pix
    first_part = np.real(hp.sphtfunc.map2alm(intermediate, lmax=L_MAX_SCALARS))
    second_part = np.array([0, 0, 0] + list((1/2)*(1/cls[3:])*s[3:]))
    grad_variable = first_part + second_part
    return - grad_variable + (1/2)*grad_constant_part


def proposal(gradient_log, s):
    return s + step_size*gradient_log + np.sqrt(config.var_mala)*np.sqrt(2*step_size)*np.random.normal(size=dimension_sph)


def compute_MH_ratio(grad_log, grad_log_prop, s, s_pix, prop, prop_pix, observations, cls):
    return compute_log_density(prop, prop_pix, observations, cls) + compute_log_proposal(prop, s, grad_log_prop) - compute_log_density(s, s_pix, observations, cls) - compute_log_proposal(s, prop, grad_log)

def mala(cls, observations, grad_constant_part):
    #Input: list of Cls, where Cl is present 2l+1 times, once for each different m.
    #Output: a sample from the normal distribution, via mala.
    acceptance = 0
    #s = np.sqrt(cls)*np.random.normal(size= dimension_sph)
    print(config.var_mala)
    s = np.zeros(dimension_sph)
    s_pixel = hp.sphtfunc.alm2map(s.astype(complex), nside=NSIDE)
    grad_log_s = compute_gradient_log(s, s_pixel, grad_constant_part, cls)
    history = []
    for i in range(N_iter):
        history.append(s)
        s_prop = proposal(grad_log_s, s)
        s_prop_pix = hp.sphtfunc.alm2map(s.astype(complex), nside=NSIDE)
        grad_log_prop = compute_gradient_log(s, s_prop_pix, grad_constant_part, cls)
        r = compute_MH_ratio(grad_log_s, grad_log_prop, s, s_pixel, s_prop, s_prop_pix, observations, cls)
        if np.log(np.random.uniform()) < r:
            s = s_prop
            s_pixel = s_prop_pix
            grad_log_s = grad_log_prop
            acceptance += 1

        print(r)
        print("\n")

    print(acceptance/N_iter)
    return s, history









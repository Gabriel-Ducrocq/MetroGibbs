import healpy as hp
import numpy as np
from classy import Class
import config
import json
import matplotlib.pyplot as plt
import pylab

import samplingInvGamm

cosmo = Class()

LENSING = 'yes'
OUTPUT_CLASS = 'tCl pCl lCl'

COSMO_PARAMS_NAMES = ["n_s", "omega_b", "omega_cdm", "100*theta_s", "ln10^{10}A_s", "tau_reio"]
COSMO_PARAMS_MEAN = np.array([0.9665, 0.02242, 0.11933, 1.04101, 3.047, 0.0561])
COSMO_PARAMS_SIGMA = np.array([0.0038, 0.00014, 0.00091, 0.00029, 0.014, 0.0071])
LiteBIRD_sensitivities = np.array([36.1, 19.6, 20.2, 11.3, 10.3, 8.4, 7.0, 5.8, 4.7, 7.0, 5.8, 8.0, 9.1, 11.4, 19.6])


def add_null_imag_alm(x):
    zeros = np.zeros(config.L_MAX_SCALARS)
    indexes = np.array([(i+1)*i/2 for i in range(config.L_MAX_SCALARS)])
    x = np.concatenate((np.zeros(1), np.insert(x, indexes, zeros)))
    return x

def remove_null_imag_alm(x):
    x = np.array(x)
    indexes = [(i+1)*(i+2)/2 for i in range(config.L_MAX_SCALARS)]
    x = np.delete(x, indexes)
    return x[1:]

def generate_theta():
    return np.random.normal(COSMO_PARAMS_MEAN, COSMO_PARAMS_SIGMA)

def generate_cls(theta):
    params = {'output': OUTPUT_CLASS,
              'l_max_scalars': config.L_MAX_SCALARS,
              'lensing': LENSING}
    d = {name:val for name, val in zip(COSMO_PARAMS_NAMES, theta)}
    params.update(d)
    cosmo.set(params)
    cosmo.compute()
    cls = cosmo.lensed_cl(config.L_MAX_SCALARS)
    #10^12 parce que les cls sont exprimés en kelvin carré, du coup ça donne une stdd en 10^6
    cls["tt"] *= 1e12
    cosmo.struct_cleanup()
    cosmo.empty()
    return cls["tt"]


def generate_sky_map():
    theta = generate_theta()
    cls_ = generate_cls(theta)
    s = hp.sphtfunc.synalm(cls_, lmax=config.L_MAX_SCALARS)
    s_pix = hp.sphtfunc.alm2map(s , nside=config.NSIDE)
    return s_pix + np.sqrt(config.noise_covar)*np.random.normal(size=config.Npix*config.N_Stoke), cls_, s


def plot_results(path, index, check_alm = False):
    results = np.load(path)[()]
    path_cls = np.array(results["path_cls"])
    path_alms = np.array(results["path_alms"])
    obs = results["obs_map"]
    #obs_alms = results["obs_alms"]
    obs_alms = hp.map2alm(obs, lmax=results["config"]["L_MAX_SCALARS"])
    realized_cls = hp.anafast(obs, lmax=results["config"]["L_MAX_SCALARS"])
    true_cls = results["config"]["true_spectrum"]
    N_gibbs = results["config"]["N_gibbs"]
    if not check_alm:
        plt.plot(path_cls[:, index])
        plt.show()
        plt.close()
        plt.hist(path_cls[1000:, index], bins=100)
        plt.axvline(x=realized_cls[index], color='k', linestyle='dashed', linewidth=1)
        plt.axvline(x=true_cls[index], color='k', linewidth=1)
        plt.show()
    else:
        plt.plot(path_alms[:, index].imag)
        plt.show()
        plt.close()
        plt.hist(path_alms[1000:, index].imag, bins=50)
        plt.axvline(x=obs_alms[index].imag, color='k', linestyle='dashed', linewidth=1)
        #plt.axvline(x=true_cls[index], color='k', linewidth=1)
        plt.show()


    print(true_cls)
    print(obs_alms[:10])
    print(realized_cls[index])


import healpy as hp
import numpy as np
from classy import Class
import config
import json
import matplotlib.pyplot as plt
import pylab
import healpy as hp

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
    print(realized_cls)


def compare(conj_path, crank_path, mala_path, index):
    results_conj = np.load(conj_path)[()]
    results_crank = np.load(crank_path)[()]
    results_mala = np.load(mala_path)[()]
    path_cls_conj = np.array(results_conj["path_cls"])
    path_cls_crank = np.array(results_crank["path_cls"])
    path_cls_mala = np.array(results_mala["path_cls"])
    obs = results_conj["obs_map"]

    obs_alms = hp.map2alm(obs, lmax=results_conj["config"]["L_MAX_SCALARS"])
    realized_cls = hp.anafast(obs, lmax=results_conj["config"]["L_MAX_SCALARS"])
    true_cls = results_conj["config"]["true_spectrum"]
    N_gibbs = results_conj["config"]["N_gibbs"]

    plt.plot(path_cls_conj[:, index])
    plt.plot(path_cls_crank[:, index])
    plt.plot(path_cls_mala[:, index])
    plt.show()
    plt.close()
    plt.hist(path_cls_conj[1000:, index], bins=25, density = True, alpha=0.2, label="CG")
    plt.hist(path_cls_crank[1000:, index], bins=25, density=True, alpha=0.2, label="CN")
    plt.hist(path_cls_mala[1000:, index], bins=25, density=True, alpha=0.2, label="MALA")
    plt.axvline(x=realized_cls[index], color='k', linestyle='dashed', linewidth=1)
    plt.axvline(x=true_cls[index], color='k', linewidth=1)
    plt.legend(loc="upper right")
    plt.show()

    print("N_gibbs")
    print(N_gibbs)
    print("N_mala")
    print(results_mala["config"]["N_crank"])


def get_Ylm_transp():
    all_vec = []

    for i in range(12*config.NSIDE**2):
        u = np.zeros(12*config.NSIDE**2)
        u[i] = 1
        vect = hp.map2alm(u, lmax=config.L_MAX_SCALARS)
        all_vec.append(vect)


    arr = np.array(all_vec).T
    full_mat = np.concatenate((np.conj(arr[config.L_MAX_SCALARS+1:, :]), arr), axis=0)
    return full_mat


def get_Ylm():
    y_l_m = []
    y_l_minus_m = []
    #On calcule également le monopole et le dipole !
    for i in range(config.dimension_sph):
        print(i)
        alm = np.zeros(config.dimension_sph, dtype=complex)
        if i < config.L_MAX_SCALARS+1:
            alm[i] = 1 + 0*1j
            ylm = hp.alm2map(alm, nside=config.NSIDE)
            y_l_m.append(ylm)
            y_l_minus_m.append(ylm)

        else:
            alm[i] = 1 + 0 * 1j
            re = hp.alm2map(alm, nside=config.NSIDE)
            alm[i] = 0 + 1j
            img = hp.alm2map(alm, nside=config.NSIDE)

            a_plus = (img.imag + re.real)/2
            a_minus = (re.real - img.imag)/2
            b_plus = (re.imag - img.real)/2
            b_minus = (re.imag + img.real)/2

            y_l_m.append(a_plus+1j*b_plus)
            y_l_minus_m.append(a_minus + 1j * b_minus)


    A1 = np.array(y_l_m).T
    A2 = np.array(y_l_minus_m[config.L_MAX_SCALARS+1:]).T
    A = np.concatenate((A2, A1), axis = 1)
    return A


def sph_transform_by_hand(alm, A):
    alm_conjugate = np.conjugate(alm[config.L_MAX_SCALARS+1:])
    all_alm = np.concatenate((alm_conjugate, alm))
    map = np.dot(A, all_alm)
    return map.real


def compute_grouping_matrix():
    mat = np.zeros(((config.L_MAX_SCALARS+1)**2 -4, (config.L_MAX_SCALARS+1)**2 - 4), dtype=complex)
    print(mat.shape)
    for i in range(mat.shape[0]):
        print(i)
        if i in range(config.N - (config.L_MAX_SCALARS + 2)):
            mat[i, i] = -1j
            mat[i, config.N - (config.L_MAX_SCALARS + 2) + config.L_MAX_SCALARS - 1 + i] = 1

        elif i < config.N - (config.L_MAX_SCALARS + 2) + config.L_MAX_SCALARS - 1:
            mat[i, i] = 1

        else:
            mat[i, i - (config.N - (config.L_MAX_SCALARS + 2) + config.L_MAX_SCALARS - 1)] = 1j
            mat[i, i] = 1

    mat = np.insert(mat, [0, config.N - (config.L_MAX_SCALARS + 2), config.N - (config.L_MAX_SCALARS + 2) ,
                    config.N - (config.L_MAX_SCALARS + 2) + config.L_MAX_SCALARS - 2 + 1], np.zeros(mat.shape[1]), axis=0)

    return mat


def compute_grouping_inverse():
    mat = np.zeros(((config.L_MAX_SCALARS+1)**2 - 4, (config.L_MAX_SCALARS+1)**2), dtype=complex)
    for i in range((config.L_MAX_SCALARS+1)**2 - 4):
        if i < config.N - (config.L_MAX_SCALARS + 2):
            mat[i, i+1] = (1/2)*1j
            mat[i, config.N+1+i] = -(1/2)*1j

        elif i < config.N - 3:
            mat[i, i+3] = 1

        else:
            mat[i, i + 4] = 1/2
            mat[i, i - (config.N - (config.L_MAX_SCALARS + 1))] = 1/2

    return mat


def compute_variance_matrix(cls_):
    A = get_Ylm()
    extended_cls = [cl for l in range(config.L_MAX_SCALARS + 1) for cl in cls_[l:]]
    extended_cls_real = (extended_cls[:(config.L_MAX_SCALARS + 1)] + extended_cls[(config.L_MAX_SCALARS + 2):])[2:]
    extended_cls_imag = extended_cls[(config.L_MAX_SCALARS + 2):]
    ###Attention, l'ordre est ici inversé !
    extended_cls = np.array(extended_cls_imag + extended_cls_real)
    mat = compute_grouping_matrix()

    overall_A = np.dot(A, mat)
    precision = np.dot(overall_A.T, np.dot(np.diag(1/config.noise_covar), overall_A)) + (1/extended_cls)
    variance = np.linalg.inv(precision)
    return variance, precision, mat, A
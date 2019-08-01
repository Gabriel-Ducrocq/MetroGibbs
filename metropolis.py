import numpy as np
import config
import healpy as hp
from scipy.stats import truncnorm


# Proposal is symmetric, so it disappear from the MH ratio

def propose_cls(old_cls):
    new_cls = np.concatenate((np.zeros(2),truncnorm.rvs(-old_cls[2:], np.inf, loc=old_cls[2:], scale=config.stdd_cls[2:])))
    return new_cls


def compute_log_chi_2(s_pix):
    return -(1/2)*np.sum(((config.observations - s_pix)**2)*(1/config.noise_covar))


def metropolis(s_init, cls_init):
    accept = 0
    s = s_init
    s_pix = hp.sphtfunc.alm2map(s, nside= config.NSIDE)
    cls = cls_init
    for i in range(config.N_metropolis):
        cls_new = propose_cls(cls)
        ratio = np.concatenate((np.zeros(2), np.sqrt(cls_new[2:]/cls[2:])))
        all_ratios = np.array([cl for l in range(config.L_MAX_SCALARS + 1) for cl in ratio[l:]])
        s_new = all_ratios*s
        s_pix_new = hp.sphtfunc.alm2map(s_new, nside=config.NSIDE)
        r = compute_log_chi_2(s_pix_new) - compute_log_chi_2(s_pix)
        if np.log(np.random.uniform()) < r:
            s = s_new
            cls = cls_new
            accept += 1

    print(accept/config.N_metropolis)
    return s, cls

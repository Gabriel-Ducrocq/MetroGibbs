import numpy as np
import config
import healpy as hp
from scipy.stats import truncnorm
import utils


# Proposal is symmetric, so it disappear from the MH ratio

def propose_cls(old_cls):
    new_cls = np.concatenate((np.zeros(2),truncnorm.rvs(0, np.inf, loc=old_cls[2:], scale=config.stdd_cls[2:])))
    return new_cls


def compute_log_chi_2(s_pix):
    return -(1/2)*np.sum(((config.observations - s_pix)**2)*(1/config.noise_covar))


def metropolis(s_init, cls_init):
    accept = 0
    s = s_init[:config.N] + 1j* utils.add_null_imag_alm(s_init[config.N:])
    s_pix = hp.sphtfunc.alm2map(s.astype(complex), nside= config.NSIDE)
    cls = cls_init
    for i in range(config.N_metropolis):
        cls_new = propose_cls(cls)
        ratio = np.concatenate((np.zeros(2), list(np.sqrt(cls_new[2:]/cls[2:]))))
        all_ratios = [r for l, r in enumerate(ratio) for _ in range(l+1)]
        s_new = all_ratios*s
        s_pix_new = hp.sphtfunc.alm2map(s_new.astype(complex), nside=config.NSIDE)
        r = compute_log_chi_2(s_pix_new) - compute_log_chi_2(s_pix)
        if np.log(np.random.uniform()) < r:
            s = s_new
            cls = cls_new
            accept += 1

    print(accept/config.N_metropolis)
    s_re = s.real
    s_img = utils.remove_null_imag_alm(s.imag)
    s = np.concatenate((s_re, s_img))
    return s, cls

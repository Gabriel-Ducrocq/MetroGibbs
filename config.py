import numpy as np
import healpy as hp

COSMO_PARAMS_NAMES = ["n_s", "omega_b", "omega_cdm", "100*theta_s", "ln10^{10}A_s", "tau_reio"]
COSMO_PARAMS_MEAN = np.array([0.9665, 0.02242, 0.11933, 1.04101, 3.047, 0.0561])
COSMO_PARAMS_SIGMA = np.array([0.0038, 0.00014, 0.00091, 0.00029, 0.014, 0.0071])
LiteBIRD_sensitivities = np.array([36.1, 19.6, 20.2, 11.3, 10.3, 8.4, 7.0, 5.8, 4.7, 7.0, 5.8, 8.0, 9.1, 11.4, 19.6])

LENSING = 'yes'
OUTPUT_CLASS = 'tCl pCl lCl'
observations = None

N_Stoke = 1
NSIDE=4
Npix = 12*NSIDE**2
L_MAX_SCALARS=int(2*NSIDE)
dimension_sph = int((L_MAX_SCALARS*(L_MAX_SCALARS + 1)/2)+L_MAX_SCALARS+1)


def noise_covariance_in_freq(nside):
    ##Prendre les plus basses fréquences pour le bruit (là où il est le plus petit)
    cov = LiteBIRD_sensitivities ** 2 / hp.nside2resol(nside, arcmin=True) ** 2
    return cov


noise_covar_one_pix = noise_covariance_in_freq(NSIDE)
noise_covar = np.array([noise_covar_one_pix[7] for _ in range(Npix*N_Stoke)])
N_metropolis = 1
N_gibbs = 10000

N_real_img = (L_MAX_SCALARS+1)**2
N_CN = 100000
N_mala = 100000
#N_mala = 10000
#step_size_mala = 0.001
step_size_mala = 0.00000001
#step_size_mala = 0.0000008
#step_size_mala = 0.001
N = int((L_MAX_SCALARS*(L_MAX_SCALARS + 1)/2)+L_MAX_SCALARS+1)
var_mala = np.ones(dimension_sph)   #(1/dimension_sph)**(1/3)

stdd_cls = np.ones(L_MAX_SCALARS+1)

coord = []
coord_ = []
for i in range(L_MAX_SCALARS + 1):
    dim = int(i + 1)
    e = np.ones(dim)
    e_= np.ones(dim)
    e[0] = 0
    e_[0] = np.sqrt(2)
    coord += list(e)
    coord_ += list(e_)

mask_imaginary = np.array(coord)
mask_real = np.array(coord_)
#beta_CN = 0.0000005
beta_CN = 0.00009

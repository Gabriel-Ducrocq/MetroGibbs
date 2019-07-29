import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
import healpy as hp
import config
import utils


class CG():
    def __init__(self):
        self.cls = np.zeros(config.N)
        self.denom = np.ones(config.N)

    def set_cls(self, cls):
        self.cls = np.array([cl for l in range(config.L_MAX_SCALARS+1) for cl in cls[l:]])
        print(self.cls.shape)
        self.denom = np.concatenate([np.zeros(2), 1/self.cls[2:(config.L_MAX_SCALARS+1)], np.zeros(1), 1/self.cls[(config.L_MAX_SCALARS+2):]])

    def linOp(self, x):
        pix_map = hp.sphtfunc.alm2map(x.astype(complex), nside=config.NSIDE)
        first_term = hp.sphtfunc.map2alm((1/config.noise_covar)*pix_map, lmax=config.L_MAX_SCALARS)
        second_term = self.denom*x
        sol = first_term + second_term
        return sol

    def compute_mean(self, d, cls):
        self.set_cls(cls)
        A = LinearOperator((config.N_real_img, config.N_real_img), matvec=self.linOp)
        u = hp.sphtfunc.map2alm((1 / config.noise_covar) * d, lmax=config.L_MAX_SCALARS)
        re = u.real
        img = u.imag
        y = np.concatenate((re, utils.remove_null_imag_alm(img)))
        solution, err = cg(A, y)
        return solution, err

    def get_var(self, i):
        A = LinearOperator((2*config.N, 2*config.N), matvec=self.linOp)
        v = np.zeros(2*config.N)
        v[i] = 1
        solution, err = cg(A, v)
        print("Variance")
        return solution[i], err

    def run(self, d, cls):
        self.set_cls(cls)
        A = LinearOperator((config.N, config.N), matvec=self.linOp, dtype = complex)
        #Corriger à cet endroit !!!!
        omega0 = np.sqrt(self.denom)*np.random.normal(size=config.N)
        omega1 = hp.sphtfunc.map2alm((1/np.sqrt(config.noise_covar))*np.random.normal(size=config.Npix).astype(complex),lmax=config.L_MAX_SCALARS)
        u = hp.sphtfunc.map2alm(((1/config.noise_covar) * d), lmax=config.L_MAX_SCALARS)
        b = u + omega0 + omega1
        solution, err = cg(A, b)
        return solution, err


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
        self.denom = np.concatenate([np.zeros(2), 1/self.cls[2:(config.L_MAX_SCALARS+1)], np.zeros(1), 1/self.cls[(config.L_MAX_SCALARS+2):]])

    def linOp(self, x):
        pix_map = hp.sphtfunc.alm2map(x.astype(complex), nside=config.NSIDE)
        first_term = hp.sphtfunc.map2alm((1/config.noise_covar)*pix_map, lmax=config.L_MAX_SCALARS)
        second_term = self.denom*x
        sol = first_term + second_term
        return sol

    def compute_mean(self, d, cls):
        self.set_cls(cls)
        A = LinearOperator((config.N, config.N), matvec=self.linOp, dtype = complex)
        y = hp.sphtfunc.map2alm((1 / config.noise_covar) * d, lmax=config.L_MAX_SCALARS)
        solution, err = cg(A, y)
        return solution, err

    def get_var(self, i):
        A = LinearOperator((config.N, config.N), matvec=self.linOp)
        v = np.zeros(config.N).astype(complex)
        v[i] = 1
        solution, err = cg(A, v)
        return solution[i], err

    def run(self, d, cls):
        self.set_cls(cls)
        A = LinearOperator((config.N, config.N), matvec=self.linOp, dtype=complex)
        #Corriger à cet endroit !!!!
        omega0 = np.sqrt(self.denom)*np.random.normal(size=config.N)
        omega1 = hp.sphtfunc.map2alm((1/np.sqrt(config.noise_covar))*np.random.normal(size=config.Npix).astype(complex),lmax=config.L_MAX_SCALARS)
        u = hp.sphtfunc.map2alm(((1/config.noise_covar) * d), lmax=config.L_MAX_SCALARS)
        b = u + omega0 + omega1
        solution, err = cg(A, b)
        return solution, err


class CG2:
    def __init__(self):
        self.cls = np.zeros(config.N)
        self.denom = np.ones(config.N)

    def set_cls(self, cls):
        extended_cls = [cl for l in range(config.L_MAX_SCALARS+1) for cl in cls[l:]]
        extended_cls_real = (extended_cls[:(config.L_MAX_SCALARS + 1)] + extended_cls[(config.L_MAX_SCALARS + 2):])[2:]
        extended_cls_imag = extended_cls[(config.L_MAX_SCALARS + 2):]
        self.cls = np.array(extended_cls_real + extended_cls_imag)

    def linOp(self, x):
        x_real = np.concatenate((np.zeros(2), x[:(config.dimension_sph-3)]))
        x_real = np.concatenate((x_real[:(config.L_MAX_SCALARS + 1)], np.zeros(1), x_real[(config.L_MAX_SCALARS + 1):]))
        x_imag = np.concatenate((np.zeros(config.L_MAX_SCALARS+2), x[(config.dimension_sph-3):]))
        map = x_real + 1j*x_imag
        pix_map = hp.sphtfunc.alm2map(map, nside=config.NSIDE)
        first_term = flatten_map(hp.sphtfunc.map2alm((1/config.noise_covar)*pix_map, lmax=config.L_MAX_SCALARS))
        second_term = (1/self.cls)*x
        sol = first_term + second_term
        return sol

    def compute_mean(self, d, cls):
        self.set_cls(cls)
        A = LinearOperator(((config.L_MAX_SCALARS+1)**2 - 4,
                            (config.L_MAX_SCALARS+1)**2 - 4),
                           matvec=self.linOp, dtype=complex)

        y = flatten_map(hp.sphtfunc.map2alm((1 / config.noise_covar) * d, lmax=config.L_MAX_SCALARS))
        solution, err = cg(A, y)
        return solution, err

    def get_var(self, i):
        A = LinearOperator(((config.L_MAX_SCALARS + 1) ** 2 - 4,
                            (config.L_MAX_SCALARS + 1) ** 2 - 4),
                           matvec=self.linOp, dtype=complex)
        v = np.zeros((config.L_MAX_SCALARS + 1) ** 2 - 4).astype(complex)
        v[i] = 1
        solution, err = cg(A, v)
        return solution[i], err

    def run(self, d, cls):
        self.set_cls(cls)
        A = LinearOperator((config.N, config.N), matvec=self.linOp, dtype=complex)
        #Corriger à cet endroit !!!!
        omega0 = np.sqrt(self.denom)*np.random.normal(size=config.N)
        omega1 = hp.sphtfunc.map2alm((1/np.sqrt(config.noise_covar))*np.random.normal(size=config.Npix).astype(complex),lmax=config.L_MAX_SCALARS)
        u = hp.sphtfunc.map2alm(((1/config.noise_covar) * d), lmax=config.L_MAX_SCALARS)
        b = u + omega0 + omega1
        solution, err = cg(A, b)
        return solution, err


def flatten_map(s):
    s_real = s.real[[i for i in range(len(s)) if i != 0 and i != 1 and i != (config.L_MAX_SCALARS+1)]]
    s_imag = s.imag[[i for i in range((config.L_MAX_SCALARS+2),len(s))]]
    s_flatten = np.concatenate((s_real, s_imag))
    return s_flatten
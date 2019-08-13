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
        solution, err = cg(A, y, tol=1e-8)
        return solution, err

    def get_var(self, i):
        A = LinearOperator((config.N, config.N), matvec=self.linOp)
        v = np.zeros(config.N).astype(complex)
        v[i] = 1
        solution, err = cg(A, v,tol=1e-8)
        return solution[i], err

    def run(self, d, cls):
        self.set_cls(cls)
        A = LinearOperator((config.N, config.N), matvec=self.linOp, dtype=complex)
        #Corriger à cet endroit !!!!
        omega0 = np.sqrt(self.denom)*np.random.normal(size=config.N)
        omega1 = hp.sphtfunc.map2alm((1/np.sqrt(config.noise_covar))*np.random.normal(size=config.Npix).astype(complex),lmax=config.L_MAX_SCALARS)
        u = hp.sphtfunc.map2alm(((1/config.noise_covar) * d), lmax=config.L_MAX_SCALARS)
        b = u + omega0 + omega1
        solution, err = cg(A, b, tol=1e-8)
        return solution, err






def flatten_map2(s):
    s_real = s.real[[i for i in range(len(s)) if i != 0 and i != 1 and i != (config.L_MAX_SCALARS+1)]]
    s_imag = s.imag[[i for i in range((config.L_MAX_SCALARS+2),len(s))]]
    s_flatten = np.concatenate((s_real, s_imag))
    return s_flatten

def unflat_map_to_pix2(s):
    real_part = np.concatenate((np.zeros(2), s[:(config.L_MAX_SCALARS-1)] , np.zeros(1),
                                 s[(config.L_MAX_SCALARS-1):(config.dimension_sph-3)]))

    imag_part = np.concatenate((np.zeros(config.L_MAX_SCALARS+2), s[(config.dimension_sph-3):]))
    return real_part + 1j*imag_part


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
        map = unflat_map_to_pix2(x)
        pix_map = hp.sphtfunc.alm2map(map, nside=config.NSIDE)
        first_term = flatten_map2(hp.sphtfunc.map2alm((1/config.noise_covar)*pix_map, lmax=config.L_MAX_SCALARS))
        second_term = (1/self.cls)*x
        sol = first_term + second_term
        return sol

    def compute_mean(self, d, cls):
        self.set_cls(cls)
        A = LinearOperator(((config.L_MAX_SCALARS+1)**2 - 4,
                            (config.L_MAX_SCALARS+1)**2 - 4),
                           matvec=self.linOp, dtype=complex)

        y = flatten_map(hp.sphtfunc.map2alm((1 / config.noise_covar) * d, lmax=config.L_MAX_SCALARS))
        solution, err = cg(A, y, tol=1e-8)
        return solution, err

    def get_var(self, i):
        A = LinearOperator(((config.L_MAX_SCALARS + 1) ** 2 - 4,
                            (config.L_MAX_SCALARS + 1) ** 2 - 4),
                           matvec=self.linOp, dtype=complex)
        v = np.zeros((config.L_MAX_SCALARS + 1) ** 2 - 4)
        v[i] = 1
        solution, err = cg(A, v, tol=1e-8)
        return solution[i], err

    def run(self, d, cls):
        self.set_cls(cls)
        A = LinearOperator(((config.L_MAX_SCALARS + 1) ** 2 - 4, (config.L_MAX_SCALARS + 1) ** 2 - 4),
                           matvec=self.linOp, dtype=complex)
        omega0 = np.sqrt(1/self.cls)*np.random.normal(size=len(self.cls))
        omega1 = hp.sphtfunc.map2alm((1/np.sqrt(config.noise_covar))*np.random.normal(size=config.Npix).astype(complex)
                                     ,lmax=config.L_MAX_SCALARS)
        u = hp.sphtfunc.map2alm(((1/config.noise_covar) * d), lmax=config.L_MAX_SCALARS)
        b = omega0 + flatten_map(u+omega1)
        solution, err = cg(A, b, tol=1e-8)
        return solution, err


def flatten_map(s):
    s_real = s.real[[i for i in range(len(s)) if i != 0 and i != 1 and i != (config.L_MAX_SCALARS+1)]]
    s_imag = s.imag[[i for i in range((config.L_MAX_SCALARS+2),len(s))]]
    s_flatten = np.concatenate((s_real, s_imag))
    return s_flatten


#### Third version

def flatten_map3(s):
    s_real = s.real[[i for i in range(len(s)) if i != 0 and i != 1 and i != (config.L_MAX_SCALARS+1)]]
    s_imag = s.imag[[i for i in range((config.L_MAX_SCALARS+2),len(s))]]
    s_flatten = np.concatenate((s_imag, s_real))
    return s_flatten



class CG3:
    def __init__(self):
        self.cls = np.zeros(config.N)
        self.denom = np.ones(config.N)
        self.variance = np.zeros((config.N, config.N))
        self.precision = np.zeros((config.N, config.N))

    def set_cls(self, cls_):
        extended_cls = [cl for l in range(config.L_MAX_SCALARS+1) for cl in cls_[l:]]
        extended_cls_real = (extended_cls[:(config.L_MAX_SCALARS + 1)] + extended_cls[(config.L_MAX_SCALARS + 2):])[2:]
        extended_cls_imag = extended_cls[(config.L_MAX_SCALARS + 2):]
        self.cls = np.array(extended_cls_imag + extended_cls_real)
        self.set_matrices(cls_)

    def set_matrices(self, cls_):
        self.variance, self.precision, _, _ = utils.compute_variance_matrix(cls_)

    def linOp(self, x):
        sol = np.dot(self.precision, x)
        return sol

    def compute_mean(self, d, cls):
        self.set_cls(cls)
        A = LinearOperator(((config.L_MAX_SCALARS+1)**2 - 4,
                            (config.L_MAX_SCALARS+1)**2 - 4),
                           matvec=self.linOp, dtype=complex)

        y = flatten_map3(hp.sphtfunc.map2alm((1 / config.noise_covar) * d, lmax=config.L_MAX_SCALARS))
        solution, err = cg(A, y, tol=1e-8)
        return solution, err

    def get_var(self, i):
        A = LinearOperator(((config.L_MAX_SCALARS + 1) ** 2 - 4,
                            (config.L_MAX_SCALARS + 1) ** 2 - 4),
                           matvec=self.linOp, dtype=complex)
        v = np.zeros((config.L_MAX_SCALARS + 1) ** 2 - 4)
        v[i] = 1
        solution, err = cg(A, v, tol=1e-8)
        return solution[i], err

    def run(self, d, cls):
        self.set_cls(cls)
        A = LinearOperator(((config.L_MAX_SCALARS + 1) ** 2 - 4, (config.L_MAX_SCALARS + 1) ** 2 - 4),
                           matvec=self.linOp, dtype=complex)
        omega0 = np.sqrt(1/self.cls)*np.random.normal(size=len(self.cls))
        omega1 = hp.sphtfunc.map2alm((1/np.sqrt(config.noise_covar))*np.random.normal(size=config.Npix).astype(complex)
                                     ,lmax=config.L_MAX_SCALARS)
        u = hp.sphtfunc.map2alm(((1/config.noise_covar) * d), lmax=config.L_MAX_SCALARS)
        b = omega0 + flatten_map3(u+omega1)
        solution, err = cg(A, b, tol=1e-8)
        return solution, err

    def test(self, cls):
        self.set_cls(cls)
        A = LinearOperator(((config.L_MAX_SCALARS + 1) ** 2 - 4, (config.L_MAX_SCALARS + 1) ** 2 - 4),
                           matvec=self.linOp, dtype=complex)
        u = np.ones((config.L_MAX_SCALARS + 1) ** 2 - 4)
        sol, err = cg(A,u)
        return sol


######## Fourth version


class CG4:
    def __init__(self, grouping_mat, transfo,transfo_to_alm):
        self.cls = np.zeros(config.N)
        self.denom = np.ones(config.N)
        self.mat = grouping_mat
        self.alm_to_pix = transfo
        self.pix_to_alm = transfo_to_alm

    def set_cls(self, cls):
        extended_cls = [cl for l in range(config.L_MAX_SCALARS+1) for cl in cls[l:]]
        extended_cls_real = (extended_cls[:(config.L_MAX_SCALARS + 1)] + extended_cls[(config.L_MAX_SCALARS + 2):])[2:]
        extended_cls_imag = extended_cls[(config.L_MAX_SCALARS + 2):]
        self.cls = np.array(extended_cls_imag + extended_cls_real)

    def linOp(self, x):
        #map = np.dot(self.mat, x)
        map = unflat_map_to_pix4(x)
        map = np.concatenate((np.conj(map[config.L_MAX_SCALARS+1:]), map))
        #pix_map = hp.sphtfunc.alm2map(map, nside=config.NSIDE)
        pix_map = np.dot(self.alm_to_pix, map)
        #first_term = flatten_map4(hp.sphtfunc.map2alm((1/config.noise_covar)*pix_map, lmax=config.L_MAX_SCALARS))
        #first_term = np.dot(self.mat.T, np.dot(self.A_T.T, (1/config.noise_covar)*pix_map))
        interm = np.dot(self.pix_to_alm, (1 / config.noise_covar) * pix_map)[config.N - (config.L_MAX_SCALARS +1):]
        first_term = flatten_map4(interm)
        #first_term = np.dot(self.mat.T, np.dot(self.A_T.T, (1 / config.noise_covar) * pix_map))
        second_term = (1/self.cls)*x
        sol = first_term + second_term
        return sol

    def compute_mean(self, d, cls):
        self.set_cls(cls)
        A = LinearOperator(((config.L_MAX_SCALARS+1)**2 - 4,
                            (config.L_MAX_SCALARS+1)**2 - 4),
                           matvec=self.linOp, dtype=complex)

        y = flatten_map4(hp.sphtfunc.map2alm((1 / config.noise_covar) * d, lmax=config.L_MAX_SCALARS))
        solution, err = cg(A, y, tol=1e-8)
        return solution, err

    def get_var(self, i):
        A = LinearOperator(((config.L_MAX_SCALARS + 1) ** 2 - 4,
                            (config.L_MAX_SCALARS + 1) ** 2 - 4),
                           matvec=self.linOp, dtype=complex)
        v = np.zeros((config.L_MAX_SCALARS + 1) ** 2 - 4)
        v[i] = 1
        solution, err = cg(A, v, tol=1e-8)
        return solution[i], err

    def run(self, d, cls):
        self.set_cls(cls)
        A = LinearOperator(((config.L_MAX_SCALARS + 1) ** 2 - 4, (config.L_MAX_SCALARS + 1) ** 2 - 4),
                           matvec=self.linOp, dtype=complex)
        omega0 = np.sqrt(1/self.cls)*np.random.normal(size=len(self.cls))
        omega1 = hp.sphtfunc.map2alm((1/np.sqrt(config.noise_covar))*np.random.normal(size=config.Npix).astype(complex)
                                     ,lmax=config.L_MAX_SCALARS)
        u = hp.sphtfunc.map2alm(((1/config.noise_covar) * d), lmax=config.L_MAX_SCALARS)
        b = omega0 + flatten_map4(u+omega1)
        solution, err = cg(A, b, tol=1e-8)
        return solution, err

    def test(self, cls):
        self.set_cls(cls)
        A = LinearOperator(((config.L_MAX_SCALARS + 1) ** 2 - 4, (config.L_MAX_SCALARS + 1) ** 2 - 4),
                           matvec=self.linOp, dtype=complex)
        u = np.ones((config.L_MAX_SCALARS + 1) ** 2 - 4)
        sol, err = cg(A,u)
        return sol


def flatten_map4(s):
    s_real = s.real[[i for i in range(len(s)) if i != 0 and i != 1 and i != (config.L_MAX_SCALARS+1)]]
    s_imag = s.imag[[i for i in range((config.L_MAX_SCALARS+2),len(s))]]
    s_flatten = np.concatenate((s_imag, s_real))
    return s_flatten


def unflat_map_to_pix4(s):
    #real_part = np.concatenate((np.zeros(2), s[:(config.L_MAX_SCALARS-1)] , np.zeros(1),
    #                             s[(config.L_MAX_SCALARS-1):(config.dimension_sph-3)]))

    #imag_part = np.concatenate((np.zeros(config.L_MAX_SCALARS+2), s[(config.dimension_sph-3):]))
    imag_part = np.concatenate((np.zeros(config.L_MAX_SCALARS+2), s[:config.N-config.L_MAX_SCALARS-1-1]))
    real_part = s[config.N-config.L_MAX_SCALARS-1-1:]
    real_part = np.insert(real_part, [0, 0, config.L_MAX_SCALARS - 1], 0)
    return real_part + 1j*imag_part
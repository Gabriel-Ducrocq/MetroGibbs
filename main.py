import mala
import utils
import matplotlib.pyplot as plt
import time
from CrankNicolson import crankNicolson
import metropolis
import config
import numpy as np
import runner
import healpy as hp
import conjugateGradient
from gradientDescent import gradient_ascent

CLS = np.ones(config.N)
Int = 10

def main():
    d, cls_, s_obs = utils.generate_sky_map()
    print("TO PRINT")
    print(s_obs[:10])
    print(hp.map2alm(d, lmax=config.L_MAX_SCALARS)[:10])
    config.observations = d

    conjgrad = conjugateGradient.CG()
    true_mean, err = conjgrad.compute_mean(d, cls_)
    if err != 0:
        print("Conjugate gradient did not converge")

    #true_var, err = conjgrad.get_var(Int)
    """
    grad_constant_part = mala.compute_gradient_log_constant_part(d)
    history, s = mala.mala(cls_, d, grad_constant_part)
    #history, _ = crankNicolson(cls_, d)
    path = np.array(history)[:, Int].real
    print(true_var)
    plt.plot(path)
    plt.axhline(y=true_mean[Int].real, color='k', linewidth=1)
    plt.axhline(y=true_mean[Int].real + 2*np.sqrt(true_var), color='red', linewidth=1)
    plt.axhline(y=true_mean[Int].real - 2*np.sqrt(true_var), color='red', linewidth=1)
    plt.show()
    plt.close()
    plt.hist(np.array(history)[:, Int].real, bins = 50)
    plt.axvline(x=true_mean[Int].real, color='k', linewidth=1)
    plt.show()
    """
    h_cls, h_s = runner.gibbs_conj(d)
    results_conj = {"path_cls":h_cls, "path_alms":h_s, "obs_map":d, "obs_alms":s_obs,"config":{"NSIDE": config.NSIDE,
                                                            "L_MAX_SCALARS":config.L_MAX_SCALARS,
                                                           "N_gibbs":config.N_gibbs, "true_spectrum":cls_,
                                                                                          "N":config.N_gibbs}}

    h_cls, h_s = runner.gibbs_crank(d)
    results_crank = {"path_cls":h_cls, "path_alms":h_s, "obs_map":d, "obs_alms":s_obs,"config":{"NSIDE": config.NSIDE,
                                                            "L_MAX_SCALARS":config.L_MAX_SCALARS,
                                                           "N_gibbs":config.N_gibbs, "true_spectrum":cls_,
                                                            "N":config.N_gibbs, "N_crank": config.N_CN, "beta":config.beta_CN}}

    h_cls, h_s = runner.gibbs_mala(d)
    results_mala = {"path_cls":h_cls, "path_alms":h_s, "obs_map":d, "obs_alms":s_obs,"config":{"NSIDE": config.NSIDE,
                                                            "L_MAX_SCALARS":config.L_MAX_SCALARS,
                                                           "N_gibbs":config.N_gibbs, "true_spectrum":cls_,
                                                            "N":config.N_gibbs, "N_crank": config.N_CN, "tau":config.step_size_mala,
                                                            "var_mala":config.var_mala}}

    np.save("results_conj.npy", results_conj)
    np.save("results_crank.npy", results_crank)
    np.save("results_mala.npy", results_mala)


if __name__ == "__main__":
    #main()
    #utils.plot_results("results_short.npy", 7, check_alm=False)
    utils.compare("results_conj.npy", "results_crank.npy", "results_mala.npy", 8)

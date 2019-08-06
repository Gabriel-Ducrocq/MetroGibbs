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
import mala

Int = 149

def main():
    d, cls_, s_obs = utils.generate_sky_map()
    _, cls_others, _ = utils.generate_sky_map()
    config.observations = d

    conjgrad2 = conjugateGradient.CG2()
    true_mean2, err = conjgrad2.compute_mean(d, cls_others)
    if err != 0:
        print("Conjugate gradient did not converge")
        return None

    conjgrad1 = conjugateGradient.CG()
    true_mean1, err = conjgrad1.compute_mean(d, cls_others)
    if err != 0:
        print("Conjugate gradient did not converge")
        return None

    #true_mean2 = mala.unflat_map_to_pix(true_mean2)
    #h_cg = []
    #for i in range(10000):
    #    sol, err = conjgrad.run(d, cls_others)
    #    h_cg.append(sol)

    #h, estim_mean = gradient_ascent(d, cls_)
    #h = np.array(h)
    #plt.plot(h[:, Int].imag)
    #plt.axhline(y=true_mean[Int].imag, color="k", linewidth=1)
    #plt.show()
    #true_var, err = conjgrad.get_var(Int)
    grad_cst = mala.compute_gradient_log_constant_part2(d)
    history, s, warm_start = mala.mala2(cls_others, d, grad_cst, ratio=False)
    #history_r, s, warm_start = mala.mala2(cls_others, d, grad_cst, ratio=True)
    #history, s = crankNicolson(cls_, d)
    h = np.array(history)[:, Int]
    #h_r = np.array(history_r)[:, Int]
    plt.plot(h)
    #plt.plot(h_r)
    plt.axhline(y=true_mean1[Int].real, color='red', linewidth=1)
    plt.axhline(y=true_mean2[Int].real, color='green', linewidth=1)
    plt.show()
    plt.close()
    plt.hist(h[2000:].real, bins=25, label="UALA", density = True)
    #plt.hist(np.array(h_cg)[:, Int].real, bins=25, label="CG", density = True)
    plt.axvline(x=true_mean2[Int].real, color='red', linewidth=1)
    plt.legend(loc="upper right")
    plt.show()
    plt.close()
    """
    history, s = gradient_ascent(d, cls_)
    print(true_mean[Int])
    plt.plot(np.array(history)[:, Int].imag)
    plt.axhline(y=true_mean[Int].imag, color='red', linewidth=1)
    plt.show()
    plt.close()
    """
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
    """
    """
    h_means = []
    for i in range(100):
        d, cls_, s_obs = utils.generate_sky_map()
        conjgrad = conjugateGradient.CG()
        true_mean, err = conjgrad.compute_mean(d, cls_)
        if err != 0:
            print("Conjugate gradient did not converge")
            break

        h, estim_mean = gradient_ascent(d, cls_)
        diff = (estim_mean - true_mean)
        dist = np.sqrt(np.sum(np.concatenate((diff.real, diff.imag)) ** 2))
        h_means.append(dist)

    plt.plot(h_g)
    plt.axhline(y = np.mean(h_means), color = "k", linewidth = 1)
    plt.show()
    """
    #np.save("results_conj_no_grad.npy", results_conj)
    #np.save("results_crank_no_grad.npy", results_crank)
    #np.save("results_mala_no_grad.npy", results_mala)



if __name__ == "__main__":
    main()
#    utils.plot_results("results_crank_no_grad.npy", 120, check_alm=False)
#    utils.compare("results_conj_no_grad.npy", "results_crank_no_grad.npy", "results_mala_no_grad.npy", 120)

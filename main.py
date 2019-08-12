import mala
import utils
import matplotlib.pyplot as plt
import time
from CrankNicolson import crankNicolson, crankNicolson2
import metropolis
import config
import numpy as np
import runner
import healpy as hp
import conjugateGradient
from gradientDescent import gradient_ascent, gradient_ascent2
import mala

Int = 30

def main():

    d, cls_, s_obs = utils.generate_sky_map()
    _, cls_others, _ = utils.generate_sky_map()
    config.observations = d

    A = utils.get_Ylm()
    extended_cls = [cl for l in range(config.L_MAX_SCALARS + 1) for cl in cls_[l:]]
    extended_cls_real = (extended_cls[:(config.L_MAX_SCALARS + 1)] + extended_cls[(config.L_MAX_SCALARS + 2):])[2:]
    extended_cls_imag = extended_cls[(config.L_MAX_SCALARS + 2):]
    ###Attention, l'ordre est ici inversé !
    extended_cls = np.array(extended_cls_imag + extended_cls_real)

    mat = np.zeros(((config.L_MAX_SCALARS+1)**2 -4, (config.L_MAX_SCALARS+1)**2 -4), dtype=complex)
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
                    config.N - (config.L_MAX_SCALARS + 2) + config.L_MAX_SCALARS -2 + 1], np.zeros(mat.shape[1]), axis=0)

    input = np.concatenate((2*np.array(range(1, config.N - config.L_MAX_SCALARS - 2 + 1)), np.array(range(1, config.L_MAX_SCALARS-1 + 1))
                   , 2*np.array(range(1, config.N - config.L_MAX_SCALARS - 2 + 1))))

    #map = np.dot(mat,input)
    #print(map)
    overall_A = np.dot(A, mat)
    M = np.dot(overall_A.T, np.dot(np.diag(1/config.noise_covar), overall_A)) + extended_cls
    M_inv = np.linalg.inv(M)

    var_mano = M_inv[Int, Int]
    """
    extended_cls = [cl for l in range(config.L_MAX_SCALARS + 1) for cl in cls_[l:]]
    #extended_cls_real = (extended_cls[:(config.L_MAX_SCALARS + 1)] + extended_cls[(config.L_MAX_SCALARS + 2):])[2:]
    #extended_cls_imag = extended_cls[(config.L_MAX_SCALARS + 2):]
    #extended_cls = np.array(extended_cls_real + extended_cls_imag)
    diag = extended_cls
    denom_second = 1/np.array(extended_cls)
    denom_first = 1/np.array(extended_cls[config.L_MAX_SCALARS+1:])
    denom_second[[0, 1, config.L_MAX_SCALARS + 1]] = 0
    denom_first[0] = 0
    inverse_C = np.concatenate((denom_first, denom_second))

    cg_Matrix = np.diag(inverse_C) + M
    print(cg_Matrix)
    inv_Matrix = np.linalg.solve(cg_Matrix, np.ones(81))
    print(inv_Matrix[Int, Int])
    """

    conjgrad2 = conjugateGradient.CG2()
    conjgrad1 = conjugateGradient.CG()
    true_mean2, err = conjgrad2.compute_mean(d, cls_others)
    if err != 0:
        print("Conjugate gradient did not converge")
        return None

    variance_int1, err = conjgrad1.get_var(Int)


    variance_int, err = conjgrad2.get_var(Int)
    variance_int = variance_int.real
    if err != 0:
        print("Conjugate gradient did not converge for variance")
        return None



    print(variance_int1)
    print(variance_int)


    h_cg2 = []
    for i in range(10000):
        print(i)
        sol, err = conjgrad2.run(d, cls_others)
        h_cg2.append(sol)
    np.random.normal(size=config.Npix).astype(complex)
    h_cg1 = []
    for i in range(10000):
        print(i)
        sol, err = conjgrad1.run(d, cls_others)
        h_cg1.append(mala.flatten_map(sol))


    #h, estim_mean = gradient_ascent(d, cls_)
    h1 = np.array(h_cg1)
    h2 = np.array(h_cg2)
    #emp_var = np.var(h[:, Int])
    #print(h[:, Int])
    plt.hist(h1[:, Int], bins=50, alpha=0.5, density=True, label="Old")
    plt.hist(h2[:, Int], bins=50, alpha=0.5, density=True, label="New")
    plt.axvline(x=true_mean2[Int], color="k", linewidth=1)
    plt.axvline(x=true_mean2[Int] + np.sqrt(var_mano), color="k", linewidth=1)
    plt.axvline(x=true_mean2[Int] - np.sqrt(var_mano), color="k", linewidth=1)
    plt.legend(loc="upper right")
    #plt.axvline(x=true_mean2[Int] + np.sqrt(emp_var), color="red", linewidth=1)
    #plt.axvline(x=true_mean2[Int] - np.sqrt(emp_var), color="red", linewidth=1)
    plt.show()
    print(variance_int)
    print(variance_int1)

    """
    grad_cst = mala.compute_gradient_log_constant_part3(d)
    #history, s  = mala.mala3(cls_others, d, grad_cst)
    print("\n")
    print("\n")
    print("\n")
    #history, s, _, _ = mala.mala2(cls_others, d, grad_cst, unadjusted=False)
    #history_r, s, warm_start = mala.mala2(cls_others, d, grad_cst, ratio=True)
    history, s = crankNicolson2(cls_, d)
    h = np.array(history)[:, Int]
    #h_r = np.array(history_r)[:, Int]
    plt.plot(h)
    #plt.plot(h_r)
    plt.axhline(y=true_mean2[Int], color='green', linewidth=1)
    plt.axhline(y=true_mean2[Int] + np.sqrt(variance_int), color='green', linewidth=1)
    plt.axhline(y=true_mean2[Int] - np.sqrt(variance_int), color='green', linewidth=1)
    plt.show()
    plt.close()
    emp_var = np.var(h)
    plt.hist(h, label="ULA", density=True, alpha=0.5)
    #plt.hist(np.array(h_cg)[:, Int], bins=25, label="CG", density=True, alpha=0.5)
    plt.axvline(x=true_mean2[Int], color='red', linewidth=1)
    plt.axvline(x=true_mean2[Int] + np.sqrt(variance_int), color='green', linewidth=1)
    plt.axvline(x=true_mean2[Int] - np.sqrt(variance_int), color='green', linewidth=1)
    plt.axvline(x=true_mean2[Int] + np.sqrt(emp_var), color='red', linewidth=1)
    plt.axvline(x=true_mean2[Int] - np.sqrt(emp_var), color='red', linewidth=1)
    plt.legend(loc="upper right")
    plt.show()
    plt.close()
    print(variance_int)
    """
    """
    history, s = gradient_ascent2(d, cls_)
    h = np.array(history)
    plt.plot(h[:, Int])
    plt.axhline(y=true_mean2[Int], color='red', linewidth=1)
    plt.show()
    plt.close()


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
    #np.save("results_conj_other_implem.npy", results_conj)
    #np.save("results_crank_other_implem.npy", results_crank)
    #np.save("results_mala_other_implem.npy", results_mala)



if __name__ == "__main__":
    main()
#    utils.plot_results("results_crank_no_grad.npy", 120, check_alm=False)
#    utils.compare("results_conj_other_implem.npy", "results_crank_other_implem.npy", "results_mala_other_implem.npy", 50)

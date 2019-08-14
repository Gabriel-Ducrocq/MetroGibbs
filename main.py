import mala
import utils
import matplotlib.pyplot as plt
import time
from CrankNicolson import crankNicolson_good
import metropolis
import config
import numpy as np
import runner
import healpy as hp
import conjugateGradient
from gradientDescent import gradient_ascent, gradient_ascent2, unflat_map_to_pix, gradient_ascent_good
import mala

Int = 30

def main():

    d, cls_, s_obs = utils.generate_sky_map()
    _, cls_others, _ = utils.generate_sky_map()
    config.observations = d
    group_mat = utils.compute_grouping_matrix()

    alm = np.random.normal(size=config.N) + 1j * np.random.normal(size=config.N)
    alm_origin = alm.copy()
    alm.imag[:config.L_MAX_SCALARS+1] = 0
    alm[[0, 1, config.L_MAX_SCALARS+1]] = 0
    print(alm)
    print(alm_origin)

    A = utils.get_Ylm()
    A_transp = utils.get_Ylm_transp()
    map1 = utils.sph_transform_by_hand(alm, A)
    map2 = hp.alm2map(alm, nside=config.NSIDE)
    print(map1 - map2)
    map2_bis = hp.alm2map(alm_origin, nside=config.NSIDE)
    print(map2 - map2_bis)

    conjgrad3 = conjugateGradient.CG3()
    conjgrad4 = conjugateGradient.CG4(group_mat, A, A_transp)


    #testeur = np.random.normal(size=(config.L_MAX_SCALARS+1)**2) + 1j*np.random.normal(size=(config.L_MAX_SCALARS+1)**2)
    #testeur.imag[:config.L_MAX_SCALARS + 1] = 0
    #testeur[[0, 1, config.L_MAX_SCALARS + 1]] = 0
    #m = np.dot(group_mat.T, testeur)
    #m2 = conjugateGradient.flatten_map4(testeur[config.N - (config.L_MAX_SCALARS + 1):])
    #print(m)
    #print(m2)

    u = np.concatenate((2*np.ones(config.N - (config.L_MAX_SCALARS + 2)), np.ones(config.L_MAX_SCALARS-1),
                    3*np.ones(config.N - (config.L_MAX_SCALARS + 2))))

    u = np.concatenate((np.random.normal(size=config.N - (config.L_MAX_SCALARS + 2)),
                        np.random.normal(size=config.L_MAX_SCALARS-1),
                    np.random.normal(size=config.N - (config.L_MAX_SCALARS + 2))))

    #print(conjugateGradient.unflat_map_to_pix4(u) - np.dot(group_mat, u)[config.N - (config.L_MAX_SCALARS +1):])
    m1 = np.dot(A_transp, np.dot(A, np.dot(group_mat, u)))
    m2 = hp.map2alm(hp.alm2map(conjugateGradient.unflat_map_to_pix4(u), nside=config.NSIDE), lmax=config.L_MAX_SCALARS)
    print(m2-m1[config.N-(config.L_MAX_SCALARS+2)+1:])
    print(m1)

    #print(A_transp.shape)
    #e1 = np.dot(group_mat.T, np.dot(A_transp,d))
    #e2 = hp.map2alm(d, lmax=config.L_MAX_SCALARS)
    #print(e1)
    #print(e2)
    #print(e1[config.N - (config.L_MAX_SCALARS + 1):] - e2)
    #print(e1 - e2)




    u = np.concatenate((2*np.ones(config.N - (config.L_MAX_SCALARS + 2)), np.ones(config.L_MAX_SCALARS-1),
                    3*np.ones(config.N - (config.L_MAX_SCALARS + 2))))

    #u = np.concatenate((np.random.normal(size=config.N - (config.L_MAX_SCALARS + 2)), np.random.normal(size=config.L_MAX_SCALARS-1),
    #                np.random.normal(size=config.N - (config.L_MAX_SCALARS + 2))))

    #u[0] = 0
    #u[config.N - config.L_MAX_SCALARS] = 0
    #u[config.N - config.L_MAX_SCALARS +1] = 0
    #u[config.N] = 0

    degroup_mat = utils.compute_grouping_inverse()
    e = np.dot(group_mat, u)
    print(e[74+4])
    print(e[74 - (config.N - (config.L_MAX_SCALARS +1) + 2) + 1 - 4])
    print(74 - (config.N - (config.L_MAX_SCALARS +1) + 2) + 1 - 4)
    v = np.dot(degroup_mat, np.dot(group_mat, u))
    print(v)
    print((config.N - (config.L_MAX_SCALARS +1) + 2) + 1 - 4)
    print(np.dot(group_mat, u))
    #e = np.dot(degroup_mat, np.dot(A_transp, np.dot(A, np.dot(group_mat, u))))
    #print(e - u)
    #print(u)
    #print("\n")
    #print(np.vstack((u, e)).T)


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
    conjgrad3 = conjugateGradient.CG3()
    conjgrad4 = conjugateGradient.CG4(group_mat, A, A_transp)
    true_mean2, err = conjgrad2.compute_mean(d, cls_others)
    if err != 0:
        print("Conjugate gradient did not converge")
        return None

    true_mean3, err = conjgrad3.compute_mean(d, cls_others)
    if err != 0:
        print("Conjugate gradient did not converge")
        return None

    true_mean4, err = conjgrad4.compute_mean(d, cls_others)
    if err != 0:
        print("Conjugate gradient did not converge")
        return None


    variance_int1, err = conjgrad1.get_var(Int)


    variance_int, err = conjgrad2.get_var(Int)
    variance_int2 = variance_int.real
    if err != 0:
        print("Conjugate gradient did not converge for variance")
        return None

    variance_int3, err = conjgrad3.get_var(Int)
    variance_int3 = variance_int.real
    if err != 0:
        print("Conjugate gradient did not converge for variance")
        return None

    """
    variance_true, precision_true, mat, A = utils.compute_variance_matrix(cls_others)
    chol_var = np.linalg.cholesky(variance_true)
    chol_precision = np.linalg.cholesky(precision_true)

    u = hp.map2alm((1/config.noise_covar)*d, lmax=config.L_MAX_SCALARS)
    u = conjugateGradient.flatten_map3(u)
    mean = np.dot(variance_true, u)

    true_mean2 = conjugateGradient.flatten_map3(unflat_map_to_pix(true_mean2))
    print("\n")
    print(mean[Int])
    print(true_mean2[Int])
    print(true_mean4[Int])
    print(true_mean3[Int])
    print("\n")
    print(variance_int1)
    print(variance_int2)
    print(variance_int3)
    print(variance_true[Int, Int])
    print(conjgrad3.get_var(Int))
    print(conjgrad4.get_var(Int))

    h_cg2 = []
    for i in range(1000):
        print(i)
        sol, err = conjgrad2.run(d, cls_others)
        h_cg2.append(sol)


    h_cg1 = []
    for i in range(1000):
        print(i)
        sol, err = conjgrad1.run(d, cls_others)
        h_cg1.append(mala.flatten_map(sol))


    h_cg3 = []
    for i in range(1000):
        print(i)
        sol, err = conjgrad3.run(d, cls_others)
        h_cg3.append(sol)
        
    
    h_cg4 = []
    for i in range(1000):
        print(i)
        sol, err = conjgrad4.run(d, cls_others)
        h_cg4.append(sol)

    h_chol = []
    true_mean = np.dot(variance_true,
                      conjugateGradient.flatten_map3(hp.sphtfunc.map2alm((1 / config.noise_covar) * d, lmax=config.L_MAX_SCALARS)))
    for i in range(1000):
        print(i)
        #v = np.dot(chol_var, np.random.normal(size = 77)) + true_mean
        #omega0 = np.sqrt(1/conjgrad3.cls)*np.random.normal(size=len(conjgrad3.cls))
        #omega1 = hp.sphtfunc.map2alm((1/np.sqrt(config.noise_covar))*np.random.normal(size=config.Npix)
        #                             ,lmax=config.L_MAX_SCALARS)
        #omega1 = np.dot(degroup_mat, np.dot(A_transp, (1 / np.sqrt(config.noise_covar))*np.random.normal(size=config.Npix)))
        #u = hp.sphtfunc.map2alm(((1/config.noise_covar) * d), lmax=config.L_MAX_SCALARS)
        u = np.dot(degroup_mat,np.dot(A_transp, (1 / config.noise_covar) * d))
        #b = omega0 + conjugateGradient.flatten_map3(u+omega1)
        #b = omega0 + u + omega1
        b = np.dot(chol_precision, np.random.normal(size = precision_true.shape[1])) + u
        v = np.dot(variance_true, b)
        h_chol.append(v)



    #h, estim_mean = gradient_ascent(d, cls_)
    #h1 = np.array(h_cg1)
    #h2 = np.array(h_cg2)
    h4 = np.array(h_cg4)
    hchol = np.array(h_chol)
    stdd2 = np.std(hchol[:, Int])
    #emp_var = np.var(h[:, Int])
    #print(h[:, Int])
    #plt.hist(h1[:, Int], bins=25, alpha=0.2, density=True, label="Old")
    #plt.hist(h2[:, Int], bins=25, alpha=0.2, density=True, label="New")
    plt.hist(h4[:, Int], bins=15, alpha=0.5, density=True, label="CG4")
    plt.hist(hchol[:, Int], bins=15, alpha=0.5, density=True, label="Cholesky")
    plt.axvline(x=true_mean4[Int], color="k", linewidth=1)
    plt.axvline(x=true_mean4[Int] + np.sqrt(variance_true[Int, Int]), color="k", linewidth=1)
    plt.axvline(x=true_mean4[Int] - np.sqrt(variance_true[Int, Int]), color="k", linewidth=1)
    plt.axvline(x=true_mean4[Int] + stdd2, color="r", linewidth=1)
    plt.axvline(x=true_mean4[Int] - stdd2, color="r", linewidth=1)
    plt.legend(loc="upper right")
    #plt.axvline(x=true_mean2[Int] + np.sqrt(emp_var), color="red", linewidth=1)
    #plt.axvline(x=true_mean2[Int] - np.sqrt(emp_var), color="red", linewidth=1)
    plt.show()

    """

    #ICI POUR MALA !!!
    grad_cst = utils.compute_gradient_log_constant_part(d)
    #history, s  = mala.mala3(cls_others, d, grad_cst)
    print("\n")
    print("\n")
    print("\n")
    #history, s = mala.mala_good(cls_others, d, grad_cst, unadjusted=False)
    history, s = crankNicolson_good(cls_, d)
    h = np.array(history)[:, Int]
    #h_r = np.array(history_r)[:, Int]
    plt.plot(h)
    #plt.plot(h_r)
    plt.axhline(y=true_mean4[Int], color='green', linewidth=1)
    plt.axhline(y=true_mean4[Int] + np.sqrt(variance_int), color='green', linewidth=1)
    plt.axhline(y=true_mean4[Int] - np.sqrt(variance_int), color='green', linewidth=1)
    plt.show()
    plt.close()
    #emp_var = np.var(h)
    #plt.hist(h, label="ULA", density=True, alpha=0.5)
    #plt.hist(np.array(h_cg4)[:, Int], bins=25, label="CG", density=True, alpha=0.5)
    #plt.axvline(x=true_mean4[Int], color='red', linewidth=1)
    #plt.axvline(x=true_mean4[Int] + np.sqrt(variance_int), color='green', linewidth=1)
    #plt.axvline(x=true_mean4[Int] - np.sqrt(variance_int), color='green', linewidth=1)
    #plt.axvline(x=true_mean4[Int] + np.sqrt(emp_var), color='red', linewidth=1)
    #plt.axvline(x=true_mean4[Int] - np.sqrt(emp_var), color='red', linewidth=1)
    #plt.legend(loc="upper right")
    #plt.show()
    #plt.close()
    #print(variance_int)

    """
    #Ici pour gradient !!!
    history, s = gradient_ascent_good(d, cls_others)
    h = np.array(history)
    plt.plot(h[:, Int])
    plt.axhline(y=true_mean4[Int], color='red', linewidth=1)
    plt.axhline(y=true_mean3[Int], color='green', linewidth=1)
    plt.show()
    plt.close()
    print(h[0, Int])
    print(history[0][Int])
    print(np.array(history).shape)
    print(history)
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

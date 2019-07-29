import mala
import utils
import matplotlib.pyplot as plt
import time
import CrankNicolson
import metropolis
import config
import numpy as np
import runner
import healpy as hp

Int = 60
CLS = np.ones(config.N)

def main():
    d, cls_, s_obs = utils.generate_sky_map()
    print("TO PRINT")
    print(s_obs[:10])
    print(hp.map2alm(d, lmax=config.L_MAX_SCALARS)[:10])
    config.observations = d
    #print(true_mean[:3])
    #print(err)
    #var, err = conjgrad.get_var(Int)
    all_sim = []
    #for i in range(100000):
    #    sim = conjgrad.run(d, cls_)
    #    all_sim.append(sim)
    #print(s_obs.shape)
    #for i in range(50000):
    #    ss = samplingInvGamm.sampling(s_obs)
    #    all_sim.append(ss)

    #paths = np.array(all_sim)
    #plt.hist(paths[:,4], bins=100)
    #plt.axvline(x=np.mean(paths[:, 4]),  color='k', linestyle='dashed', linewidth=1)
    #plt.axvline(x=cls_[4], color='k', linewidth=1)
    #plt.show()
    #other_mean, err = np.mean(np.array(all_sim), axis=0)
    #print(other_mean)
    #print(len(true_mean))


    h_cls, h_s = runner.gibbs(d)
    results = {"path_cls":h_cls, "path_alms":h_s, "obs_map":d, "obs_alms":s_obs,"config":{"NSIDE": config.NSIDE,
                                                            "L_MAX_SCALARS":config.L_MAX_SCALARS,
                                                           "N_gibbs":config.N_gibbs, "true_spectrum":cls_}}

    np.save("results.npy", results)

    """
    #grad_constant_part = mala.compute_gradient_log_constant_part(d)
    start_time = time.time()
    #result = metropolis.metropolis(s_init, theta_init, cls_init)

    #s, h = mala.mala(cls_init, d, grad_constant_part)
    s, h = CrankNicolson.crankNicolson(cls_init, d)
    components = np.array(h)
    print(components.shape)
    real_mean = true_mean[Int]

    print("True variance")
    print(true_mean[50])
    print(var)

    #stdd_emp = np.sqrt(np.var(part[50000:]))
    #mean_emp = np.mean(part[50000:])
    plt.plot(components[:, Int].real)
    plt.axhline(y=real_mean, color='r', linestyle='-')
    plt.axhline(y=real_mean + np.sqrt(var), color='g', linestyle='-')
    plt.axhline(y=real_mean - np.sqrt(var), color='g', linestyle='-')
    plt.axhline(y=real_mean + 2*np.sqrt(var), color='g', linestyle='-')
    plt.axhline(y=real_mean - 2*np.sqrt(var), color='g', linestyle='-')
    #plt.axhline(y=mean_emp + 2*stdd_emp, color='r', linestyle='-')
    #plt.axhline(y=mean_emp - 2*stdd_emp, color='r', linestyle='-')
    #plt.axhline(y=mean_emp + stdd_emp, color='r', linestyle='-')
    #plt.axhline(y=mean_emp - stdd_emp, color='r', linestyle='-')
    print(time.time() - start_time)
    plt.show()
    plt.close()
    comparison = np.random.normal(0, np.sqrt(var), size=99000)
    comparison2 = np.random.normal(0, np.sqrt(cls_init[Int]), size=9900)
    #plt.hist(part[60000:], bins=50, alpha=0.5, label="CN", density=True)
    #plt.hist(comparison, bins=50, alpha=0.5, label="Posterior", density=True)
    #plt.hist(comparison2, bins=50, alpha=0.5, label="Prior", density=True)
    #plt.axvline(x=true_mean[Int] - np.sqrt(var), color='g', linestyle='-')
    #plt.axvline(x=true_mean[Int] + np.sqrt(var), color='g', linestyle='-')
    #plt.axvline(x=true_mean[Int] - 2*np.sqrt(var), color='g', linestyle='-')
    #plt.axvline(x=true_mean[Int] + 2*np.sqrt(var), color='g', linestyle='-')
    plt.legend(loc="upper right")
    plt.show()
    """


if __name__ == "__main__":
    #main()
    utils.plot_results("results.npy", 2, check_alm=False)

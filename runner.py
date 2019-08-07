import numpy as np
import utils
import config
from conjugateGradient import CG2
import samplingInvGamm
import metropolis
from CrankNicolson import crankNicolson
import mala


def gibbs_conj(d):
    _, cls_init, s_init = utils.generate_sky_map()
    conjgrad = CG2()
    history_s = []
    history_cls = []
    cls = cls_init
    s = s_init
    grad_constant_part = mala.compute_gradient_log_constant_part(d)
    for i in range(config.N_gibbs):
        print(i)
        #s, cls = metropolis.metropolis(s, cls)
        cls = samplingInvGamm.sampling(s)
        s, err = conjgrad.run(d, cls)
        if err != 0:
            print("Conjugate Gradient did not converge")
            break

        history_s.append(s)
        history_cls.append(cls)
        s = mala.unflat_map_to_pix(s)

    return history_cls, history_s


def gibbs_crank(d):
    _, cls_init, s_init = utils.generate_sky_map()
    history_s = []
    history_cls = []
    cls = cls_init
    s = s_init
    grad_constant_part = mala.compute_gradient_log_constant_part(d)
    for i in range(config.N_gibbs):
        print(i)
        #s, cls = metropolis.metropolis(s, cls)
        cls = samplingInvGamm.sampling(s)
        h, s = crankNicolson(cls, d)

        history_s.append(s)
        history_cls.append(cls)

    return history_cls, history_s


def gibbs_mala(d):
    _, cls_init, s_init = utils.generate_sky_map()
    history_s = []
    history_cls = []
    s = s_init
    grad_constant_part = mala.compute_gradient_log_constant_part2(d)
    for i in range(config.N_gibbs):
        print(i)
        #s, cls = metropolis.metropolis(s, cls)
        cls = samplingInvGamm.sampling(s)

        h, s, warm_start, h_g = mala.mala2(cls, d, grad_constant_part)
        history_s.append(s)
        history_cls.append(cls)
        s = mala.unflat_map_to_pix(s)

    return history_cls, history_s
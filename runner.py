import numpy as np
import utils
import config
from conjugateGradient import CG
import samplingInvGamm
import metropolis


def gibbs(d):
    _, cls_init, s_init = utils.generate_sky_map()
    conjgrad = CG()
    history_s = []
    history_cls = []
    cls = cls_init
    s = s_init
    for i in range(config.N_gibbs):
        print(i)
        #s, cls = metropolis.metropolis(s, cls)
        cls = samplingInvGamm.sampling(s)
        s, err = conjgrad.run(d, cls)
        if err != 0:
            print("Conjugate Gradient did not converge")
            break

        #cls = samplingInvGamm.sampling(s)
        #cls_half = samplingInvGamm.sampling(s_half)
        #history_s.append(s_half)
        #history_cls.append(cls_half)
        history_s.append(s)
        history_cls.append(cls)

    return history_cls, history_s


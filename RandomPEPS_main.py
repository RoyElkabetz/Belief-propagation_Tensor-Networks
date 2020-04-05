

import numpy as np
import copy as cp
from scipy import linalg
import matplotlib.pyplot as plt
import ncon
import time
import pickle
import pandas as pd



import SimpleUpdate as BP
import ncon_lists_generator as nlg
import DoubleEdgeFactorGraphs as defg
import StructureMatrixGenerator as tnf
import RandomPEPS as hmf
import bmpslib as bmps

def randomPEPSmainFunction():



    #np.random.seed(seed=9)

    N, M = 4, 4
    bc = 'open'
    dE = 1e-5
    t_max = 200
    dumping = 0.2
    epsilon = 1e-5
    D_max = [3]
    mu = -1
    sigma = 0
    Jk = np.random.normal(mu, sigma, np.int((N - 1) * M + (M - 1) * N))
    dt = [0.5, 0.1, 0.05, 0.01, 0.005]
    iterations = 10
    Dp = [16, 32]
    d = 2

    smat, imat = tnf.finitePEPSobcStructureMatrixGenerator(N, M)
    BP_data = []
    SU_data = []

    for D in D_max:
        b, time_su = hmf.RandomPEPS_SU(N, M, Jk, dE, D, bc, dt, iterations)
        TT0, LL0 = b[0], b[1]
        a, time_bp = hmf.RandomPEPS_BP(N, M, Jk, dE, D, t_max, epsilon, dumping, bc, dt, iterations, [TT0, LL0])
        BP_data.append(a)
        SU_data.append(b)



    rho_SU = []
    rho_SU_0_bmps = []
    rho_SU_0_bmps_single = []


    data_bp = BP_data
    data_su = SU_data

    indices = []
    counter = 0
    counter2 = N * (M - 1) - 1 + (N - 1) * (M - 1) + 1
    counter3 = 0
    while counter3 < N:
        indices += range(counter, counter + M - 1)
        indices.append(counter2)
        counter += (M - 1)
        counter2 += 1
        counter3 += 1
    indices.pop()

    print('calc_rdms')
    for ii in range(len(D_max)):

        graph = data_bp[ii]
        TT_SU_0, LL_SU_0, TT_SU, LL_SU = data_su[ii]
        TT_SU_0_bmps = cp.deepcopy(TT_SU_0)

    #
    ######### CALCULATING REDUCED DENSITY MATRICES  ########
    #

        for i in range(len(TT_SU)):
            rho_SU.append(BP.singleSiteRDM(i, TT_SU, LL_SU, smat))
        rho_BP = graph.calculateRDMSfromFactorBeliefs()

        TT_SU_0_bmps = BP.absorbAllTensorNetWeights(TT_SU_0_bmps, LL_SU_0, smat)
        TT_SU_0_bmps = tnf.PEPS_OBC_broadcast_to_Itai(TT_SU_0_bmps, [N, M], d, D_max[ii])
        SU_0_peps = bmps.peps(N, M)
        for t, T in enumerate(TT_SU_0_bmps):
            i, j = np.unravel_index(t, [N, M])
            SU_0_peps.set_site(T, i, j)
        for dp_idx, dp in enumerate(Dp):
            print('Dp:', dp)
            rho_SU_0_bmps.append(bmps.calculate_PEPS_2RDM(SU_0_peps, dp))
            rho_SU_0_bmps_single.append([])
            for jj in indices:
                rho_SU_0_bmps_single[dp_idx].append(np.einsum(rho_SU_0_bmps[dp_idx][jj], [0, 1, 2, 2], [0, 1]))
            rho_SU_0_bmps_single[dp_idx].append(np.einsum(rho_SU_0_bmps[dp_idx][jj], [1, 1, 2, 3], [2, 3]))



    return rho_SU_0_bmps_single, rho_SU, rho_BP




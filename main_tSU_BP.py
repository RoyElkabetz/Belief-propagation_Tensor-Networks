import numpy as np
import random
import time
import matplotlib.pyplot as plt
import copy as cp
import pickle
import pandas as pd

import RandomPEPS as rpeps
import StructureMatrixGenerator as tnf
import DoubleEdgeFactorGraphs as defg
import SimpleUpdate as BP
import bmpslib as bmps


########################################################################################################################
#                                                                                                                      #
#                                    TRIVIAL SIMPLE UPDATE (TSU) ON RANDOM PEPS                                        #
#                                                                                                                      #
########################################################################################################################


def randomPEPSmainFunction(N, M):

    # PEPS parameters
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

    # Set lists
    BP_data = []
    SU_data = []
    rho_SU = []
    rho_SU_0_bmps = []
    rho_SU_0_bmps_single = []

    # Run Simple Update & Belief Propagation
    for D in D_max:
        b, time_su = rpeps.RandomPEPS_SU(N, M, Jk, dE, D, bc, dt, iterations)
        TT0, LL0 = b[0], b[1]
        a, time_bp = rpeps.RandomPEPS_BP(N, M, Jk, dE, D, t_max, epsilon, dumping, bc, dt, iterations, [TT0, LL0])
        BP_data.append(a)
        SU_data.append(b)

    # Arrange single site rdms order in list as in bmpslib
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

    # Caclualting reduced density matrices
    for ii in range(len(D_max)):
        graph = BP_data[ii]
        TT_SU_0, LL_SU_0, TT_SU, LL_SU = SU_data[ii]
        TT_SU_0_bmps = cp.deepcopy(TT_SU_0)

        # RDMS using BP and SU
        for i in range(len(TT_SU)):
            rho_SU.append(BP.singleSiteRDM(i, TT_SU, LL_SU, smat))
        rho_BP = graph.calculateRDMSfromFactorBeliefs()

        # RDMS using BMPO from bmpslib
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


########################################################################################################################
#                                                                                                                      #
#                     TRIVIAL SIMPLE UPDATE (TSU) vs BELIEF PROPAGATION (BP) IN PEPS EXPERIMENT                        #
#                                                                                                                      #
########################################################################################################################

# Number of experiments
n = 100

# PEPS width and length
N, M = 4, 4

# Info for file saving
date = "2020_04_05" + "_"
experimentNum = "1" + "_"
PEPSsize = str(int(N * M)) + "_"
file_name = date + experimentNum + PEPSsize

# Set lists for ttd (total trace distance)
ttd_SU_bMPO = np.zeros((1, n), dtype=float)
ttd_SU_BP = np.zeros((1, n), dtype=float)
ttd_bMPO16_bMPO32 = np.zeros((1, n), dtype=float)

# Run experiment
for i in range(n):
    print('\n')
    print('i:', i)
    rho_SU_0_bmps_single, rho_SU, rho_BP = randomPEPSmainFunction(N, M)
    rho_SU_0_bmps16_single = rho_SU_0_bmps_single[0]
    rho_SU_0_bmps32_single = rho_SU_0_bmps_single[1]
    for j in range(N):
        ttd_SU_bMPO[0][i] += BP.traceDistance(rho_SU_0_bmps16_single[j], rho_SU[j]) / N
        ttd_SU_BP[0][i] += BP.traceDistance(rho_BP[j], rho_SU[j]) / N
        ttd_bMPO16_bMPO32[0][i] += BP.traceDistance(rho_SU_0_bmps16_single[j], rho_SU_0_bmps32_single[j]) / N
av_ttd_SU_bMPO = np.sum(ttd_SU_bMPO) / n
av_ttd_SU_BP = np.sum(ttd_SU_BP) / n
av_ttd_bMPO16_bMPO32 = np.sum(ttd_bMPO16_bMPO32) / n

# Display to screen
print('\n')
print('----------------------------------------------------------------')
print('                           RESULTS                              ')
print('----------------------------------------------------------------')
print(' Average total trace-distance (bMPO16 - bMPO32) = %.12f' % av_ttd_bMPO16_bMPO32)
print(' Average total trace-distance (SU - bMPO16)     = %.12f' % av_ttd_SU_bMPO)
print(' Average total trace-distance (SU - BP)         = %.12f' % av_ttd_SU_BP)
print('----------------------------------------------------------------')


# Saving data from all experiments to .m file
a1 = [ttd_SU_bMPO, av_ttd_SU_bMPO]
a2 = [ttd_SU_BP, av_ttd_SU_BP]
a3 = [ttd_bMPO16_bMPO32, av_ttd_bMPO16_bMPO32]
pickle.dump(a1, open(file_name + 'OBC_Random_PEPS_SU_bMPO16.p', "wb"))
pickle.dump(a2, open(file_name + 'OBC_Random_PEPS_SU_BP.p', "wb"))
pickle.dump(a3, open(file_name + 'OBC_Random_PEPS_bMPO16_bMPO32.p', "wb"))

# Saving averages results to .xlsx file
data = np.zeros((n, 3), dtype=float)
data[:, 0] = ttd_SU_BP
data[:, 1] = ttd_SU_bMPO
data[:, 2] = ttd_bMPO16_bMPO32
df = pd.DataFrame(data, columns=['SU-BP', 'SU-bMPO(16)', 'bMPO(16)-bMPO(32)'], index=range(1, n + 1))
filepath = '2020_04_03_1_16_trivial-SU_BP_experiment' + '.xlsx'
df.to_excel(filepath, index=True)



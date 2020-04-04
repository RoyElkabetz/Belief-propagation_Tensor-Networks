import numpy as np
import random
import time
import matplotlib.pyplot as plt
import copy as cp
import pickle
import pandas as pd

import RandomPEPS_main as rpm
import StructureMatrixGenerator as tnf
import DoubleEdgeFactorGraphs as defg
import SimpleUpdate as BP


########################################################################################################################
#                                                                                                                      #
#                     TRIVIAL SIMPLE UPDATE (TSU) vs BELIEF PROPAGATION (BP) IN PEPS EXPERIMENT                        #
#                                                                                                                      #
########################################################################################################################


n = 10
N = 100
ttd_SU_bMPO = np.zeros((1, n), dtype=float)
ttd_SU_BP = np.zeros((1, n), dtype=float)
ttd_bMPO16_bMPO32 = np.zeros((1, n), dtype=float)
#ttd_bMPO8_bMPO16, av_ttd_bMPO8_bMPO16 = pickle.load(open("2DTN_data/2019_02_27_2_100_OBC_Random_PEPS_D_3_SU0_8vs16.p", "rb"))

for i in range(n):
    print('\n')
    print('i:', i)
    rho_SU_0_bmps_single, rho_SU, rho_BP = rpm.randomPEPSmainFunction()
    rho_SU_0_bmps16_single = rho_SU_0_bmps_single[0]
    rho_SU_0_bmps32_single = rho_SU_0_bmps_single[1]
    for j in range(N):
        ttd_SU_bMPO[0][i] += BP.traceDistance(rho_SU_0_bmps16_single[j], rho_SU[j]) / N
        ttd_SU_BP[0][i] += BP.traceDistance(rho_BP[j], rho_SU[j]) / N
        ttd_bMPO16_bMPO32[0][i] += BP.traceDistance(rho_SU_0_bmps16_single[j], rho_SU_0_bmps32_single[j]) / N
av_ttd_SU_bMPO = np.sum(ttd_SU_bMPO) / n
av_ttd_SU_BP = np.sum(ttd_SU_BP) / n
av_ttd_bMPO16_bMPO32 = np.sum(ttd_bMPO16_bMPO32) / n
print('\n')
print('av_ttd_SU_bMPO', av_ttd_SU_bMPO)
print('av_ttd_SU_BP', av_ttd_SU_BP)
file_name1 = "2020_04_03_1_16_OBC_Random_PEPS_SU_bMPO16"
file_name2 = "2020_04_03_1_16_OBC_Random_PEPS_SU_BP"
file_name3 = "2020_04_03_1_16_OBC_Random_PEPS_bMPO16_bMPO32"
a1 = [ttd_SU_bMPO, av_ttd_SU_bMPO]
a2 = [ttd_SU_BP, av_ttd_SU_BP]
a3 = [ttd_bMPO16_bMPO32, av_ttd_bMPO16_bMPO32]

pickle.dump(a1, open(file_name1 + '.p', "wb"))
pickle.dump(a2, open(file_name2 + '.p', "wb"))
pickle.dump(a3, open(file_name3 + '.p', "wb"))


data = np.zeros((n, 3), dtype=float)
data[:, 0] = ttd_SU_BP
data[:, 1] = ttd_SU_bMPO
data[:, 2] = ttd_bMPO16_bMPO32
# [BP energy, SU energy]
df = pd.DataFrame(data, columns=['SU-BP', 'SU-bMPO(16)', 'bMPO(16)-bMPO(32)'], index=range(1, n + 1))
filepath = '2020_04_03_1_16_trivial-SU_BP_experiment' + '.xlsx'
df.to_excel(filepath, index=True)

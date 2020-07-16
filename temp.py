import numpy as np
import random
import time
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
color_list = ['salmon', 'limegreen', 'mediumturquoise', 'cornflowerblue', 'fuchsia', 'khaki']

import copy as cp
import pickle
import pandas as pd
import sys
#from ipywidgets import IntProgress
#from IPython.display import display


import RandomPEPS as rpeps
import StructureMatrixGenerator as smg
import trivialSimpleUpdate as tsu
import DoubleEdgeFactorGraphs as defg
import SimpleUpdate as su
import bmpslib as bmps

# tSU and BP parameters
N, M = 10, 10                                                  # NxM PEPS
bc = 'open'                                                   # boundary conditions
dw = 1e-6                                                    # maximal error allowed between two-body RDMS
d = 2                                                         # tensor network physical bond dimension
bond_dimensions = [2, 3]                                   # maximal virtual bond dimensions allowed for truncation
t_max = 1000                                               # maximal number of BP iterations
epsilon = 1e-10                                               # convergence criteria for BP messages (not used)
dumping = 0.                                                  # BP messages dumping between [0, 1]
iterations = 1000                                             # maximal number of tSU iterations
sched = 'parallel'                                            # tSU scheduling scheme
num_experiments = 20                                       # number of random experiments for each bond dimension
smat, _ = smg.finitePEPSobcStructureMatrixGenerator(N, M)     # generating the PEPS structure matrix
n, m = smat.shape

ATD_D = []         # Averaged Trace Distance (ATD) for each virtual bond dimension D
BP_num_D = []      # numbeer of BP iterations
tSU_num_D = []     # numbeer of tSU iterations

for D_max in bond_dimensions:
    ATD_tot = []
    BP_iters = []
    tSU_iters = []
    print('\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
    print('|               D = {}               |'.format(D_max))
    print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')


    for e in range(num_experiments):
        #if e % 10 == 0:
        print(e)

        # draw some random PEPS Tensor Network
        tensors, weights = smg.randomTensornetGenerator(smat, d, D_max)
        BP_tensors, BP_weights = cp.deepcopy(tensors), cp.deepcopy(weights)

        # constructing the dual double-edge factor graph and run a single BP iteration
        graph = defg.defg()
        graph = su.TNtoDEFGtransform(graph, BP_tensors, BP_weights, smat)
        graph.sumProduct(1, epsilon, dumping, initializeMessages=1, printTime=0, RDMconvergence=0)
        BP_rdm = []
        for j in range(m):
            BP_rdm.append(tsu.BPdoubleSiteRDM1(j, BP_tensors, BP_weights, smat, cp.deepcopy(graph.messages_n2f)))

        # run BP and calculate two body rdms between two consecutive BP iterations
        for t in range(t_max):
            graph.sumProduct(1, epsilon, dumping, initializeMessages=1, printTime=0, RDMconvergence=0)

            ATD_BP = 0
            BP_rdm_next = []
            for j in range(m):
                BP_rdm_next.append(tsu.BPdoubleSiteRDM1(j,
                                                        BP_tensors,
                                                        BP_weights,
                                                        smat,
                                                        cp.deepcopy(graph.messages_n2f)))

                ATD_BP += tsu.traceDistance(BP_rdm_next[j], BP_rdm[j])
                BP_rdm[j] = BP_rdm_next[j]
            ATD_BP /= m
            # print('The ATD_BP is: {} at iteration {}'.format(ATD_BP, t))
            if ATD_BP < dw:
                # print('\n')
                # print('The final ATD_BP is: {} at iteration {}'.format(ATD_BP, t + 1))
                break
        BP_iters.append(t + 2)

        # calculate the double site rdm in tsu
        tSU_rdm = []
        for i in range(m):
            tSU_rdm.append(tsu.doubleSiteRDM(i, tensors, weights, smat))

            # trivial SU run
        for i in range(iterations):
            tensors_next, weights_next = tsu.trivialsimpleUpdate(tensors,
                                                                 weights,
                                                                 smat,
                                                                 D_max,
                                                                 scheduling='parallel')
            ATD = 0
            tSU_rdm_next = []
            for j in range(m):
                tSU_rdm_next.append(tsu.doubleSiteRDM(j, tensors_next, weights_next, smat))
                ATD += tsu.traceDistance(tSU_rdm_next[j], tSU_rdm[j])
                tSU_rdm[j] = tSU_rdm_next[j]
            ATD /= m
            if ATD < dw:
                # print('The ATD is: {} at iteration {}'.format(ATD, i))
                tensors = tensors_next
                weights = weights_next
                break
            tensors = tensors_next
            weights = weights_next
        tSU_iters.append(i + 1)

        # calculate Averaged Trace Distance between the BP and tSU rdms.
        ATD_BP_tSU = 0
        for i in range(m):
            ATD_BP_tSU += tsu.traceDistance(BP_rdm[i], tSU_rdm[i])
        ATD_BP_tSU /= m
        # print('the total ATD between BP and tSU is {}'.format(ATD_BP_tSU))
        ATD_tot.append(ATD_BP_tSU)
    ATD_D.append(ATD_tot)
    BP_num_D.append(BP_iters)
    tSU_num_D.append(tSU_iters)

data_name = 'data' + str(N) + 'x' + str(M) + '_random_PEPS'
description_name =  'parameters' + str(N) + 'x' + str(M) + '_random_PEPS'
data = np.asarray([ATD_D, BP_num_D, tSU_num_D])
parameters = np.asarray([['ATD', 'BP', 'tSU'],
                         ['N x M', [N, M]],
                         ['bond_dimensions', bond_dimensions],
                         ['bc', bc],
                         ['d', d],
                         ['dw', dw],
                         ['BP t_max', t_max],
                         ['BP epsilon', epsilon],
                         ['BP dumping', dumping],
                         ['tSU t_max', iterations],
                         ['num of experiments', num_experiments]])
np.save(data_name, data)
np.save(description_name, parameters)

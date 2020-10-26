import numpy as np
import random
import time
import matplotlib.pyplot as plt
import copy as cp
import pickle
import pandas as pd

import RandomPEPS as rpeps
import StructureMatrixGenerator as smg
import trivialSimpleUpdate as tsu
import DoubleEdgeFactorGraphs as defg
import SimpleUpdate as su
import bmpslib as bmps


########################################################################################################################
#                                                                                                                      #
#                                    TRIVIAL SIMPLE UPDATE (TSU) ON RANDOM PEPS                                        #
#                                                                                                                      #
########################################################################################################################
np.random.seed(1)

# tSU parameters
N, M = 5, 5
bc = 'open'
dw = 1e-10
D_max = 3
t_max = 100
epsilon = 1e-5
dumping = 0.2
iterations = 30
d = 2
smat, _ = smg.finitePEPSobcStructureMatrixGenerator(N, M)
tensors_tsu, weights_tsu = smg.randomTensornetGenerator(smat, d, D_max)
n, m = smat.shape

# SU parameters
Z = np.array([[1, 0], [0, -1]])
Y = np.array([[0, -1j], [1j, 0]])
X = np.array([[0, 1], [1, 0]])
Sz = 0.5 * Z
Sy = 0.5 * Y
Sx = 0.5 * X
Opi = [Sx, Sy, Sz]
Opj = [Sx, Sy, Sz]
Op_field = np.eye(d)
timeStep = 0.00
interactionConstants = [1] * m
dE = 1e-5

errors_tsu = []
errors_su = []


for i in range(iterations):
    tensors_tsu_next, weights_tsu_next = tsu.trivialsimpleUpdate(tensors_tsu,
                                                                 weights_tsu,
                                                                 smat,
                                                                 D_max)
    error = np.sum(np.abs(np.asarray(weights_tsu) - np.asarray(weights_tsu_next)))
    errors_tsu.append(error)
    if error < dw:
        print('The error is: {}'.format(error))
        tensors_tsu = tensors_tsu_next
        weights_tsu = weights_tsu_next
        break
    tensors_tsu = tensors_tsu_next
    weights_tsu = weights_tsu_next

# constructing the dual double-edge factor graph
graph = defg.defg()
graph = su.TNtoDEFGtransform(graph, tensors_tsu, weights_tsu, smat)
s = time.time()
graph.sumProduct(t_max, epsilon, dumping, printTime=1)
tot = time.time() - s
graph.calculateFactorsBeliefs()


tSU_1rdm = []
for i in range(n):
    tSU_1rdm.append(su.singleSiteRDM(i, tensors_tsu, weights_tsu, smat))

plt.figure()
plt.plot(range(len(errors_tsu)), errors_tsu)
plt.show()


# calculate two site RDMs
tSU_2rdm = []
for i in range(m):
    tSU_2rdm.append(tsu.doubleSiteRDM(i, tensors_tsu, weights_tsu, smat))

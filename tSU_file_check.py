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
tensors_su, weights_su = cp.deepcopy(tensors_tsu), cp.deepcopy(weights_tsu)
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





for i in range(iterations):
    tensors_su_next, weights_su_next = su.simpleUpdate(tensors_su,
                                                       weights_su,
                                                       timeStep,
                                                       interactionConstants,
                                                       0,
                                                       Opi,
                                                       Opj,
                                                       Op_field,
                                                       smat,
                                                       D_max,
                                                       'SU',
                                                       graph=None)

    error = np.sum(np.abs(np.asarray(weights_su) - np.asarray(weights_su_next)))
    errors_su.append(error)
    if error < dw:
        print('The error is: {}'.format(error))
        tensors_su = tensors_su_next
        weights_su = weights_su_next
        break
    tensors_su = tensors_su_next
    weights_su = weights_su_next

SU_1rdm = []
tSU_1rdm = []
ATD1 = 0
for i in range(n):
    SU_1rdm.append(su.singleSiteRDM(i, tensors_su, weights_su, smat))
    tSU_1rdm.append(su.singleSiteRDM(i, tensors_tsu, weights_tsu, smat))
    ATD1 += su.traceDistance(SU_1rdm[i], tSU_1rdm[i])
    print(i, su.traceDistance(SU_1rdm[i], tSU_1rdm[i]))
print('\nATD = {}\n'.format(ATD1))

plt.figure()
plt.plot(range(len(errors_su)), errors_su)
plt.plot(range(len(errors_tsu)), errors_tsu)
plt.legend(['SU', 'tSU'])
plt.show()


# calculate two site RDMs
SU_2rdm = []
tSU_2rdm = []
ATD2 = 0
for i in range(m):
    SU_2rdm.append(tsu.doubleSiteRDM(i, tensors_su, weights_su, smat))
    tSU_2rdm.append(tsu.doubleSiteRDM(i, tensors_tsu, weights_tsu, smat))
    ATD2 += tsu.traceDistance(SU_2rdm[i], tSU_2rdm[i])
    print(i, tsu.traceDistance(SU_2rdm[i], tSU_2rdm[i]))
print('\nATD = {}\n'.format(ATD2))
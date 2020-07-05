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
N, M = 4, 4
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
interactionConstants = [-1] * m
dE = 1e-5

errors_tsu = []
errors_su = []

'''
for i in range(iterations):
    tensors_tsu_next, weights_tsu_next = tsu.trivialsimpleUpdate(tensors_tsu, weights_tsu, smat, D_max)
    error = np.sum(np.abs(np.asarray(weights_tsu) - np.asarray(weights_tsu_next)))
    errors_tsu.append(error)
    if error < dw:
        print('The error is: {}'.format(error))
        tensors_tsu = tensors_tsu_next
        weights_tsu = weights_tsu_next
        break
    tensors_tsu = tensors_tsu_next
    weights_tsu = weights_tsu_next
'''
# constructing the dual double-edge factor graph
pre_graph = defg.defg()
pre_graph = su.TNtoDEFGtransform(pre_graph, tensors_su, weights_su, smat)
s = time.time()
pre_graph.sumProduct(t_max, epsilon, dumping, printTime=1)
pre_tot = time.time() - s
pre_graph.calculateFactorsBeliefs()



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

# constructing the dual double-edge factor graph
post_graph = defg.defg()
post_graph = su.TNtoDEFGtransform(post_graph, tensors_su, weights_su, smat)
s = time.time()
post_graph.sumProduct(t_max, epsilon, dumping, printTime=1)
post_tot = time.time() - s
post_graph.calculateFactorsBeliefs()

rho_SU = []
# RDMS using BP and SU
for i in range(n):
    rho_SU.append(su.singleSiteRDM(i, tensors_su, weights_su, smat))
rho_pre_graph = pre_graph.calculateRDMSfromFactorBeliefs()
rho_post_graph = pre_graph.calculateRDMSfromFactorBeliefs()


d_pre_post = 0
d_pre_su = 0
d_post_su = 0
for i in range(n):
    d_pre_post += su.traceDistance(rho_pre_graph[i], rho_pre_graph[i])
    d_pre_su += su.traceDistance(rho_pre_graph[i], rho_SU[i])
    d_post_su += su.traceDistance(rho_post_graph[i], rho_SU[i])

print('\nd(pre, post) = {}\nd(pre, su) = {}\nd(post, su) = {}'.format(d_pre_post / n, d_pre_su / n, d_post_su / n))

for _ in range(1):
    tensors_su_next, weights_su_next = su.simpleUpdate(tensors_su,
                                                       weights_su,
                                                       0.1,
                                                       interactionConstants,
                                                       0,
                                                       Opi,
                                                       Opj,
                                                       Op_field,
                                                       smat,
                                                       D_max,
                                                       'SU',
                                                       graph=None)
    tensors_su = tensors_su_next
    weights_su = weights_su_next

rho_next_SU = []
# RDMS using BP and SU
for i in range(n):
    rho_next_SU.append(su.singleSiteRDM(i, tensors_su_next, weights_su_next, smat))
d_post_su_next = 0
for i in range(n):
    d_post_su_next += su.traceDistance(rho_post_graph[i], rho_next_SU[i])

print('d(post, su-next) = {}'.format(d_post_su_next))





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

plt.figure()
#plt.plot(range(len(errors_tsu)), errors_tsu)
plt.plot(range(len(errors_su)), errors_su)
#plt.legend(['tSU', 'SU'])
plt.show()

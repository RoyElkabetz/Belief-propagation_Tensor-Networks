import numpy as np
import random
import time
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
color_list = ['salmon', 'limegreen', 'mediumturquoise', 'cornflowerblue', 'fuchsia', 'khaki']

import copy as cp
import pickle
import pandas as pd


import RandomPEPS as rpeps
import StructureMatrixGenerator as smg
import trivialSimpleUpdate as tsu
import DoubleEdgeFactorGraphs as defg
import SimpleUpdate as su
import bmpslib as bmps

# calculate the Bethe free energy
def Bethe_Free_Energy(defg, TN=None):
    if TN:
        tensors, weights, smat = TN
        tensors = su.absorbAllTensorNetWeights(tensors, weights, smat)
    factors = defg.factors
    nodes = defg.nodes

    factorBeliefs = defg.factorsBeliefs
    nodeBeliefs = defg.nodesBeliefs

    Bethe_energy_term = 0
    Bethe_entropy_term = 0

    # calculate the energy term (first term)
    for f in factors.keys():
        i = int(f[1:])
        if TN:
            tensor = tensors[i]
        else:
            tensor = factors[f][1]
        idx = list(range(len(tensor.shape)))
        idx_conj = list(range(len(tensor.shape) - 1, 2 * len(tensor.shape) - 1))
        idx_conj[0] = idx[0]
        idx_final = []
        for i in range(1, len(idx)):
            idx_final.append(idx[i])
            idx_final.append(idx_conj[i])
        factor = np.einsum(tensor, idx, np.conj(tensor), idx_conj, idx_final)
        fbelief = factorBeliefs[f]
        Bethe_energy_term += np.sum(fbelief * np.log10(fbelief / factor))

    # calculate the entropy term (second term)
    for n in nodes.keys():
        d_n = len(nodes[n][1])
        nbelief = nodeBeliefs[n]
        Bethe_entropy_term += (1 - d_n) * np.sum(nbelief * np.log10(nbelief))

    return Bethe_energy_term + Bethe_entropy_term






# SU and BP parameters
N, M = 4, 4                                                 # NxM PEPS
bc = 'open'                                                   # boundary conditions
dw = 1e-6                                                     # maximal error allowed between two-body RDMS
d = 2                                                       # tensor network physical bond dimension
bond_dimension = 2                                   # maximal virtual bond dimensions allowed for truncation
t_max = 100                                                   # maximal number of BP iterations
epsilon = 1e-6                                              # convergence criteria for BP messages (not used)
dumping = 0.2                                                  # BP messages dumping between [0, 1]
iterations = 100                                              # maximal number of tSU iterations
BPU_iterations = 100                                          # maximal number of BPU iterations
num_experiments = 10                                     # number of random experiments for each bond dimension
smat, _ = smg.finitePEPSobcStructureMatrixGenerator(N, M)     # generating the PEPS structure matrix
n, m = smat.shape
Bethe_values_pre_tsu = []
Bethe_values_post_tsu = []

# ITE parameters
Z = np.array([[1, 0], [0, -1]])
Y = np.array([[0, -1j], [1j, 0]])
X = np.array([[0, 1], [1, 0]])
Sz = 0.5 * Z
Sy = 0.5 * Y
Sx = 0.5 * X
Opi = [Sx, Sy, Sz]
Opj = [Sx, Sy, Sz]
Op_field = np.eye(d)
timeStep = [0.1, 0.01, 0.001]

for e in range(num_experiments):
    # draw some random PEPS Tensor Network
    tensors, weights = smg.randomTensornetGenerator(smat, d, bond_dimension)
    interactionConstants = -np.random.rand(m)
    print('experiment = ', e)

    BPU_graph = defg.defg()
    BPU_graph = su.TNtoDEFGtransform(BPU_graph, tensors, weights, smat)
    BPU_graph.sumProduct(t_max, epsilon, dumping, initializeMessages=1, printTime=0, RDMconvergence=0)

    for dt in timeStep:
        for i in range(BPU_iterations):
            if i % 40 == 0:
                print('i, dt = ', i, dt)
            weights_prev = cp.deepcopy(weights)
            tensors_next, weights_next = su.simpleUpdate(tensors,
                                                         weights,
                                                         dt,
                                                         interactionConstants,
                                                         0,
                                                         Opi,
                                                         Opj,
                                                         Op_field,
                                                         smat,
                                                         bond_dimension,
                                                         'BP',
                                                         graph=BPU_graph)
            BPU_graph.sumProduct(t_max,
                                 epsilon,
                                 dumping,
                                 initializeMessages=1,
                                 printTime=0,
                                 RDMconvergence=0)

            if np.sum(np.abs(np.asarray(weights_prev) - np.asarray(weights_next))) < dt * 1e-2:
                tensors = tensors_next
                weights = weights_next
                break
            tensors = tensors_next
            weights = weights_next
    ground_state_energy = su.energyPerSite(tensors,
                                           weights,
                                           smat,
                                           interactionConstants,
                                           0,
                                           Opi,
                                           Opj,
                                           Op_field)
    print('The ground state Energy (per site) is: {}'.format(np.round(ground_state_energy, 6)))

    # calculate the DEFG node and factor beliefs
    BPU_graph.calculateFactorsBeliefs()
    BPU_graph.calculateNodesBeliefs()
    bethe = Bethe_Free_Energy(BPU_graph)
    print(bethe)
    Bethe_values_pre_tsu.append(np.real(bethe))

    # run few iterations of trivial Simple Update algorithm in order to get the "quasi-canonical" PEPS representation
    # of the AFH ground-state
    errors = []
    for i in range(iterations):
        weights_prev = cp.deepcopy(weights)
        tensors_next, weights_next = tsu.trivialsimpleUpdate(tensors,
                                                             weights,
                                                             smat,
                                                             bond_dimension)
        error = np.sum(np.abs(np.asarray(weights) - np.asarray(weights_next)))
        errors.append(error)
        if error < dw:
            print('The final error is: {} in {} iterations'.format(error, i))
            tensors = tensors_next
            weights = weights_next
            break
        tensors = tensors_next
        weights = weights_next

    ground_state_energy = su.energyPerSite(tensors,
                                           weights,
                                           smat,
                                           interactionConstants,
                                           0,
                                           Opi,
                                           Opj,
                                           Op_field)
    print('The ground state Energy (per site) is: {}'.format(np.round(ground_state_energy, 6)))

    # save the fixed-point Tensor Net
    tensors_fixedPoint = cp.deepcopy(tensors)
    weights_fixedPoint = cp.deepcopy(weights)



    # calculate the DEFG node and factor beliefs
    bethe = Bethe_Free_Energy(BPU_graph, TN=[tensors, weights, smat])
    print(bethe)
    Bethe_values_post_tsu.append(np.real(bethe))

date = '18Jul2020_D_2'
name = 'Bethe_values_' + str(N) + 'x' + str(M) + '_FH_PEPS_' + date
#np.save(name, np.asarray(Bethe_values))

plt.figure()
plt.plot(list(range(1, num_experiments + 1)), Bethe_values_pre_tsu, 'o', color=mcd.CSS4_COLORS[color_list[0]])
plt.plot(list(range(1, num_experiments + 1)), Bethe_values_post_tsu, 'o', color=mcd.CSS4_COLORS[color_list[1]])
plt.plot(list(range(1, num_experiments + 1)), np.mean(np.asarray(Bethe_values_pre_tsu)) * np.ones((num_experiments, 1)), '--', color=mcd.CSS4_COLORS[color_list[0]])
plt.plot(list(range(1, num_experiments + 1)), np.mean(np.asarray(Bethe_values_post_tsu)) * np.ones((num_experiments, 1)), '--', color=mcd.CSS4_COLORS[color_list[1]])
plt.title('Bethe free energy of 4x4 random PEPS')
plt.xlabel('Experiment number')
plt.ylabel('Bethe free energy value')
plt.legend(['pre tSU VALUES', 'post tSU VALUES'])
plt.grid()
plt.show()

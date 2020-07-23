import numpy as np
import random
import time
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
color_list = ['salmon', 'limegreen', 'mediumturquoise', 'cornflowerblue', 'fuchsia', 'khaki']

import copy as cp
import pickle
import pandas as pd
from datetime import date
print(date.today())

import RandomPEPS as rpeps
import StructureMatrixGenerator as smg
import trivialSimpleUpdate as tsu
import DoubleEdgeFactorGraphs as defg
import SimpleUpdate as su
import bmpslib as bmps


def Bethe_Free_Energy(defg):
    # a function which calculate the Bethe free energy
    factors = defg.factors
    nodes = defg.nodes
    factorBeliefs = defg.factorsBeliefs
    nodeBeliefs = defg.nodesBeliefs
    Bethe_energy_term = 0
    Bethe_entropy_term = 0

    # calculate the energy term (first term)
    for f in factors.keys():
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
    return np.real(Bethe_energy_term), np.real(Bethe_entropy_term)






# BPU, tSU and BP parameters
N, M = 4, 4                                                   # NxM PEPS
bc = 'open'                                                   # boundary conditions
dw = 1e-2                                                     # maximal error allowed between two-body RDMS
d = 2                                                         # tensor network physical bond dimension
bond_dimension = 2                                            # maximal virtual bond dimensions allowed for truncation
t_max = 1000                                                  # maximal number of BP iterations
epsilon = 1e-10                                                # convergence criteria for BP messages (not used)
dumping = 0.                                                  # BP messages dumping between [0, 1]
BPU_iterations = 2000                                         # maximal number of BPU iterations
num_experiments = 10                                          # number of random experiments for each bond dimension
smat, _ = smg.finitePEPSobcStructureMatrixGenerator(N, M)     # generating the PEPS structure matrix
n, m = smat.shape
tSU_full_iters = 5
single_tSU_iterations = np.int(m * tSU_full_iters)            # maximal number of tSU iterations


# placeholders for Bethe free energy terms
Bethe_entropy_values = np.zeros((num_experiments, single_tSU_iterations + 1))
Bethe_energy_values = np.zeros((num_experiments, single_tSU_iterations + 1))


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


for e in range(num_experiments):
    # draw some random PEPS Tensor Network
    tensors, weights = smg.randomTensornetGenerator(smat, d, bond_dimension)
    # draw some random uniform(-1, 0) ATH couplings
    interactionConstants = -np.random.rand(m)
    #interactionConstants = [-1] * m

    # calculate the BP fixed point messages and the two-body RDMs
    BPU_graph = defg.defg()
    BPU_graph = su.TNtoDEFGtransform(BPU_graph, tensors, weights, smat)
    BPU_graph.sumProduct(t_max, epsilon, dumping, initializeMessages=1, printTime=0, RDMconvergence=0)
    BP_rdm = []
    for j in range(m):
        BP_rdm.append(tsu.BPdoubleSiteRDM1(j,
                                           cp.deepcopy(tensors),
                                           cp.deepcopy(weights),
                                           smat,
                                           cp.deepcopy(BPU_graph.messages_n2f)))

    # run BPU until convergence criteria is met
    # the convergence criteria is taken to be an upper bound over the Averaged Trace Distance (ATD) Between
    # two consecutive BPU iterations two body RDMs --- ATD < 1e-6
    dt = 0.1
    counter = 0
    while dw * dt > 1e-6:
        for i in range(BPU_iterations):
            counter += 1
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

            ATD = 0
            BP_rdm_next = []
            for j in range(m):
                BP_rdm_next.append(tsu.BPdoubleSiteRDM1(j,
                                                        cp.deepcopy(tensors_next),
                                                        cp.deepcopy(weights_next),
                                                        smat,
                                                        cp.deepcopy(BPU_graph.messages_n2f)))

                ATD += su.traceDistance(BP_rdm[j], BP_rdm_next[j])
            BP_rdm = BP_rdm_next
            ATD /= m
            if i % 100 == 0:
                print('experiment #, i, dt, ATD = {}, {}, {}, {}'.format(e, i, dt, ATD))
            if ATD < dw * dt:
                dt /= 2
                BP_rdm = BP_rdm_next
                tensors = tensors_next
                weights = weights_next
                break
            tensors = tensors_next
            weights = weights_next
    print('BPU converged in {} iterations'.format(counter))
    ground_state_energy = su.energyPerSite(tensors,
                                           weights,
                                           smat,
                                           interactionConstants,
                                           0,
                                           Opi,
                                           Opj,
                                           Op_field)
    print('The ground state Energy (per site) is: {}'.format(np.round(ground_state_energy, 6)))

    # generate a random list of edges for single tSU implementation
    edgesOrder = np.tile(np.arange(m), tSU_full_iters)
    np.random.shuffle(edgesOrder)

    # calculate the Bethe free energy and run tSU
    for ii, edge in enumerate(edgesOrder):

        # calculate the DEFG node and factor beliefs
        BPU_graph.calculateFactorsBeliefs()
        BPU_graph.calculateNodesBeliefs()

        # Bethe free energy calculation
        bethe_energy_term, bethe_entropy_term = Bethe_Free_Energy(BPU_graph)
        #print('\nBethe energy term {}'.format(bethe_energy_term))
        #print('Bethe entropy term {}'.format(bethe_entropy_term))
        Bethe_energy_values[e, ii] = bethe_energy_term
        Bethe_entropy_values[e, ii] = bethe_entropy_term

        # run a single edge iteration of trivial BPU algorithm
        # (which is the same as tSU but also with DEFG updating of factors)
        tensors_next, weights_next = su.simpleUpdate(tensors,
                                                     weights,
                                                     0,
                                                     interactionConstants,
                                                     0,
                                                     Opi,
                                                     Opj,
                                                     Op_field,
                                                     smat,
                                                     bond_dimension,
                                                     'BP',
                                                     graph=BPU_graph,
                                                     singleEdge=edge)
        BPU_graph.sumProduct(t_max,
                             epsilon,
                             dumping,
                             initializeMessages=1,
                             printTime=1,
                             RDMconvergence=0)
        tensors = tensors_next
        weights = weights_next

    # calculate the DEFG node and factor beliefs
    BPU_graph.calculateFactorsBeliefs()
    BPU_graph.calculateNodesBeliefs()

    # Bethe free energy calculation
    bethe_energy_term, bethe_entropy_term = Bethe_Free_Energy(BPU_graph)
    #print('\nBethe energy term {}'.format(bethe_energy_term))
    #print('Bethe entropy term {}'.format(bethe_entropy_term))
    Bethe_energy_values[e, single_tSU_iterations] = bethe_energy_term
    Bethe_entropy_values[e, single_tSU_iterations] = bethe_entropy_term




name1 = 'Bethe_first_term_' + str(num_experiments) + '_experiments_' + str(N) + 'x' + str(M) + '_AFH_PEPS_with_random_couplings_' + str(date.today())
name2 = 'Bethe_second_term_' + str(num_experiments) + '_experiments_' + str(N) + 'x' + str(M) + '_AFH_PEPS_with_random_couplings_' + str(date.today())
#name1 = 'Bethe_first_term_' + str(num_experiments) + '_experiments_' + str(N) + 'x' + str(M) + '_random_PEPS_' + str(date.today())
#name2 = 'Bethe_second_term_' + str(num_experiments) + '_experiments_' + str(N) + 'x' + str(M) + '_random_PEPS_' + str(date.today())
#name1 = 'Bethe_first_term_' + str(num_experiments) + '_experiments_' + str(N) + 'x' + str(M) + '_AFH_PEPS_ground_state_' + str(date.today())
#name2 = 'Bethe_second_term_' + str(num_experiments) + '_experiments_' + str(N) + 'x' + str(M) + '_AFH_PEPS_ground_state_' + str(date.today())
np.save(name1, np.asarray(Bethe_energy_values))
np.save(name2, np.asarray(Bethe_entropy_values))


equation = r'$F_{Bethe}[q]=\sum_{\alpha}\sum_{\mathbf{z}_{\alpha}}q_{\alpha}(\mathbf{z}_{\alpha})\log \left(\frac{q_{\alpha}(\mathbf{z}_{\alpha})}{f_{\alpha}(\mathbf{z}_{\alpha})} \right)  +  \sum_{i}\sum_{z_i} (1-d_i) q_i(z_i) \log q_i(z_i)$'
left_term = r'$\sum_{\alpha}\sum_{\mathbf{z}_{\alpha}}q_{\alpha}(\mathbf{z}_{\alpha})\log \left(\frac{q_{\alpha}(\mathbf{z}_{\alpha})}{f_{\alpha}(\mathbf{z}_{\alpha})} \right)$'
right_term = r'$\sum_{i}\sum_{z_i} (1-d_i) q_i(z_i) \log q_i(z_i)$'

#title = str(N) + 'x' + str(M) + ' AFH PEPS with random couplings\n' + 'Bethe free energy left term as a function of the number of single tSU steps\n'
'''
color_list1 = list(mcd.CSS4_COLORS.keys())
plt.figure()
for i in range(num_experiments):
    plot_name = 'exp-' + str(i)
    plt.plot(list(range(single_tSU_iterations)),
             np.diff(Bethe_energy_values[i, :]),
             label=plot_name)
plt.title(title + equation, fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('# of single tSU iterations', fontsize=12)
plt.ylabel('diff(' + left_term + ')', fontsize=12)
plt.legend()
plt.grid()
plt.show()
#color=mcd.CSS4_COLORS[color_list1[2 * i]],

title = str(N) + 'x' + str(M) + ' AFH PEPS with random couplings\n' + 'Bethe free energy right term as a function of the number of single tSU steps\n'
plt.figure()
for i in range(num_experiments):
    plot_name = 'exp-' + str(i)
    plt.plot(list(range(single_tSU_iterations)),
             np.diff(Bethe_entropy_values[i, :]),
             label=plot_name)
plt.title(title + equation, fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('# of single tSU iterations', fontsize=12)
plt.ylabel('diff(' + right_term + ')', fontsize=12)
plt.legend()
plt.grid()
plt.show()


title = str(N) + 'x' + str(M) + ' AFH PEPS with random couplings\n' + 'Bethe free energy as a function of the number of single tSU steps\n'
plt.figure()
for i in range(num_experiments):
    plot_name = 'exp-' + str(i)
    plt.plot(list(range(single_tSU_iterations + 1)),
             Bethe_entropy_values[i, :] + Bethe_energy_values[i, :],
             label=plot_name)
plt.title(title + equation, fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('# of single tSU iterations', fontsize=12)
plt.ylabel(r'$F_{Bethe}$', fontsize=12)
plt.legend()
plt.grid()
plt.show()
'''
title = str(N) + 'x' + str(M) + ' AFH PEPS ground-state with random couplings \n' + 'Bethe free energy as a function of the number of single tSU steps\n'
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
for i in range(num_experiments):
    plot_name = 'exp-' + str(i)
    ax1.plot(list(range(single_tSU_iterations)),
             np.diff(Bethe_energy_values[i, :]),
             label=plot_name)
    ax2.plot(list(range(single_tSU_iterations)),
             np.diff(Bethe_entropy_values[i, :]),
             label=plot_name)
    ax3.plot(list(range(single_tSU_iterations + 1)),
             Bethe_energy_values[i, :] + Bethe_entropy_values[i, :],
             label=plot_name)
ax1.set_title(title + equation)
ax3.set_xlabel('# of single tSU iterations', fontsize=10)
ax1.set_ylabel('diff(' + left_term + ')', fontsize=8)
ax2.set_ylabel('diff(' + right_term + ')', fontsize=8)
ax3.set_ylabel(r'$F_{Bethe}$', fontsize=12)
ax1.grid()
ax2.grid()
#ax3.legend()
#plt.tight_layout()
plt.show()



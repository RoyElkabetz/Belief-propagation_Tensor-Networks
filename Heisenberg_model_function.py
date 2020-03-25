
import numpy as np
import copy as cp
import time
import pickle
import ncon
from scipy import linalg
import matplotlib.pyplot as plt


import SimpleUpdate as BP
import ncon_lists_generator as nlg
import DoubleEdgeFactorGraphs as defg
import StructureMatrixGenerator as tnf
import bmpslib as bmps



# --------------------------------------------------------------------------------------------- #
#   Heisenberg_PEPS_BP                                                                          #
#                                                                                               #
#   Run the BP truncation algorithm with the given initial conditions                           #
#   over a spin 1/2 N x M AFH PEPS.                                                             #
#                                                                                               #
#   Parameters:                                                                                 #
#   N           - PEPS # of rows                                                                #
#   M           - PEPS # of columns                                                             #
#   Jk          - list of interactions coefficients (for every edge)                            #
#   dE          - energy stopping criteria                                                      #
#   D_max       - maximal bond dimension                                                        #
#   t_max       - maximal BP iterations                                                         #
#   epsilon     - BP messages stoping criteria                                                  #
#   dumping     - BP messages dumping:   m(t + 1) = (1 - dumping) * m(t + 1) + dumping * m(t)   #
#   bc          - PEPS boundary condition ( 'periodic' / 'open' )                               #
#   t_list      - list of dt for ITE                                                            #
#   iterations  - # of maximal iterations for each dt in t_lis                                  #
#   TN          - if not None TN = [tensor_list, bond_vectors_list]                             #
#                                                                                               #
#   Return:                                                                                     #
#   TT          - tensors list                                                                  #
#   LL          - bond vectors list                                                             #
#   BP_energy   - list of PEPS energy per site calculated at each iteration                     #
# --------------------------------------------------------------------------------------------- #


def Heisenberg_PEPS_BP(N, M, Jk, dE, D_max, t_max, epsilon, dumping, bc, t_list, iterations, TN=None):

    # Tensor Network parameters
    d = D_max  # virtual bond dimension
    p = 2  # physical bond dimension
    h = 0  # Hamiltonian: magnetic field coeffs
    BP_energy = []
    print('running Heisenberg_PEPS_BP: D = ', D_max)
    # pauli matrices
    pauli_z = np.array([[1, 0], [0, -1]])
    pauli_y = np.array([[0, -1j], [1j, 0]])
    pauli_x = np.array([[0, 1], [1, 0]])
    sz = 0.5 * pauli_z
    sy = 0.5 * pauli_y
    sx = 0.5 * pauli_x
    Opi = [sx, sy, sz]
    Opj = [sx, sy, sz]
    Op_field = np.eye(p)


    # generating the PEPS structure matrix


    if bc == 'open':
        smat, imat = tnf.finitePEPSobcStructureMatrixGenerator(N, M)
    if bc == 'periodic':
        smat, imat = tnf.squareFinitePEPSpbcStructureMatrixGenerator(N * M)
    '''
    smat = np.array([[1, 2, 3, 4, 0, 0, 0, 0],
                     [1, 2, 0, 0, 3, 4, 0, 0],
                     [0, 0, 1, 2, 0, 0, 3, 4],
                     [0, 0, 0, 0, 1, 2, 3, 4]])
    '''
    n, m = smat.shape


    # generating tensors and bond vectors

    if TN:
        TT, LL = TN
    else:
        if bc == 'open':
            TT, LL = tnf.randomTensornetGenerator(smat, p, d)
        if bc == 'periodic':
            TT, LL = tnf.randomTensornetGenerator(smat, p, d)


    # constructing the dual double-edge factor graph

    graph = defg.Graph()
    graph = BP.PEPStoDEnFG_transform(graph, TT, LL, smat)
    graph.sum_product(t_max, epsilon, dumping)


    # iterating the BP truncation algorithm

    for dt in t_list:
        for j in range(iterations):
            #print('BP_N, D max, dt, j = ', N, D_max, dt, j)
            TT1, LL1 = BP.simpleUpdate(TT, LL, dt, Jk, h, Opi, Opj, Op_field, smat, D_max, 'BP', graph)
            graph.sum_product(t_max, epsilon, dumping, 'init_with_old_messages')
            energy1 = BP.BP_energy_per_site_using_factor_belief(graph, smat, Jk, h, Opi, Opj, Op_field)
            TT2, LL2 = BP.simpleUpdate(TT1, LL1, dt, Jk, h, Opi, Opj, Op_field, smat, D_max, 'BP', graph)
            graph.sum_product(t_max, epsilon, dumping, 'init_with_old_messages')
            energy2 = BP.BP_energy_per_site_using_factor_belief(graph, smat, Jk, h, Opi, Opj, Op_field)

            BP_energy.append(np.real(energy1))
            BP_energy.append(np.real(energy2))

            #print(energy1, energy2)

            if np.abs(energy1 - energy2) < dE * dt:
                TT = TT2
                LL = LL2
                break
            else:
                TT = TT2
                LL = LL2

    return [graph, TT, LL, BP_energy]









# --------------------------------------------------------------------------------------------- #
#   Heisenberg_PEPS_gPEPS                                                                       #
#                                                                                               #
#   Run the gPEPS algorithm with the given initial conditions                                   #
#   over a spin 1/2 N x M AFH PEPS.                                                             #
#                                                                                               #
#   Parameters:                                                                                 #
#   N           - PEPS # of rows                                                                #
#   M           - PEPS # of columns                                                             #
#   Jk          - list of interactions coefficients (for every edge)                            #
#   dE          - energy stopping criteria                                                      #
#   D_max       - maximal bond dimension                                                        #
#   bc          - PEPS boundary condition ( 'periodic' / 'open' )                               #
#   t_list      - list of dt for ITE                                                            #
#   iterations  - # of maximal iterations for each dt in t_lis                                  #
#                                                                                               #
#   Return:                                                                                     #
#   TT          - tensors list                                                                  #
#   LL          - bond vectors list                                                             #
#   gPEPS_energy   - list of PEPS energy per site calculated at each iteration                  #
# --------------------------------------------------------------------------------------------- #


def Heisenberg_PEPS_gPEPS(N, M, Jk, dE, D_max, bc, t_list, iterations):

    # Tensor Network parameters

    d = 2  # virtual bond dimension
    p = 2  # physical bond dimension
    h = 0  # Hamiltonian: magnetic field coeffs
    gPEPS_energy = []
    print('running Heisenberg_PEPS_SU: D = ', D_max)

    pauli_z = np.array([[1, 0], [0, -1]])
    pauli_y = np.array([[0, -1j], [1j, 0]])
    pauli_x = np.array([[0, 1], [1, 0]])
    sz = 0.5 * pauli_z
    sy = 0.5 * pauli_y
    sx = 0.5 * pauli_x
    Opi = [sx, sy, sz]
    Opj = [sx, sy, sz]
    Op_field = np.eye(p)


    # generating the PEPS structure matrix

    if bc == 'open':
        smat, imat = tnf.finitePEPSobcStructureMatrixGenerator(N, M)
    if bc == 'periodic':
        smat, imat = tnf.squareFinitePEPSpbcStructureMatrixGenerator(N * M)
    '''
    smat = np.array([[1, 2, 3, 4, 0, 0, 0, 0],
                     [1, 2, 0, 0, 3, 4, 0, 0],
                     [0, 0, 1, 2, 0, 0, 3, 4],
                     [0, 0, 0, 0, 1, 2, 3, 4]])
    '''
    n, m = smat.shape


    # generating tensors and bond vectors

    if bc == 'open':
        TT, LL = tnf.randomTensornetGenerator(smat, p, d)
    if bc == 'periodic':
        TT, LL = tnf.randomTensornetGenerator(smat, p, d)


    # iterating the gPEPS algorithm

    for dt in t_list:
        for j in range(iterations):
            #print('N, D max, dt, j = ', N * M, D_max, dt, j)
            TT1, LL1 = BP.simpleUpdate(TT, LL, dt, Jk, h, Opi, Opj, Op_field, smat, D_max, 'gPEPS')
            TT2, LL2 = BP.simpleUpdate(TT1, LL1, dt, Jk, h, Opi, Opj, Op_field, smat, D_max, 'gPEPS')
            energy1 = BP.energy_per_site(TT1, LL1, smat, Jk, h, Opi, Opj, Op_field)
            energy2 = BP.energy_per_site(TT2, LL2, smat, Jk, h, Opi, Opj, Op_field)

            gPEPS_energy.append(energy1)
            gPEPS_energy.append(energy2)

            #print(energy1, energy2)

            if np.abs(energy1 - energy2) < dE * dt:
                TT = TT2
                LL = LL2
                break
            else:
                TT = TT2
                LL = LL2

    return [TT, LL, gPEPS_energy]




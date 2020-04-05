
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



#################################################################################################
#   RandomPEPS_BP                                                                               #
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
#################################################################################################


def RandomPEPS_BP(N, M, Jk, dE, D_max, t_max, epsilon, dumping, bc, t_list, iterations, TN=None):

    # Tensor Network parameters
    D = D_max  # virtual bond dimension
    d = 2  # physical bond dimension
    h = 0  # Hamiltonian: magnetic field coeff

    Op = np.eye(d) / np.sqrt(3)
    Opi = [Op, Op, Op]
    Opj = [Op, Op, Op]
    Op_field = np.eye(d)

    # generating the PEPS structure matrix
    if bc == 'open':
        smat, imat = tnf.finitePEPSobcStructureMatrixGenerator(N, M)
    if bc == 'periodic':
        smat, imat = tnf.squareFinitePEPSpbcStructureMatrixGenerator(N * M)

    # generating tensors and bond vectors
    if TN:
        TT, LL = TN
    else:
        if bc == 'open':
            TT, LL = tnf.randomTensornetGenerator(smat, d, D)
        if bc == 'periodic':
            TT, LL = tnf.randomTensornetGenerator(smat, d, D)

    # constructing the dual double-edge factor graph
    graph = defg.defg()
    graph = BP.TNtoDEFGtransform(graph, TT, LL, smat)
    s = time.time()
    graph.sumProduct(t_max, epsilon, dumping)
    tot = time.time() - s
    graph.calculateFactorsBeliefs()
    return graph, tot


#################################################################################################
#   RandomPEPS_Simple_Update                                                                    #
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
#################################################################################################


def RandomPEPS_SU(N, M, Jk, dE, D_max, bc, t_list, iterations):

    # Tensor Network parameters
    D = D_max  # virtual bond dimension
    d = 2  # physical bond dimension
    h = 0  # Hamiltonian: magnetic field coeffs
    Op = np.eye(d) / np.sqrt(3)
    Opi = [Op, Op, Op]
    Opj = [Op, Op, Op]
    Op_field = np.eye(d)

    # generating the PEPS structure matrix
    if bc == 'open':
        smat, imat = tnf.finitePEPSobcStructureMatrixGenerator(N, M)
    if bc == 'periodic':
        smat, imat = tnf.squareFinitePEPSpbcStructureMatrixGenerator(N * M)

    # generating tensors and bond vectors
    if bc == 'open':
        TT, LL = tnf.randomTensornetGenerator(smat, d, D)
    if bc == 'periodic':
        TT, LL = tnf.randomTensornetGenerator(smat, d, D)
    TT0, LL0 = cp.deepcopy(TT), cp.deepcopy(LL)

    # iterating the gPEPS algorithm
    s = time.time()
    for dt in t_list:
        for j in range(iterations):
            #print('N, D max, dt, j = ', N * M, D_max, dt, j)
            TT1, LL1 = BP.simpleUpdate(TT, LL, dt, Jk, h, Opi, Opj, Op_field, smat, D_max, 'SU')
            TT2, LL2 = BP.simpleUpdate(TT1, LL1, dt, Jk, h, Opi, Opj, Op_field, smat, D_max, 'SU')
            error = 0
            for i in range(len(LL1)):
                error += np.sum(np.abs(LL1[i] - LL2[i]))
            #print('error = ', error)
            if error < dE * dt:
                TT = TT2
                LL = LL2
                break
            else:
                TT = TT2
                LL = LL2
        if error < dE * dt:
            break
    tot = time.time() - s
    return [TT0, LL0, TT, LL], tot




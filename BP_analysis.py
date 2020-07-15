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

#####################################################################################################################
#                                                                                                                   #
#                                    TRIVIAL SIMPLE UPDATE (TSU) ON RANDOM PEPS                                     #
#                                                                                                                   #
#####################################################################################################################
np.random.seed(1)

# tSU parameters
N, M = 5, 5
bc = 'open'
dw = 1e-17
D_max = 2
t_max = 100
epsilon = 1e-10
dumping = 0.
iterations = 10
d = 2
smat, _ = smg.finitePEPSobcStructureMatrixGenerator(N, M)
tensors, weights = smg.randomTensornetGenerator(smat, d, D_max)
n, m = smat.shape


# constructing the dual double-edge factor graph and run a parallel BP
graph1 = defg.defg()
graph1 = su.TNtoDEFGtransform(graph1, cp.deepcopy(tensors), cp.deepcopy(weights), smat)
graph1.sumProduct(t_max, epsilon, dumping, initializeMessages=1, printTime=1, RDMconvergence=0, scheduling='series')

BP_par_rdm = []
for j in range(m):
        BP_par_rdm.append(tsu.BPdoubleSiteRDM1(j,
                                           cp.deepcopy(tensors),
                                           cp.deepcopy(weights),
                                           smat,
                                           cp.deepcopy(graph1.messages_n2f)))
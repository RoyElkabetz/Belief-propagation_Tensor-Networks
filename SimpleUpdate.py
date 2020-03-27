
import numpy as np
import ncon as ncon
import copy as cp
from scipy import linalg

import StructureMatrixGenerator as tnf
import ncon_lists_generator as nlg




def simpleUpdate(tensors,
                 weights,
                 timeStep,
                 interactionConst,
                 fieldConst,
                 iOperators,
                 jOperators,
                 fieldOperators,
                 smat,
                 Dmax,
                 type,
                 graph=None):
    """
    The Simple Update algorithm implementation on a general finite tensor network specified by a structure matrix
    :param tensors: list of tensors in the tensor network [T1, T2, T3, ..., Tn]
    :param weights: list of lambda weights [L1, L2, ..., Lm]
    :param timeStep: Imaginary Time Evolution (ITE) time step
    :param interactionConst: J_{ij} constants in the Hamiltonian
    :param fieldConst: field constant in the Hamiltonian
    :param iOperators: the operators associated with the i^th tensor in the Hamiltonian
    :param jOperators: the operators associated with the j^th tensor in the Hamiltonian
    :param fieldOperators: the operators associated with the field term in the Hamiltonian
    :param smat: tensor network structure matrix
    :param Dmax: maximal bond dimension
    :param type: type of algorithm to use 'BP' or 'SU'
    :param graph: the tensor network dual double-edge factor graph
    :return: updated tensors list and weights list
    """
    tensors = cp.deepcopy(tensors)
    weights = cp.deepcopy(weights)
    n, m = np.shape(smat)
    for Ek in range(m):
        lambda_k = weights[Ek]

        # Find tensors Ti, Tj and their corresponding indices connected along edge Ek.
        Ti, Tj = getTensors(Ek, tensors, smat)

        # collect edges and remove the Ek edge from both lists
        iEdgesNidx, jEdgesNidx = getTensorsEdges(Ek, smat)

        # absorb environment (lambda weights) into tensors
        Ti[0] = absorbWeights(Ti[0], iEdgesNidx, weights)
        Tj[0] = absorbWeights(Tj[0], jEdgesNidx, weights)

        # permuting the indices associated with edge Ek tensors Ti, Tj with their 1st index
        Ti = indexPermute(Ti)
        Tj = indexPermute(Tj)

        # Group all virtual indices Em!=Ek to form Pl, Pr MPS tensors
        Pl = rankNrank3(Ti[0])
        Pr = rankNrank3(Tj[0])

        # SVD decomposing of Pl, Pr to obtain Q1, R and Q2, L sub-tensors, respectively
        R, sr, Q1 = truncationSVD(Pl, [0, 1], [2], keepS='yes')
        L, sl, Q2 = truncationSVD(Pr, [0, 1], [2], keepS='yes')
        R = R.dot(np.diag(sr))
        L = L.dot(np.diag(sl))

        # RQ decomposition of Pl, Pr to obtain R, Q1 and L, Q2 sub-tensors, respectively (needs fixing)
        #R, Q1 = linalg.rq(np.reshape(Pl, [Pl.shape[0] * Pl.shape[1], Pl.shape[2]]))
        #L, Q2 = linalg.rq(np.reshape(Pr, [Pr.shape[0] * Pr.shape[1], Pr.shape[2]]))

        # reshaping R and L into rank 3 tensors with shape (physical_dim, Ek_dim, Q(1/2).shape[0])
        i_physical_dim = Ti[0].shape[0]
        j_physical_dim = Tj[0].shape[0]
        R = rank2rank3(R, i_physical_dim)  # (i, Ek, Q1) (following the dimensions)
        L = rank2rank3(L, j_physical_dim)  # (j, Ek, Q2)

        # Contract the ITE gate with R, L, and lambda_k to form theta tensor.
        theta = imaginaryTimeEvolution(R,
                                       L,
                                       lambda_k,
                                       Ek,
                                       timeStep,
                                       interactionConst,
                                       fieldConst,
                                       iOperators,
                                       jOperators,
                                       fieldOperators)  # (Q1, i', j', Q2)

        # Obtain R', L', lambda'_k tensors by applying an SVD to theta
        if type == 'SU':
            R_tild, lambda_k_tild, L_tild = truncationSVD(theta, [0, 1], [2, 3], keepS='yes', maxEigenvalNumber=Dmax) # with truncation
        if type == 'BP':
            R_tild, lambda_k_tild, L_tild = truncationSVD(theta, [0, 1], [2, 3], keepS='yes') # without truncation
        # (Q1 * i', D') # (D', D') # (D', j' * Q2)

        # reshaping R_tild and L_tild back to rank 3 tensor
        R_tild = np.reshape(R_tild, (Q1.shape[0], i_physical_dim, R_tild.shape[1]))  # (Q1, i', D')
        R_tild = np.transpose(R_tild, [1, 2, 0])  # (i', D', Q1)
        L_tild = np.reshape(L_tild, (L_tild.shape[0], j_physical_dim, Q2.shape[0]))  # (D', j', Q2)
        L_tild = np.transpose(L_tild, [1, 0, 2])  # (j', D', Q2)

        # Glue back the R', L', sub-tensors to Q1, Q2, respectively, to form updated tensors P'l, P'r.
        Pl_prime = np.einsum('ijk,kl->ijl', R_tild, Q1)
        Pr_prime = np.einsum('ijk,kl->ijl', L_tild, Q2)

        # Reshape back the P`l, P`r to the original rank-(z + 1) tensors Ti, Tj
        Ti_new_shape = list(Ti[0].shape)
        Ti_new_shape[1] = len(lambda_k_tild)
        Tj_new_shape = list(Tj[0].shape)
        Tj_new_shape[1] = len(lambda_k_tild)
        Ti[0] = rank3rankN(Pl_prime, Ti_new_shape)
        Tj[0] = rank3rankN(Pr_prime, Tj_new_shape)

        # permuting back the legs of Ti and Tj
        Ti = indexPermute(Ti)
        Tj = indexPermute(Tj)

        # Remove bond matrices lambda_m from virtual legs m != Ek to obtain the updated tensors Ti~, Tj~.
        Ti[0] = absorbInverseWeights(Ti[0], iEdgesNidx, weights)
        Tj[0] = absorbInverseWeights(Tj[0], jEdgesNidx, weights)

        # Normalize and save new Ti Tj and lambda_k
        tensors[Ti[1][0]] = Ti[0] / tensorNorm(Ti[0])
        tensors[Tj[1][0]] = Tj[0] / tensorNorm(Tj[0])
        weights[Ek] = lambda_k_tild / np.sum(lambda_k_tild)

        # single edge BP update (uncomment for single edge BP implemintation)
        if type == 'BP':
            tensors, weights = BPupdate_single_edge(tensors, weights, smat, Dmax, Ek, graph)

    return tensors, weights

########################################################################################################################
#                                        Simple Update auxiliary functions                                             #
########################################################################################################################

def getTensors(edge, tensors, smat):
    """
    Given an edge collect neighboring tensors.
    :param edge: edge number {0, 1, ..., m-1}.
    :param tensors: list of tensors.
    :param smat: structure matrix (n x m).
    :return: two lists of Ti Tj tensors, [tensor, [#, 'tensor_number'], [#, 'tensor_index_along_edge']]
    """
    tensorNumber = np.nonzero(smat[:, edge])[0]
    tensorIndexAlongEdge = smat[tensorNumber, edge]
    Ti = [tensors[tensorNumber[0]],
          [tensorNumber[0], 'tensor_number'],
          [tensorIndexAlongEdge[0], 'tensor_index_along_edge']
          ]
    Tj = [tensors[tensorNumber[1]],
          [tensorNumber[1], 'tensor_number'],
          [tensorIndexAlongEdge[1], 'tensor_index_along_edge']
          ]
    return Ti, Tj


def getConjTensors(edge, tensors, smat):
    """
    Given an edge collect neighboring tensors.
    :param edge: edge number {0, 1, ..., m-1}.
    :param tensors: list of tensors.
    :param smat: structure matrix (n x m).
    :return: two lists of Ti Tj conjugate tensors, [tensor, [#, 'tensor_number'], [#, 'tensor_index_along_edge']]
    """
    tensorNumber = np.nonzero(smat[:, edge])[0]
    tensorIndexAlongEdge = smat[tensorNumber, edge]
    Ti = [np.conj(tensors[tensorNumber[0]]),
          [tensorNumber[0], 'tensor_number'],
          [tensorIndexAlongEdge[0], 'tensor_index_along_edge']
          ]
    Tj = [np.conj(tensors[tensorNumber[1]]),
          [tensorNumber[1], 'tensor_number'],
          [tensorIndexAlongEdge[1], 'tensor_index_along_edge']
          ]
    return Ti, Tj


def getTensorsEdges(edge, smat):
    """
    Given an edge, collect neighboring tensors edges and indices
    :param edge: edge number {0, 1, ..., m-1}.
    :param smat: structure matrix (n x m).
    :return: two lists of Ti, Tj edges and associated indices with 'edge' and its index removed.
    """
    tensorNumber = np.nonzero(smat[:, edge])[0]
    iEdgesNidx = [list(np.nonzero(smat[tensorNumber[0], :])[0]),
                  list(smat[tensorNumber[0], np.nonzero(smat[tensorNumber[0], :])[0]])
                  ]  # [edges, indices]
    jEdgesNidx = [list(np.nonzero(smat[tensorNumber[1], :])[0]),
                  list(smat[tensorNumber[1], np.nonzero(smat[tensorNumber[1], :])[0]])
                  ]  # [edges, indices]
    # remove 'edge' and its associated index from both i, j lists.
    iEdgesNidx[0].remove(edge)
    iEdgesNidx[1].remove(smat[tensorNumber[0], edge])
    jEdgesNidx[0].remove(edge)
    jEdgesNidx[1].remove(smat[tensorNumber[1], edge])
    return iEdgesNidx, jEdgesNidx


def getEdges(tensorIndex, smat):
    """
    Given an index of a tensor, return all of its edges and associated indices.
    :param tensorIndex: the tensor index in the structure matrix
    :param smat: structure matrix
    :return: list of two lists [[edges], [indices]].
    """
    edges = np.nonzero(smat[tensorIndex, :])[0]
    indices = smat[tensorIndex, edges]
    return [edges, indices]


def getAllTensorsEdges(edge, smat):
    """
    Given an edge, collect neighboring tensors edges and indices
    :param edge: edge number {0, 1, ..., m-1}.
    :param smat: structure matrix (n x m).
    :return: two lists of Ti, Tj edges and associated indices.
    """
    tensorNumber = np.nonzero(smat[:, edge])[0]
    iEdgesNidx = [list(np.nonzero(smat[tensorNumber[0], :])[0]),
                  list(smat[tensorNumber[0], np.nonzero(smat[tensorNumber[0], :])[0]])
                  ]  # [edges, indices]
    jEdgesNidx = [list(np.nonzero(smat[tensorNumber[1], :])[0]),
                  list(smat[tensorNumber[1], np.nonzero(smat[tensorNumber[1], :])[0]])
                  ]  # [edges, indices]
    return iEdgesNidx, jEdgesNidx


def absorbWeights(tensor, edgesNidx, weights):
    """
    Absorb neighboring lambda weights into tensor.
    :param tensor: tensor
    :param edgesNidx: list of two lists [[edges], [indices]].
    :param weights: list of lambda weights.
    :return: the new tensor list [new_tensor, [#, 'tensor_number'], [#, 'tensor_index_along_edge']]
    """
    for i in range(len(edgesNidx[0])):
        tensor = np.einsum(tensor, list(range(len(tensor.shape))), weights[int(edgesNidx[0][i])], [int(edgesNidx[1][i])], list(range(len(tensor.shape))))
    return tensor


def absorbSqrtWeights(tensor, edgesNidx, weights):
    """
    Absorb square root of neighboring lambda weights into tensor.
    :param tensor: tensor
    :param edgesNidx: list of two lists [[edges], [indices]].
    :param weights: list of lambda weights.
    :return: the new tensor list [new_tensor, [#, 'tensor_number'], [#, 'tensor_index_along_edge']]
    """
    for i in range(len(edgesNidx[0])):
        tensor = np.einsum(tensor, list(range(len(tensor.shape))), np.sqrt(weights[int(edgesNidx[0][i])]),
                              [int(edgesNidx[1][i])], list(range(len(tensor.shape))))
    return tensor


def absorbWeights_twoSiteExpectationWithRectangularEnvironment(tensor, edgesNidx, weights, edgesINenv, edgesOUTenv):
    """
    Given a tensor and two lists of edges inside and on the boundary (outside) of rectangular environment
    of two site expectation, this auxilary function absorb the tensor neighboring weights according to edges environment
    lists. If edge is inside the rectangular environment, then its 'sqrt(lambda weight)' is absorbed. If edge is
    on the boundary (outside) of the rectangular environment, then its 'lambda weight' is absorbed.
    :param tensor: tensor inside rectangular environment
    :param edgesNidx: list of two lists [[edges], [indices]]
    :param weights: list of lambda weights
    :param edgesINenv: list of edges inside the rectangular environment
    :param edgesOUTenv: list of edges on the boundary of the rectangular environment
    :return: the new tensor list [new_tensor, [#, 'tensor_number'], [#, 'tensor_index_along_edge']]
    """
    for i in range(len(edgesNidx[0])):
        if edgesNidx[0][i] in edgesINenv:
            tensor = np.einsum(tensor, list(range(len(tensor.shape))), np.sqrt(weights[int(edgesNidx[0][i])]),
                               [int(edgesNidx[1][i])], list(range(len(tensor.shape))))
        elif edgesNidx[0][i] in edgesOUTenv:
            tensor = np.einsum(tensor, list(range(len(tensor.shape))), weights[int(edgesNidx[0][i])],
                               [int(edgesNidx[1][i])], list(range(len(tensor.shape))))
        else:
            raise IndexError('Error: 00001')
    return tensor


def absorbInverseWeights(tensor, edgesNidx, weights):
    """
    Absorb inverse neighboring lambda weights into tensor.
    :param tensor: tensor
    :param edgesNidx: list of two lists [[edges], [indices]].
    :param weights: list of lambda weights.
    :return: the new tensor list [new_tensor, [#, 'tensor_number'], [#, 'tensor_index_along_edge']]
    """
    for i in range(len(edgesNidx[0])):
        tensor = np.einsum(tensor, list(range(len(tensor.shape))),
                              weights[int(edgesNidx[0][i])] ** (-1), [int(edgesNidx[1][i])], list(range(len(tensor.shape))))
    return tensor


def indexPermute(tensor):
    """
    Swapping the 'tensor_index_along_edge' index with the 1st index
    :param tensor: [tensor, [#, 'tensor_number'], [#, 'tensor_index_along_edge']]
    :return: the list with the permuted tensor [permuted_tensor, [#, 'tensor_number'], [#, 'tensor_index_along_edge']]
    """
    permutation = np.array(list(range(len(tensor[0].shape))))
    permutation[[1, tensor[2][0]]] = permutation[[tensor[2][0], 1]]
    tensor[0] = np.transpose(tensor[0], permutation)
    return tensor


def rankNrank3(tensor):
    """
    Taking a rank-N tensor (N >= 3) and make it a rank-3 tensor by grouping all indices (2, 3, ..., N - 1).
    :param tensor: the tensor
    :return: the reshaped rank-3 tensor.
    """
    if len(tensor.shape) < 3:
        raise IndexError('Error: 00002')
    shape = np.array(tensor.shape)
    newShape = [shape[0], shape[1], np.prod(shape[2:])]
    newTensor = np.reshape(tensor, newShape)
    return newTensor


def rank2rank3(tensor, physicalDimension):
    """
    Taking a rank-2 tensor and make it a rank-3 tensor by splitting its first dimension. This function is used for
    extracting back the physical dimension of a reshaped tensor.
    :param tensor: rank-2 tensor
    :param physicalDimension: the physical dimension of the tensor.
    :return: rank-3 new tensor such that:
             newTensor.shape = (oldTensor.shape[0], oldTensor.shape[0] / physicalDimension, oldTensor.shape[1])
    """
    if len(tensor.shape) is not 2:
        raise IndexError('Error: 00003')
    newTensor = np.reshape(tensor, [physicalDimension, int(tensor.shape[0] / physicalDimension), tensor.shape[1]])
    return newTensor


def rank3rankN(tensor, oldShape):
    """
    Returning a tensor to its original rank-N rank.
    :param tensor: rank-3 tensor
    :param oldShape: the tensor's original shape
    :return: the tensor in its original shape.
    """
    newTensor = np.reshape(tensor, oldShape)
    return newTensor


def truncationSVD(tensor, leftIdx, rightIdx, keepS=None, maxEigenvalNumber=None):
    """
    Taking a rank-N tensor reshaping it to rank-2 tensor and preforming an SVD operation with/without truncation.
    :param tensor: the tensor
    :param leftIdx: indices to move into 0th index
    :param rightIdx: indices to move into 1st index
    :param keepS: if not None: will return U, S, V^(dagger). if None: will return U * sqrt(S), sqrt(S) * V^(dagger)
    :param maxEigenvalNumber: maximal number of eigenvalues to keep (truncation)
    :return: U, S, V^(dagger) or U * sqrt(S), sqrt(S) * V^(dagger)
    """
    shape = np.array(tensor.shape)
    leftDim = np.prod(shape[[leftIdx]])
    rightDim = np.prod(shape[[rightIdx]])
    if keepS is not None:
        U, S, Vh = np.linalg.svd(tensor.reshape(leftDim, rightDim), full_matrices=False)
        if maxEigenvalNumber is not None:
            U = U[:, 0:maxEigenvalNumber]
            S = S[0:maxEigenvalNumber]
            Vh = Vh[0:maxEigenvalNumber, :]
        return U, S, Vh
    else:
        U, S, Vh = np.linalg.svd(tensor.reshape(leftDim, rightDim), full_matrices=False)
        if maxEigenvalNumber is not None:
            U = U[:, 0:maxEigenvalNumber]
            S = S[0:maxEigenvalNumber]
            Vh = Vh[0:maxEigenvalNumber, :]
        U = np.einsum(U, [0, 1], np.sqrt(S), [1], [0, 1])
        Vh = np.einsum(np.sqrt(S), [0], Vh, [0, 1], [0, 1])
    return U, Vh


def imaginaryTimeEvolution(iTensor,
                           jTensor,
                           middleWeightVector,
                           commonEdge,
                           timeStep,
                           interactionConst,
                           fieldConst,
                           iOperators,
                           jOperators,
                           fieldOperators):
    """
    Applying Imaginary Time Evolution (ITE) on a pair of interacting tensors and returning a rank-4 tensor \theta with
    physical bond dimensions d(i') and d(j') and shape (Q1, d(i'), d(j'), Q2). Q1, Q2 are the dimensions of the QR and
    LQ matrices. The shape of the unitaryGate should be (d(i), d(j), d(i'), d(j')).
    :param iTensor: the left tensor
    :param jTensor: the right tensor
    :param middleWeightVector: the lambda weight associated with the left and right tensors common edge
    :param commonEdge: the tensors common edge
    :param timeStep: the ITE time step
    :param interactionConst: list of interaction constants J_{ij} (len(List) = # of edges)
    :param fieldConst: the field constant usually written as h
    :param iOperators: the operators associated with the i^th tensor in the Hamiltonian
    :param jOperators: the operators associated with the j^th tensor in the Hamiltonian
    :param fieldOperators: the operators associated with the field term in the Hamiltonian
    :return: A rank-4 tensor with shape (Q1, d(i'), d(j'), Q2)
    """
    d = iOperators[0].shape[0]  # physical bond dimension
    interactionHamiltonian = np.zeros((d ** 2, d ** 2), dtype=complex)
    for i in range(len(iOperators)):
        interactionHamiltonian += np.kron(iOperators[i], jOperators[i])
    Hamiltonian = -interactionConst[commonEdge] * interactionHamiltonian\
                  - 0.25 * fieldConst * (np.kron(np.eye(d), fieldOperators)
                  + np.kron(fieldOperators, np.eye(d)))  # 0.25 is for square lattice
    unitaryGate = np.reshape(linalg.expm(-timeStep * Hamiltonian), [d, d, d, d])
    weightMatrix = np.diag(middleWeightVector)
    A = np.einsum(iTensor, [0, 1, 2], weightMatrix, [1, 3], [0, 3, 2])           # A.shape = (d(i), Weight_Vector, Q1)
    A = np.einsum(A, [0, 1, 2], jTensor, [3, 1, 4], [2, 0, 3, 4])                # A.shape = (Q1, d(i), d(j), Q2)
    theta = np.einsum(A, [0, 1, 2, 3], unitaryGate, [1, 2, 4, 5], [0, 4, 5, 3])  # theta.shape = (Q1, d(i'), d(j'), Q2)
    return theta


def tensorNorm(tensor):
    """
    :param tensor: the tensor
    :return: the norm
    """
    tensorConj = np.conj(cp.copy(tensor))
    idx = list(range(len(tensor.shape)))
    norm = np.sqrt(np.einsum(tensor, idx, tensorConj, idx))
    return norm


def updateDEFG(edge, tensors, weights, smat, doubleEdgeFactorGraph):
    """
    DEFG update (description needs to be added)
    :param edge:
    :param tensors:
    :param weights:
    :param smat:
    :param doubleEdgeFactorGraph:
    :return: None
    """
    iFactor, jFactor = getTensors(edge, tensors, smat)
    iEdges, jEdges = getAllTensorsEdges(edge, smat)
    iFactor = absorbSqrtWeights(cp.deepcopy(iFactor), iEdges, weights)
    jFactor = absorbSqrtWeights(cp.deepcopy(jFactor), jEdges, weights)
    doubleEdgeFactorGraph.factors['f' + str(iFactor[1][0])][1] = iFactor[0]
    doubleEdgeFactorGraph.factors['f' + str(jFactor[1][0])][1] = jFactor[0]
    doubleEdgeFactorGraph.nodes['n' + str(edge)][0] = len(weights[edge])


########################################################################################################################
#                                         Simple Update expectations and rdms                                          #
########################################################################################################################


def singleSiteExpectation(tensorIndex, tensors, weights, smat, localOp):
    """
    This function calculates the local expectation value of a single tensor network site using the weights as
    environment.
    :param tensorIndex: the index of the tensor in the structure matrix
    :param tensors: list of tensors in the tensorNet
    :param weights: list of weights
    :param smat: the structure matrix
    :param localOp: the local operator for the expectation value
    :return: single site expectation
    """
    edgeNidx = getEdges(tensorIndex, smat)
    site = absorbWeights(cp.copy(tensors[tensorIndex]), edgeNidx, weights)
    siteConj = absorbWeights(np.conj(cp.copy(tensors[tensorIndex])), edgeNidx, weights)
    normalization = siteNorm(tensorIndex, tensors, weights, smat)

    # setting lists for ncon
    siteIdx = list(range(len(site.shape)))
    siteConjIdx = list(range(len(siteConj.shape)))
    siteConjIdx[0] = len(siteConj.shape)
    localOpIdx = [siteConjIdx[0], siteIdx[0]]
    expectation = ncon.ncon([site, siteConj, localOp], [siteIdx, siteConjIdx, localOpIdx]) / normalization
    return expectation


def siteNorm(tensorIndex, tensors, weights, smat):
    """
    Calculate the normalization of a single tensor network site using the weights as environment (sam as calculating
    this site expectation with np.eye(d)).
    :param tensorIndex: the index of the tensor in the structure matrix
    :param tensors: list of tensors in the tensorNet
    :param weights: list of weights
    :param smat: the structure matrix
    :return: site normalization
    """
    edgeNidx = getEdges(tensorIndex, smat)
    site = absorbWeights(cp.copy(tensors[tensorIndex]), edgeNidx, weights)
    siteConj = absorbWeights(np.conj(cp.copy(tensors[tensorIndex])), edgeNidx, weights)
    normalization = np.einsum(site, list(range(len(site.shape))), siteConj, list(range(len(siteConj.shape))))
    return normalization


def two_site_expectation(Ek, TT, LL, smat, Oij):
    TT = cp.deepcopy(TT)
    LL = cp.deepcopy(LL)

    # calculating the two site normalized expectation given a mutual edge Ek of those two sites (tensors) and the operator Oij
    lamda_k = cp.copy(LL[Ek])

    ## (a) Find tensors Ti, Tj and their corresponding legs connected along edge Ek.
    Ti, Tj = getTensors(Ek, TT, smat)
    Ti_conj, Tj_conj = getConjTensors(Ek, TT, smat)

    # collecting all neighboring (edges, dimensions) without the Ek (edge, dimension)
    i_dim, j_dim = getTensorsEdges(Ek, smat)

    ## (b) Absorb bond vectors (lambdas) to all Em != Ek of Ti, Tj tensors
    Ti[0] = absorbWeights(Ti[0], i_dim, LL)
    Tj[0] = absorbWeights(Tj[0], j_dim, LL)
    Ti_conj[0] = absorbWeights(Ti_conj[0], i_dim, LL)
    Tj_conj[0] = absorbWeights(Tj_conj[0], j_dim, LL)

    ## preparing list of tensors and indices for ncon function
    s = 1000
    t = 2000
    lamda_k_idx = [t, t + 1]
    lamda_k_conj_idx = [t + 2, t + 3]
    Oij_idx = [s, s + 1, s + 2, s + 3]  # (i, j, i', j')

    Ti_idx = list(range(len(Ti[0].shape)))
    Ti_conj_idx = list(range(len(Ti_conj[0].shape)))
    Ti_idx[0] = Oij_idx[0]  # i
    Ti_conj_idx[0] = Oij_idx[2]  # i'
    Ti_idx[Ti[2][0]] = lamda_k_idx[0]
    Ti_conj_idx[Ti_conj[2][0]] = lamda_k_conj_idx[0]

    Tj_idx = list(range(len(Ti[0].shape) + 1, len(Ti[0].shape) + 1 + len(Tj[0].shape)))
    Tj_conj_idx = list(range(len(Ti_conj[0].shape) + 1, len(Ti_conj[0].shape) + 1 + len(Tj_conj[0].shape)))
    Tj_idx[0] = Oij_idx[1]  # j
    Tj_conj_idx[0] = Oij_idx[3]  # j'
    Tj_idx[Tj[2][0]] = lamda_k_idx[1]
    Tj_conj_idx[Tj_conj[2][0]] = lamda_k_conj_idx[1]

    # two site expectation calculation
    tensors = [Ti[0], Ti_conj[0], Tj[0], Tj_conj[0], Oij, np.diag(lamda_k), np.diag(lamda_k)]
    indices = [Ti_idx, Ti_conj_idx, Tj_idx, Tj_conj_idx, Oij_idx, lamda_k_idx, lamda_k_conj_idx]
    two_site_expec = ncon.ncon(tensors, indices)

    ## prepering list of tensors and indices for two site normalization
    p = Ti[0].shape[0]
    eye = np.reshape(np.eye(p * p), (p, p, p, p))
    eye_idx = Oij_idx

    tensors = [Ti[0], Ti_conj[0], Tj[0], Tj_conj[0], eye, np.diag(lamda_k), np.diag(lamda_k)]
    indices = [Ti_idx, Ti_conj_idx, Tj_idx, Tj_conj_idx, eye_idx, lamda_k_idx, lamda_k_conj_idx]
    two_site_norm = ncon.ncon(tensors, indices)
    two_site_expec /= two_site_norm
    return two_site_expec


def two_site_reduced_density_matrix(Ek, TT, LL, smat):
    TT = cp.deepcopy(TT)
    LL = cp.deepcopy(LL)

    # calculating the two site reduced density matrix given a mutual edge Ek of those two sites (tensors)
    lamda_k = cp.copy(LL[Ek])

    ## (a) Find tensors Ti, Tj and their corresponding legs connected along edge Ek.
    Ti, Tj = getTensors(Ek, TT, smat)
    Ti_conj, Tj_conj = getConjTensors(Ek, TT, smat)

    # collecting all neighboring (edges, dimensions) without the Ek (edge, dimension)
    i_dim, j_dim = getTensorsEdges(Ek, smat)

    ## (b) Absorb bond vectors (lambdas) to all Em != Ek of Ti, Tj tensors
    Ti = absorbWeights(Ti, i_dim, LL)
    Tj = absorbWeights(Tj, j_dim, LL)
    Ti_conj = absorbWeights(Ti_conj, i_dim, LL)
    Tj_conj = absorbWeights(Tj_conj, j_dim, LL)

    ## preparing list of tensors and indices for ncon function

    t = 2000
    lamda_k_idx = [t, t + 1]
    lamda_k_conj_idx = [t + 2, t + 3]

    Ti_idx = range(len(Ti[0].shape))
    Ti_conj_idx = range(len(Ti_conj[0].shape))
    Ti_idx[0] = -1  # i
    Ti_conj_idx[0] = -3  # i'
    Ti_idx[Ti[2][0]] = lamda_k_idx[0]
    Ti_conj_idx[Ti_conj[2][0]] = lamda_k_conj_idx[0]

    Tj_idx = range(len(Ti[0].shape) + 1, len(Ti[0].shape) + 1 + len(Tj[0].shape))
    Tj_conj_idx = range(len(Ti_conj[0].shape) + 1, len(Ti_conj[0].shape) + 1 + len(Tj_conj[0].shape))
    Tj_idx[0] = -2  # j
    Tj_conj_idx[0] = -4  # j'
    Tj_idx[Tj[2][0]] = lamda_k_idx[1]
    Tj_conj_idx[Tj_conj[2][0]] = lamda_k_conj_idx[1]

    # two site expectation calculation
    tensors = [Ti[0], Ti_conj[0], Tj[0], Tj_conj[0], np.diag(lamda_k), np.diag(lamda_k)]
    indices = [Ti_idx, Ti_conj_idx, Tj_idx, Tj_conj_idx, lamda_k_idx, lamda_k_conj_idx]
    rdm = ncon.ncon(tensors, indices)
    rdm = rdm.reshape(rdm.shape[0] * rdm.shape[1], rdm.shape[2] * rdm.shape[3])
    rdm /= np.trace(rdm)
    return rdm


def two_site_expectation_with_environment(Ek, env_size, network_shape, TT1, LL1, smat, Oij):
    TT = cp.deepcopy(TT1)
    TTconj = conjTN(cp.deepcopy(TT1))
    LL = cp.deepcopy(LL1)
    p = Oij.shape[0]
    Iop = np.eye(p ** 2).reshape(p, p, p, p)

    # get th environment matrix and the lists of inside and outside edges
    emat = tnf.PEPS_OBC_edge_rect_env(Ek, smat, network_shape, env_size)
    inside, outside = tnf.PEPS_OBC_divide_edge_regions(emat, smat)
    omat = np.arange(smat.shape[0]).reshape(emat.shape)
    tensors_indices = omat[np.nonzero(emat > -1)]

    # absorb edges
    for t in tensors_indices:
        edge_leg = getEdges(t, smat)
        TT[t] = absorbWeights_twoSiteExpectationWithRectangularEnvironment(TT[t], edge_leg, LL, inside, outside)
        TTconj[t] = absorbWeights_twoSiteExpectationWithRectangularEnvironment(TTconj[t], edge_leg, LL, inside, outside)

    # lists and ncon
    t_list, i_list, o_list = nlg.ncon_list_generator_two_site_expectation_with_env_peps_obc(TT, TTconj, Oij, smat, emat, Ek, tensors_indices, inside, outside)
    t_list_n, i_list_n, o_list_n = nlg.ncon_list_generator_two_site_expectation_with_env_peps_obc(TT, TTconj, Iop, smat, emat, Ek, tensors_indices, inside, outside)
    expec = ncon.ncon(t_list, i_list, o_list)
    norm = ncon.ncon(t_list_n, i_list_n, o_list_n)
    expectation = expec / norm
    return expectation


def two_site_exact_expectation(TT, LL, smat, edge, operator):
    TTstar = conjTN(TT)
    TT_tilde = absorb_all_sqrt_bond_vectors(TT, LL, smat)
    TTstar_tilde = absorb_all_sqrt_bond_vectors(TTstar, LL, smat)
    T_list, idx_list = nlg.ncon_list_generator_two_site_exact_expectation_peps(TT_tilde, TTstar_tilde, smat, edge, operator)
    T_list_norm, idx_list_norm = nlg.ncon_list_generator_braket_peps(TT_tilde, TTstar_tilde, smat)
    exact_expectation = ncon.ncon(T_list, idx_list) / ncon.ncon(T_list_norm, idx_list_norm)
    return exact_expectation


def conjTN(TT):
    TTconj = []
    for i in range(len(TT)):
        TTconj.append(np.conj(TT[i]))
    return TTconj


def energy_per_site(TT, LL, smat, Jk, h, Opi, Opj, Op_field):
    TT = cp.deepcopy(TT)
    LL = cp.deepcopy(LL)
    # calculating the normalized energy per site(tensor)
    p = Opi[0].shape[0]
    Aij = np.zeros((p ** 2, p ** 2), dtype=complex)
    for i in range(len(Opi)):
        Aij += np.kron(Opi[i], Opj[i])
    energy = 0
    n, m = np.shape(smat)
    for Ek in range(m):
        Oij = np.reshape(-Jk[Ek] * Aij - 0.25 * h * (np.kron(np.eye(p), Op_field) + np.kron(Op_field, np.eye(p))), (p, p, p, p))
        energy += two_site_expectation(Ek, TT, LL, smat, Oij)
    energy /= n
    return energy


def energy_per_site_with_environment(network_shape, env_size, TT, LL, smat, Jk, h, Opi, Opj, Op_field):
    TT = cp.deepcopy(TT)
    LL = cp.deepcopy(LL)
    # calculating the normalized energy per site(tensor)
    p = Opi[0].shape[0]
    Aij = np.zeros((p ** 2, p ** 2), dtype=complex)
    for i in range(len(Opi)):
        Aij += np.kron(Opi[i], Opj[i])
    energy = 0
    n, m = np.shape(smat)
    for Ek in range(m):
        print(Ek)
        Oij = np.reshape(-Jk[Ek] * Aij - 0.25 * h * (np.kron(np.eye(p), Op_field) + np.kron(Op_field, np.eye(p))), (p, p, p, p))
        energy += two_site_expectation_with_environment(Ek, env_size, network_shape, TT, LL, smat, Oij)
    energy /= n
    return energy


def exact_energy_per_site(TT, LL, smat, Jk, h, Opi, Opj, Op_field):
    # calculating the normalized exact energy per site(tensor)
    TT = cp.deepcopy(TT)
    LL = cp.deepcopy(LL)
    p = Opi[0].shape[0]
    Aij = np.zeros((p ** 2, p ** 2), dtype=complex)
    for i in range(len(Opi)):
        Aij += np.kron(Opi[i], Opj[i])
    energy = 0
    n, m = np.shape(smat)
    for Ek in range(m):
        Oij = np.reshape(-Jk[Ek] * Aij - 0.25 * h * (np.kron(np.eye(p), Op_field) + np.kron(Op_field, np.eye(p))), (p, p, p, p))
        energy += two_site_exact_expectation(TT, LL, smat, Ek, Oij)
    energy /= n
    return energy


def BP_energy_per_site_using_factor_belief(graph, smat, Jk, h, Opi, Opj, Op_field):

    p = Opi[0].shape[0]
    Aij = np.zeros((p ** 2, p ** 2), dtype=complex)
    for i in range(len(Opi)):
        Aij += np.kron(Opi[i], Opj[i])
    energy = 0
    n, m = np.shape(smat)
    for Ek in range(m):
        Oij = np.reshape(-Jk[Ek] * Aij - 0.25 * h * (np.kron(np.eye(p), Op_field) + np.kron(Op_field, np.eye(p))), (p, p, p, p))
        tensors = np.nonzero(smat[:, Ek])[0]
        fi_belief, fj_belief = graph.two_factors_belief('f' + str(tensors[0]), 'f' + str(tensors[1]))
        fi_idx = range(len(fi_belief.shape))
        fj_idx = range(len(fi_belief.shape), len(fi_belief.shape) + len(fj_belief.shape))
        Oij_idx = [1000, 1001, 1002, 1003]
        fi_idx[0] = Oij_idx[0]
        fi_idx[1] = Oij_idx[2]
        fj_idx[0] = Oij_idx[1]
        fj_idx[1] = Oij_idx[3]
        iedges, jedges = getTensorsEdges(Ek, smat)
        '''
        common_edges = []
        for idx_edgei, edgei in enumerate(iedges[0]):
            if edgei in jedges[0]:
                idx_edgej = jedges[0].index(edgei)
                fi_idx[2 * iedges[1][idx_edgei] + 1] = fj_idx[2 * jedges[1][idx_edgej] + 1]
                fi_idx[2 * iedges[1][idx_edgei]] = fj_idx[2 * jedges[1][idx_edgej]]
                common_edges.append(edgei)
        for edge in common_edges:
            idx_edgei = iedges[0].index(edge)
            idx_edgej = jedges[0].index(edge)
            iedges[0].remove(iedges[0][idx_edgei])
            iedges[1].remove(iedges[1][idx_edgei])
            jedges[0].remove(jedges[0][idx_edgej])
            jedges[1].remove(jedges[1][idx_edgej])
        '''
        for leg_idx, leg in enumerate(iedges[1]):
            fi_idx[2 * leg + 1] = fi_idx[2 * leg]
        for leg_idx, leg in enumerate(jedges[1]):
            fj_idx[2 * leg + 1] = fj_idx[2 * leg]
        Ek_legs = smat[np.nonzero(smat[:, Ek])[0], Ek]
        fi_idx[2 * Ek_legs[0]] = fj_idx[2 * Ek_legs[1]]
        fi_idx[2 * Ek_legs[0] + 1] = fj_idx[2 * Ek_legs[1] + 1]
        E = ncon.ncon([fi_belief, fj_belief, Oij], [fi_idx, fj_idx, Oij_idx])
        norm = ncon.ncon([fi_belief, fj_belief, np.eye(p ** 2).reshape((p, p, p, p))], [fi_idx, fj_idx, Oij_idx])
        E_normalized = E / norm
        energy += E_normalized
    energy /= n
    return energy


def BP_two_site_rdm_using_factor_beliefs(Ek, graph, smat):

    tensors = np.nonzero(smat[:, Ek])[0]
    fi_belief, fj_belief = graph.two_factors_belief('f' + str(tensors[0]), 'f' + str(tensors[1]))
    fi_idx = range(len(fi_belief.shape))
    fj_idx = range(len(fi_belief.shape), len(fi_belief.shape) + len(fj_belief.shape))
    fi_idx[0] = -1
    fi_idx[1] = -3
    fj_idx[0] = -2
    fj_idx[1] = -4
    iedges, jedges = getTensorsEdges(Ek, smat)

    for leg_idx, leg in enumerate(iedges[1]):
        fi_idx[2 * leg + 1] = fi_idx[2 * leg]
    for leg_idx, leg in enumerate(jedges[1]):
        fj_idx[2 * leg + 1] = fj_idx[2 * leg]
    Ek_legs = smat[np.nonzero(smat[:, Ek])[0], Ek]
    fi_idx[2 * Ek_legs[0]] = fj_idx[2 * Ek_legs[1]]
    fi_idx[2 * Ek_legs[0] + 1] = fj_idx[2 * Ek_legs[1] + 1]
    rdm = ncon.ncon([fi_belief, fj_belief], [fi_idx, fj_idx])
    rdm = rdm.reshape(rdm.shape[0] * rdm.shape[1], rdm.shape[2] * rdm.shape[3])
    rdm /= np.trace(rdm)
    return rdm


def BP_energy_per_site_using_factor_belief_with_environment(graph, env_size, network_shape, smat, Jk, h, Opi, Opj, Op_field):
    energy = 0
    p = Opi[0].shape[0]
    Aij = np.zeros((p ** 2, p ** 2), dtype=complex)
    Iop = np.eye(Aij.shape[0]).reshape(p, p, p, p)
    for i in range(len(Opi)):
        Aij += np.kron(Opi[i], Opj[i])
    n, m = np.shape(smat)
    for Ek in range(m):
        print('Ek = ', Ek)
        Oij = np.reshape(-Jk[Ek] * Aij - 0.25 * h * (np.kron(np.eye(p), Op_field) + np.kron(Op_field, np.eye(p))), (p, p, p, p))
        f_list, i_list, o_list = nlg.ncon_list_generator_two_site_expectation_with_factor_belief_env_peps_obc_efficient(Ek, graph, env_size, network_shape, smat, Oij)
        f_list_n, i_list_n, o_list_n = nlg.ncon_list_generator_two_site_expectation_with_factor_belief_env_peps_obc_efficient(Ek, graph, env_size, network_shape, smat, Iop)
        expec = ncon.ncon(f_list, i_list, o_list)
        norm = ncon.ncon(f_list_n, i_list_n, o_list_n)
        expectation = expec / norm
        energy += expectation
    energy /= n
    return energy


def BP_energy_per_site_using_rdm_belief(graph, smat, Jk, h, Opi, Opj, Op_field):
    # calculating the normalized exact energy per site(tensor)
    if graph.rdm_belief == None:
        raise IndexError('First calculate rdm beliefs')
    p = Opi[0].shape[0]
    Aij = np.zeros((p ** 2, p ** 2), dtype=complex)
    for i in range(len(Opi)):
        Aij += np.kron(Opi[i], Opj[i])
    energy = 0
    n, m = np.shape(smat)
    for Ek in range(m):
        Oij = np.reshape(-Jk[Ek] * Aij - 0.25 * h * (np.kron(np.eye(p), Op_field) + np.kron(Op_field, np.eye(p))), (p, p, p, p))
        tensors = np.nonzero(smat[:, Ek])[0]
        fi_belief = graph.rdm_belief[tensors[0]]
        fj_belief = graph.rdm_belief[tensors[1]]
        fij = np.einsum(fi_belief, [0, 1], fj_belief, [2, 3], [0, 2, 1, 3])
        Oij_idx = [0, 1, 2, 3]
        E = np.einsum(fij, [0, 1, 2, 3], Oij, Oij_idx)
        norm = np.einsum(fij, [0, 1, 0, 1])
        E_normalized = E / norm
        energy += E_normalized
    energy /= n
    return energy


def trace_distance(a, b):
    # returns the trace distance between the two density matrices a & b
    # d = 0.5 * norm(a - b)
    eigenvalues = np.linalg.eigvals(a - b)
    d = 0.5 * np.sum(np.abs(eigenvalues))
    return d


def tensor_reduced_dm(tensor_idx, TT, LL, smat):
    TT = cp.deepcopy(TT)
    LL = cp.deepcopy(LL)
    normalization = siteNorm(tensor_idx, TT, LL, smat)
    env_edges = np.nonzero(smat[tensor_idx, :])[0]
    env_legs = smat[tensor_idx, env_edges]
    T = cp.deepcopy(TT[tensor_idx])
    T_conj = cp.deepcopy(np.conj(TT[tensor_idx]))

    ## absorb its environment
    for j in range(len(env_edges)):
        T = np.einsum(T, range(len(T.shape)), LL[env_edges[j]], [env_legs[j]], range(len(T.shape)))
        T_conj = np.einsum(T_conj, range(len(T_conj.shape)), LL[env_edges[j]], [env_legs[j]], range(len(T_conj.shape)))

    T_idx = range(len(T.shape))
    T_idx[0] = -1
    T_conj_idx = range(len(T_conj.shape))
    T_conj_idx[0] = -2
    reduced_dm = ncon.ncon([T, T_conj], [T_idx, T_conj_idx])

    return reduced_dm / normalization


def absorb_all_sqrt_bond_vectors(TT, LL, smat):
    TT = cp.deepcopy(TT)
    LL = cp.deepcopy(LL)
    n = len(TT)
    for i in range(n):
        edges = np.nonzero(smat[i, :])[0]
        legs = smat[i, edges]
        for j in range(len(edges)):
            TT[i] = np.einsum(TT[i], range(len(TT[i].shape)), np.sqrt(LL[edges[j]]), [legs[j]], range(len(TT[i].shape)))
    return TT


# ---------------------------------- BP truncation  ---------------------------------


'''
def BPupdate_all_edges(graph, TT, LL, smat, imat, Dmax):
    ## this BP truncation is implemented on all edges

    # run over all edges

    for Ek in range(len(LL)):
        the_node = 'n' + str(Ek)
        Ti, Tj = get_tensors(Ek, TT, smat, imat)
        i_dim, j_dim = get_all_edges(Ek, smat, imat)

        fi = absorb_edges_for_graph(cp.deepcopy(Ti), i_dim, LL)
        fj = absorb_edges_for_graph(cp.deepcopy(Tj), j_dim, LL)

        A, B = AnB_calculation(graph, fi, fj, the_node)
        P = find_P(A, B, Dmax)
        TT, LL = smart_truncation(TT, LL, P, Ek, smat, imat, Dmax)
        graph_update(Ek, TT, LL, smat, imat, graph)
    return TT, LL
'''

def BPupdate_single_edge(TT, LL, smat, Dmax, Ek, graph):
    ## this BP truncation is implemented on a single edge Ek

    # run BP on graph
    #graph.sum_product(t_max, epsilon, dumping, 'init_with_old_messages')

    the_node = 'n' + str(Ek)
    Ti, Tj = getTensors(Ek, TT, smat)
    i_dim, j_dim = getAllTensorsEdges(Ek, smat)

    fi = absorbSqrtWeights(cp.deepcopy(Ti), i_dim, LL)
    fj = absorbSqrtWeights(cp.deepcopy(Tj), j_dim, LL)

    A, B = AnB_calculation(graph, fi, fj, the_node)
    P = find_P(A, B, Dmax)
    TT, LL = smart_truncation(TT, LL, P, Ek, smat, Dmax)
    updateDEFG(Ek, TT, LL, smat, graph)

    return TT, LL


def PEPStoDEnFG_transform(graph, TT, LL, smat):
    # generate the double edge factor graph from PEPS
    factors_list = absorb_all_sqrt_bond_vectors(TT, LL, smat)

    # Adding virtual nodes
    n, m = np.shape(smat)
    for i in range(m):
        graph.add_node(len(LL[i]), 'n' + str(graph.node_count))
    # Adding factors
    for i in range(n):
        # generating the neighboring nodes of the i'th factor
        neighbor_nodes = {}
        edges = np.nonzero(smat[i, :])[0]
        legs = smat[i, edges]
        for j in range(len(edges)):
            neighbor_nodes['n' + str(edges[j])] = legs[j]
        graph.add_factor(neighbor_nodes, np.array(factors_list[i], dtype=complex))
    return graph


def find_P(A, B, D_max):
    A_sqrt = linalg.sqrtm(A)
    B_sqrt = linalg.sqrtm(B)


    ##  Calculate the environment matrix C and its SVD
    C = np.matmul(B_sqrt, np.transpose(A_sqrt))
    u_env, s_env, vh_env = np.linalg.svd(C, full_matrices=False)

    ##  Define P2
    new_s_env = cp.copy(s_env)
    new_s_env[D_max:] = 0
    P2 = np.zeros((len(s_env), len(s_env)))
    np.fill_diagonal(P2, new_s_env)
    P2 /= np.sum(new_s_env)

    ##  Calculating P = A^(-1/2) * V * P2 * U^(dagger) * B^(-1/2)
    P = np.matmul(np.transpose(np.linalg.inv(A_sqrt)), np.matmul(np.transpose(np.conj(vh_env)), np.matmul(P2, np.matmul(np.transpose(np.conj(u_env)), np.linalg.inv(B_sqrt)))))
    return P


def find_P_entrywise(A, B, D_max):
    A_sqrt = linalg.sqrtm(A)
    B_sqrt = linalg.sqrtm(B)

    ##  Calculate the environment matrix C and its SVD
    C = B_sqrt * A_sqrt
    u_env, s_env, vh_env = np.linalg.svd(C, full_matrices=False)

    ##  Define P2
    new_s_env = cp.copy(s_env)
    new_s_env[D_max:] = 0
    P2 = np.zeros((len(s_env), len(s_env)))
    np.fill_diagonal(P2, new_s_env)
    P2 /= np.sum(new_s_env)

    ##  Calculating P = A^(-1/2) * V * P2 * U^(dagger) * B^(-1/2)
    PP = np.matmul(np.matmul(np.transpose(np.conj(vh_env)), P2), np.transpose(np.conj(u_env)))
    P = np.linalg.inv(A_sqrt) * PP * np.linalg.inv(B_sqrt)
    return P


def smart_truncation(TT1, LL1, P, edge, smat, D_max):
    iedges, jedges = getTensorsEdges(edge, smat)
    Ti, Tj = getTensors(edge, TT1, smat)
    Ti = absorbWeights(Ti, iedges, LL1)
    Tj = absorbWeights(Tj, jedges, LL1)

    # absorb the mutual edge
    Ti[0] = np.einsum(Ti[0], range(len(Ti[0].shape)), np.sqrt(LL1[edge]), [Ti[2][0]], range(len(Ti[0].shape)))
    Tj[0] = np.einsum(Tj[0], range(len(Tj[0].shape)), np.sqrt(LL1[edge]), [Tj[2][0]], range(len(Tj[0].shape)))

    # reshaping
    Ti = indexPermute(Ti)
    Tj = indexPermute(Tj)
    i_old_shape = cp.copy(list(Ti[0].shape))
    j_old_shape = cp.copy(list(Tj[0].shape))
    Ti[0] = rankNrank3(Ti[0])
    Tj[0] = rankNrank3(Tj[0])

    # contracting P with Ti and Tj and then using SVD to generate Ti_tilde and Tj_tilde and lamda_tilde
    Ti, Tj, lamda_edge = Accordion(Ti, Tj, P, D_max)

    # reshaping back
    i_old_shape[1] = D_max
    j_old_shape[1] = D_max
    Ti[0] = rank3rankN(Ti[0], i_old_shape)
    Tj[0] = rank3rankN(Tj[0], j_old_shape)
    Ti = indexPermute(Ti)
    Tj = indexPermute(Tj)
    Ti = absorbInverseWeights(Ti, iedges, LL1)
    Tj = absorbInverseWeights(Tj, jedges, LL1)

    # saving tensors and lamda
    TT1[Ti[1][0]] = cp.deepcopy(Ti[0] / tensorNorm(Ti[0]))
    TT1[Tj[1][0]] = cp.deepcopy(Tj[0] / tensorNorm(Tj[0]))

    LL1[edge] = lamda_edge / np.sum(lamda_edge)
    return TT1, LL1

'''
def BPupdate_error(TT, LL, TT_old, LL_old, smat):
    psipsi_T_list, psipsi_idx_list = nlg.ncon_list_generator_for_BPerror(cp.deepcopy(TT), cp.deepcopy(LL), cp.deepcopy(TT), cp.deepcopy(LL), smat)
    psiphi_T_list, psiphi_idx_list = nlg.ncon_list_generator_for_BPerror(cp.deepcopy(TT), cp.deepcopy(LL), cp.deepcopy(TT_old), cp.deepcopy(LL_old), smat)
    phiphi_T_list, phiphi_idx_list = nlg.ncon_list_generator_for_BPerror(cp.deepcopy(TT_old), cp.deepcopy(LL_old), cp.deepcopy(TT_old), cp.deepcopy(LL_old), smat)
    phipsi_T_list, phipsi_idx_list = nlg.ncon_list_generator_for_BPerror(cp.deepcopy(TT_old), cp.deepcopy(LL_old), cp.deepcopy(TT), cp.deepcopy(LL), smat)

    psipsi = ncon.ncon(psipsi_T_list, psipsi_idx_list)
    psiphi = ncon.ncon(psiphi_T_list, psiphi_idx_list)
    phipsi = ncon.ncon(phipsi_T_list, phipsi_idx_list)
    phiphi = ncon.ncon(phiphi_T_list, phiphi_idx_list)

    psi_norm = np.sqrt(psipsi)
    phi_norm = np.sqrt(phiphi)
    # print('overlap_exact = ', psiphi / psi_norm / phi_norm)
    error = 2 - psiphi / psi_norm / phi_norm - phipsi / psi_norm / phi_norm
    return error
'''

def Accordion(Ti, Tj, P, D_max):
    # contracting two tensors i, j with P and SVD (with truncation) back
    L = cp.deepcopy(Ti[0])
    R = cp.deepcopy(Tj[0])

    A = np.einsum(L, [0, 1, 2], P, [1, 3], [0, 3, 2])  # (i, Ek, Q1)
    theta = np.einsum(A, [0, 1, 2], R, [3, 1, 4], [2, 0, 3, 4])  # (Q1, i, j, Q2)

    R_tild, lamda_k, L_tild = truncationSVD(theta, [0, 1], [2, 3], keep_s='yes', max_eigen_num=D_max)

    # reshaping R_tild and L_tild back
    R_tild_new_shape = [Ti[0].shape[2], Ti[0].shape[0], R_tild.shape[1]]  # (d, i, D')
    R_transpose = [1, 2, 0]
    L_tild_new_shape = [L_tild.shape[0], Tj[0].shape[0], Tj[0].shape[2]]  # (D', j, d)
    L_transpose = [1, 0, 2]

    R_tild = np.reshape(R_tild, R_tild_new_shape)
    Ti[0] = np.transpose(R_tild, R_transpose)  # (i, D', ...)
    L_tild = np.reshape(L_tild, L_tild_new_shape)
    Tj[0] = np.transpose(L_tild, L_transpose)  # (j, D', ...)

    return Ti, Tj, lamda_k


def AnB_calculation(graph, Ti, Tj, node_Ek):
    A = graph.f2n_message_chnaged_factor('f' + str(Ti[1][0]), node_Ek, graph.messages_n2f, cp.deepcopy(Ti[0]))
    B = graph.f2n_message_chnaged_factor('f' + str(Tj[1][0]), node_Ek, graph.messages_n2f, cp.deepcopy(Tj[0]))

    #A = graph.f2n_message_chnaged_factor_without_matching_dof('f' + str(Ti[1][0]), node_Ek, graph.messages_n2f, cp.deepcopy(Ti[0]))
    #B = graph.f2n_message_chnaged_factor_without_matching_dof('f' + str(Tj[1][0]), node_Ek, graph.messages_n2f, cp.deepcopy(Tj[0]))
    #print('A, A1', np.sum(np.abs(A - A1)) / np.sum(A) / np.sum(A1))
    #print('B, B1', np.sum(np.abs(B - B1)) / np.sum(B) / np.sum(B1))


    return A, B




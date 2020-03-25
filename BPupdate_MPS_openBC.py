import numpy as np
import ncon as ncon
import copy as cp
from scipy import linalg
import DoubleEdgeFactorGraphs as defg
import time
import matplotlib.pyplot as plt
import ncon_lists_generator as nlg

"""
    A module for the function PEPS_BPupdate which preform the BPupdate algorithm over a given PEPS Tensor Network 

"""


def PEPS_BPupdate(TT1, LL1, dt, Jk, h, Aij, Bij, imat, smat, D_max):
    """
    :param TT: list of tensors in the tensor network TT = [T1, T2, T3, ..., Tp]
    :param LL: list of lists of the lambdas LL = [L1, L2, ..., Ls]
    :param imat: The incidence matrix which indicates which tensor connect to which edge (as indicated in the paper)
    :param smat: The structure matrix which indicates which leg of every tensor is connected to which edge
    :param D_max: maximal virtual dimension

    """
    TT = cp.deepcopy(TT1)
    LL = cp.deepcopy(LL1)


    n, m = np.shape(imat)
    for Ek in range(m):
        lamda_k = cp.deepcopy(LL[Ek])

        ## (a) Find tensors Ti, Tj and their corresponding legs connected along edge Ek.
        Ti, Tj = get_tensors(Ek, TT, smat, imat)

        iedges = list(np.nonzero(smat[Ti[1][0], :])[0])
        ilegs = list(smat[Ti[1][0], iedges])
        jedges = list(np.nonzero(smat[Tj[1][0], :])[0])
        jlegs = list(smat[Tj[1][0], jedges])
        iedges.remove(Ek)
        ilegs.remove(smat[Ti[1][0], Ek])
        jedges.remove(Ek)
        jlegs.remove(smat[Tj[1][0], Ek])

        for ii in range(len(iedges)):
            Ti[0] = np.einsum(Ti[0], range(len(Ti[0].shape)), LL[iedges[ii]], [ilegs[ii]], range(len(Ti[0].shape)))
        for ii in range(len(jedges)):
            Tj[0] = np.einsum(Tj[0], range(len(Tj[0].shape)), LL[jedges[ii]], [jlegs[ii]], range(len(Tj[0].shape)))

        # permuting the Ek leg of tensors i and j into the 1'st dimension
        Ti = dim_perm(Ti)
        Tj = dim_perm(Tj)

        ## (e) Contract the ITE gate Uij, with R, L, and lambda_k to form theta tensor.
        theta, [l, r] = imaginary_time_evolution_MPSopenBC(Ti[0], Tj[0], lamda_k, Ek, dt, Jk, h, Aij, Bij)  # (Q1, i', j', Q2)

        ## (f) Obtain R', L', lambda'_k tensors by applying an SVD to theta
        R_tild, lamda_k_tild, L_tild = svd(theta, range(l - 1), range(l - 1, r + l - 2), keep_s='yes', max_eigen_num=D_max)
        #R_tild, lamda_k_tild, L_tild = svd(theta, range(l - 1), range(l - 1, r + l - 2), keep_s='yes')


        # reshaping R_tild and L_tild back
        if l == 2:
            R_tild_new_shape = [Ti[0].shape[0], R_tild.shape[1]]  # (i, D')
            R_transpose = [0, 1]
        if l == 3:
            R_tild_new_shape = [Ti[0].shape[2], Ti[0].shape[0], R_tild.shape[1]]  # (d, i, D')
            R_transpose = [1, 2, 0]
        if r == 2:
            L_tild_new_shape = [L_tild.shape[0], Tj[0].shape[0]]  # (D', j)
            L_transpose = [1, 0]
        if r == 3:
            L_tild_new_shape = [L_tild.shape[0], Tj[0].shape[0], Tj[0].shape[2]]  # (D', j, d)
            L_transpose = [1, 0, 2]

        R_tild = np.reshape(R_tild, R_tild_new_shape)
        Ti[0] = np.transpose(R_tild, R_transpose)    # (i, D', ...)
        L_tild = np.reshape(L_tild, L_tild_new_shape)
        Tj[0] = np.transpose(L_tild, L_transpose)  # (j, D', ...)


        # permuting back the legs of Ti and Tj
        Ti = dim_perm(Ti)
        Tj = dim_perm(Tj)

        ## (i) Remove bond matrices lambda_m from virtual legs m != Ek to obtain the updated tensors Ti~, Tj~.
        for ii in range(len(iedges)):
            Ti[0] = np.einsum(Ti[0], range(len(Ti[0].shape)), LL[iedges[ii]] ** (-1), [ilegs[ii]],
                              range(len(Ti[0].shape)))
        for ii in range(len(jedges)):
            Tj[0] = np.einsum(Tj[0], range(len(Tj[0].shape)), LL[jedges[ii]] ** (-1), [jlegs[ii]],
                              range(len(Tj[0].shape)))

        # Normalize and save new Ti Tj and lambda_k
        TT[Ti[1][0]] = Ti[0] / tensor_normalization(Ti[0])
        TT[Tj[1][0]] = Tj[0] / tensor_normalization(Tj[0])
        LL[Ek] = lamda_k_tild / np.sum(lamda_k_tild)


        #TT, LL = smart_update(TT, LL, smat, imat, Ek, D_max)
    return TT, LL


def get_tensors(edge, tensors, structure_matrix, incidence_matrix):
    tensors = cp.deepcopy(tensors)
    tidx = np.nonzero(incidence_matrix[:, edge])[0]
    tdim = structure_matrix[tidx, edge]
    Ti = [tensors[tidx[0]], [tidx[0], 'tensor_number'], [tdim[0], 'tensor_Ek_leg']]
    Tj = [tensors[tidx[1]], [tidx[1], 'tensor_number'], [tdim[1], 'tensor_Ek_leg']]
    return Ti, Tj


def get_conjugate_tensors(edge, tensors, structure_matrix, incidence_matrix):
    tidx = np.nonzero(incidence_matrix[:, edge])[0]
    tdim = structure_matrix[tidx, edge]
    Ti = [cp.deepcopy(np.conj(tensors[tidx[0]])), [tidx[0], 'tensor_number'], [tdim[0], 'tensor_Ek_leg']]
    Tj = [cp.deepcopy(np.conj(tensors[tidx[1]])), [tidx[1], 'tensor_number'], [tdim[1], 'tensor_Ek_leg']]
    return Ti, Tj


def get_edges(edge, structure_matrix, incidence_matrix):
    tidx = np.nonzero(incidence_matrix[:, edge])[0]
    i_dim = [list(np.nonzero(incidence_matrix[tidx[0], :])[0]),
             list(structure_matrix[tidx[0], np.nonzero(incidence_matrix[tidx[0], :])[0]])]
    j_dim = [list(np.nonzero(incidence_matrix[tidx[1], :])[0]),
             list(structure_matrix[tidx[1], np.nonzero(incidence_matrix[tidx[1], :])[0]])]
    # removing the Ek edge and leg
    i_dim[0].remove(edge)
    i_dim[1].remove(structure_matrix[tidx[0], edge])
    j_dim[0].remove(edge)
    j_dim[1].remove(structure_matrix[tidx[1], edge])
    return i_dim, j_dim


def absorb_edges(tensor, edges_dim, bond_vectors):
    for i in range(len(edges_dim[0])):
        tensor[0] = np.einsum(tensor[0], range(len(tensor[0].shape)), bond_vectors[edges_dim[0][i]], [edges_dim[1][i]],
                              range(len(tensor[0].shape)))
    return tensor


def remove_edges(tensor, edges_dim, bond_vectors):
    for i in range(len(edges_dim[0])):
        tensor[0] = np.einsum(tensor[0], range(len(tensor[0].shape)), bond_vectors[edges_dim[0][i]] ** (-1),
                              [edges_dim[1][i]], range(len(tensor[0].shape)))
    return tensor


def dim_perm(tensor):
    # swapping the k leg with the element in the 1 place
    permutation = np.array(range(len(tensor[0].shape)))
    permutation[[1, tensor[2][0]]] = permutation[[tensor[2][0], 1]]
    tensor[0] = np.transpose(tensor[0], permutation)
    return tensor


def rankN_to_rank3(tensor):
    # taking a rank N>=3 tensor and make it a rank 3 tensor by grouping all dimensions [2, 3, ..., N]
    if len(tensor.shape) < 3:
        return tensor
    shape = np.array(cp.copy(tensor.shape))
    new_shape = [shape[0], shape[1], np.prod(shape[2:])]
    Pi = np.reshape(tensor, new_shape)
    return Pi


def rank2_to_rank3(tensor, physical_dim):
    # taking a rank N=2 tensor and make it a rank 3 tensor where the physical dimension and the Ek dimension are [0, 1] respectively
    if len(tensor.shape) is not 2:
        raise IndexError('expecting tensor rank N=2. instead got tensor of rank=', len(tensor.shape))
    new_tensor = np.reshape(tensor, [physical_dim, tensor.shape[0] / physical_dim, tensor.shape[1]])
    return new_tensor


def rank3_to_rankN(tensor, old_shape):
    new_tensor = np.reshape(tensor, old_shape)
    return new_tensor


def svd(tensor, left_legs, right_legs, keep_s=None, max_eigen_num=None):
    shape = np.array(tensor.shape)
    left_dim = np.prod(shape[[left_legs]])
    right_dim = np.prod(shape[[right_legs]])
    if keep_s == 'yes':
        u, s, vh = np.linalg.svd(tensor.reshape(left_dim, right_dim), full_matrices=False)
        if max_eigen_num is not None:
            u = u[:, 0:max_eigen_num]
            s = s[0:max_eigen_num]
            vh = vh[0:max_eigen_num, :]
        return u, s, vh
    else:
        u, s, vh = np.linalg.svd(tensor.reshape(left_dim, right_dim), full_matrices=False)
        if max_eigen_num is not None:
            u = u[:, 0:max_eigen_num]
            s = s[0:max_eigen_num]
            vh = vh[0:max_eigen_num, :]
        u = np.einsum(u, [0, 1], np.sqrt(s), [1], [0, 1])
        vh = np.einsum(np.sqrt(s), [0], vh, [0, 1], [0, 1])
    return u, vh


def imaginary_time_evolution(left_tensor, right_tensor, bond_vector, Ek, dt, Jk, h, Aij, Bij):
    # applying ITE and returning a rank 4 tensor with physical dimensions, i' and j' at (Q1, i', j', Q2)
    # the indices of the unitary_time_op should be (i, j, i', j')
    p = np.int(np.sqrt(np.float(Aij.shape[0])))
    hij = -Jk[Ek] * Aij - 0.5 * h * Bij
    unitary_time_op = np.reshape(linalg.expm(-dt * hij), [p, p, p, p])
    bond_matrix = np.diag(bond_vector)
    A = np.einsum(left_tensor, [0, 1, 2], bond_matrix, [1, 3], [0, 3, 2])  # (i, Ek, Q1)
    A = np.einsum(A, [0, 1, 2], right_tensor, [3, 1, 4], [2, 0, 3, 4])  # (Q1, i, j, Q2)
    theta = np.einsum(A, [0, 1, 2, 3], unitary_time_op, [1, 2, 4, 5], [0, 4, 5, 3])  # (Q1, i', j', Q2)
    return theta


def imaginary_time_evolution_MPSopenBC(left_tensor, right_tensor, bond_vector, Ek, dt, Jk, h, Aij, Bij):
    # applying ITE and returning a rank 4 tensor with physical dimensions, i' and j'
    # the indices of the unitary_time_op should be (i, j, i', j')
    p = np.int(np.sqrt(np.float(Aij.shape[0])))
    hij = Jk[Ek] * Aij + 0.5 * h * Bij
    unitary_time_op = np.reshape(linalg.expm(-dt * hij), [p, p, p, p])
    bond_matrix = np.diag(bond_vector)
    l = len(left_tensor.shape)
    r = len(right_tensor.shape)
    if l == 2 and r == 3:
        A = np.einsum(left_tensor, [0, 1], bond_matrix, [1, 3], [0, 3])  # (i, Ek)
        A = np.einsum(A, [0, 1], right_tensor, [2, 1, 3], [0, 2, 3])  # (i, j, d)
        theta = np.einsum(A, [0, 1, 2], unitary_time_op, [0, 1, 4, 5], [4, 5, 2])  # (i', j', d)
    if l == 3 and r == 2:
        A = np.einsum(left_tensor, [0, 1, 2], bond_matrix, [1, 3], [0, 3, 2])  # (i, Ek, d)
        A = np.einsum(A, [0, 1, 2], right_tensor, [3, 1], [2, 0, 3])  # (d, i, j)
        theta = np.einsum(A, [0, 1, 2], unitary_time_op, [1, 2, 4, 5], [0, 4, 5])  # (d, i', j')
    if l == 2 and r == 2:
        A = np.einsum(left_tensor, [0, 1], bond_matrix, [1, 3], [0, 3])  # (i, Ek)
        A = np.einsum(A, [0, 1], right_tensor, [3, 1], [0, 3])  # (i, j)
        theta = np.einsum(A, [0, 1], unitary_time_op, [0, 1, 4, 5], [4, 5])  # (i', j')
    if l == 3 and r == 3:
        A = np.einsum(left_tensor, [0, 1, 2], bond_matrix, [1, 3], [0, 3, 2])  # (i, Ek, Q1)
        A = np.einsum(A, [0, 1, 2], right_tensor, [3, 1, 4], [2, 0, 3, 4])  # (Q1, i, j, Q2)
        theta = np.einsum(A, [0, 1, 2, 3], unitary_time_op, [1, 2, 4, 5], [0, 4, 5, 3])  # (Q1, i', j', Q2)
    return theta, [l, r]


def tensor_normalization(T):
    T_conj = np.conj(cp.deepcopy(T))
    idx = range(len(T.shape))
    norm = np.einsum(T, idx, T_conj, idx)
    return np.sqrt(norm)


def single_tensor_expectation(tensor_idx, TT, LL, imat, smat, Oi):
    TT = cp.deepcopy(TT)
    LL = cp.deepcopy(LL)
    normalization = site_norm(tensor_idx, TT, LL, imat, smat)

    env_edges = np.nonzero(imat[tensor_idx, :])[0]
    env_legs = smat[tensor_idx, env_edges]
    T = TT[tensor_idx]
    T_conj = np.conj(TT[tensor_idx])

    ## absorb its environment
    for j in range(len(env_edges)):
        T = np.einsum(T, range(len(T.shape)), LL[env_edges[j]], [env_legs[j]], range(len(T.shape)))
        T_conj = np.einsum(T_conj, range(len(T_conj.shape)), LL[env_edges[j]], [env_legs[j]], range(len(T_conj.shape)))

    T_idx = range(len(T.shape))
    T_conj_idx = range(len(T_conj.shape))
    T_conj_idx[0] = len(T_conj.shape)
    operator_idx = [T_conj_idx[0], T_idx[0]]
    expectation = ncon.ncon([T, T_conj, Oi], [T_idx, T_conj_idx, operator_idx])
    return expectation / normalization


def magnetization(TT, LL, imat, smat, Oi):
    # calculating the average magnetization per site
    magnetization = 0
    tensors_indices = range(len(TT))
    for i in tensors_indices:
        magnetization += single_tensor_expectation(i, TT, LL, imat, smat, Oi)
    magnetization /= len(TT)
    return magnetization


def site_norm(tensor_idx, TT, LL, imat, smat):
    TT = cp.deepcopy(TT)
    LL = cp.deepcopy(LL)
    env_edges = np.nonzero(imat[tensor_idx, :])[0]
    env_legs = smat[tensor_idx, env_edges]
    T = TT[tensor_idx]
    T_conj = np.conj(TT[tensor_idx])

    ## absorb its environment
    for j in range(len(env_edges)):
        T = np.einsum(T, range(len(T.shape)), LL[env_edges[j]], [env_legs[j]], range(len(T.shape)))
        T_conj = np.einsum(T_conj, range(len(T_conj.shape)), LL[env_edges[j]], [env_legs[j]], range(len(T_conj.shape)))

    T_idx = range(len(T.shape))
    T_conj_idx = range(len(T_conj.shape))
    normalization = np.einsum(T, T_idx, T_conj, T_conj_idx)
    return normalization


def two_site_expectation(Ek, TT1, LL1, imat, smat, Oij):
    TT = cp.deepcopy(TT1)
    LL = cp.deepcopy(LL1)

    # calculating the two site normalized expectation given a mutual edge Ek of those two sites (tensors) and the operator Oij
    lamda_k = cp.copy(LL[Ek])

    ## (a) Find tensors Ti, Tj and their corresponding legs connected along edge Ek.
    Ti, Tj = get_tensors(Ek, TT, smat, imat)
    Ti_conj, Tj_conj = get_conjugate_tensors(Ek, TT, smat, imat)

    # collecting all neighboring (edges, dimensions) without the Ek (edge, dimension)
    i_dim, j_dim = get_edges(Ek, smat, imat)

    ## (b) Absorb bond vectors (lambdas) to all Em != Ek of Ti, Tj tensors
    Ti = absorb_edges(Ti, i_dim, LL)
    Tj = absorb_edges(Tj, j_dim, LL)
    Ti_conj = absorb_edges(Ti_conj, i_dim, LL)
    Tj_conj = absorb_edges(Tj_conj, j_dim, LL)

    ## preparing list of tensors and indices for ncon function
    s = 1000
    t = 2000
    lamda_k_idx = [t, t + 1]
    lamda_k_conj_idx = [t + 2, t + 3]
    Oij_idx = [s, s + 1, s + 2, s + 3]  # (i, j, i', j')

    Ti_idx = range(len(Ti[0].shape))
    Ti_conj_idx = range(len(Ti_conj[0].shape))
    Ti_idx[0] = Oij_idx[0]  # i
    Ti_conj_idx[0] = Oij_idx[2]  # i'
    Ti_idx[Ti[2][0]] = lamda_k_idx[0]
    Ti_conj_idx[Ti_conj[2][0]] = lamda_k_conj_idx[0]

    Tj_idx = range(len(Ti[0].shape) + 1, len(Ti[0].shape) + 1 + len(Tj[0].shape))
    Tj_conj_idx = range(len(Ti_conj[0].shape) + 1, len(Ti_conj[0].shape) + 1 + len(Tj_conj[0].shape))
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


def two_site_exact_expectation(TT, LL, smat, edge, operator):
    TTstar = conjTN(TT)
    TT_tilde = absorb_all_bond_vectors(TT, LL, smat)
    TTstar_tilde = absorb_all_bond_vectors(TTstar, LL, smat)
    T_list, idx_list = nlg.ncon_list_generator_two_site_exact_expectation_mps(TT_tilde, TTstar_tilde, smat, edge, operator)
    T_list_norm, idx_list_norm = nlg.ncon_list_generator_braket_mps(TT_tilde, TTstar_tilde, smat)
    exact_expectation = ncon.ncon(T_list, idx_list) / ncon.ncon(T_list_norm, idx_list_norm)
    return exact_expectation


def two_site_bp_expectation(TT, LL, smat, edge, operator):
    TTstar = conjTN(TT)
    TT_tilde = cp.deepcopy(TT)
    TTstar_tilde = cp.deepcopy(TTstar)
    T_list, idx_list = nlg.ncon_list_generator_two_site_exact_expectation_mps(TT_tilde, TTstar_tilde, smat, edge, operator)
    T_list_norm, idx_list_norm = nlg.ncon_list_generator_braket_mps(TT_tilde, TTstar_tilde, smat)
    exact_expectation = ncon.ncon(T_list, idx_list) / ncon.ncon(T_list_norm, idx_list_norm)
    return exact_expectation


def conjTN(TT):
    TTconj = []
    for i in range(len(TT)):
        TTconj.append(np.conj(TT[i]))
    return TTconj


def energy_per_site(TT1, LL1, imat, smat, Jk, h, Aij, Bij):
    TT = cp.deepcopy(TT1)
    LL = cp.deepcopy(LL1)
    # calculating the normalized energy per site(tensor)
    p = np.int(np.sqrt(np.float(Aij.shape[0])))
    energy = 0
    n, m = np.shape(imat)
    for Ek in range(m):
        Oij = np.reshape(Jk[Ek] * Aij + 0.5 * h * Bij, (p, p, p, p))
        energy += two_site_expectation(Ek, TT, LL, imat, smat, Oij)
    energy /= n
    return energy


def exact_energy_per_site(TT, LL, smat, Jk, h, Aij, Bij):
    # calculating the normalized exact energy per site(tensor)
    TT = cp.deepcopy(TT)
    LL = cp.deepcopy(LL)
    p = np.int(np.sqrt(np.float(Aij.shape[0])))
    energy = 0
    n, m = np.shape(smat)
    for Ek in range(m):
        Oij = np.reshape(Jk[Ek] * Aij + 0.5 * h * Bij, (p, p, p, p))
        energy += two_site_exact_expectation(TT, LL, smat, Ek, Oij)
    energy /= n
    return energy


def BP_energy_per_site(TT, LL, smat, Jk, h, Aij, Bij):
    # calculating the normalized exact energy per site(tensor)
    TT = cp.deepcopy(TT)
    LL = cp.deepcopy(LL)
    p = np.int(np.sqrt(np.float(Aij.shape[0])))
    energy = 0
    n, m = np.shape(smat)
    for Ek in range(m):
        Oij = np.reshape(Jk[Ek] * Aij + 0.5 * h * Bij, (p, p, p, p))
        energy += two_site_bp_expectation(TT, LL, smat, Ek, Oij)
    energy /= n
    return energy


def trace_distance(a, b):
    # returns the trace distance between the two density matrices a & b
    # d = 0.5 * norm(a - b)
    eigenvalues = np.linalg.eigvals(a - b)
    d = 0.5 * np.sum(np.abs(eigenvalues))
    return d


def tensor_reduced_dm(tensor_idx, TT, LL, smat, imat):
    TT = cp.deepcopy(TT)
    LL = cp.deepcopy(LL)
    normalization = site_norm(tensor_idx, TT, LL, imat, smat)
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


def absorb_all_bond_vectors(TT, LL, smat):
    TT1 = cp.deepcopy(TT)
    LL1 = cp.deepcopy(LL)
    n = len(TT1)
    for i in range(n):
        edges = np.nonzero(smat[i, :])[0]
        legs = smat[i, edges]
        for j in range(len(edges)):
            TT1[i] = np.einsum(TT1[i], range(len(TT1[i].shape)), np.sqrt(LL1[edges[j]]), [legs[j]], range(len(TT1[i].shape)))
    return TT1


# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

def smart_update(TT1, LL1, smat, imat, edge, D_max):
    TT = cp.deepcopy(TT1)
    LL = cp.deepcopy(LL1)
    A, B = AB_contraction(TT, LL, smat, edge)


    TTstar = conjTN(TT)
    TT_tilde = absorb_all_bond_vectors(TT, LL, smat)
    TTstar_tilde = absorb_all_bond_vectors(TTstar, LL, smat)
    T_list_norm, idx_list_norm = nlg.ncon_list_generator_braket_mps(TT_tilde, TTstar_tilde, smat)
    norm = ncon.ncon(T_list_norm, idx_list_norm)
    AB_norm = np.einsum(A, [0, 1], B, [1, 0])

    P = find_P(A, B, D_max)
    TT_new, LL_new = smart_truncation(TT, LL, P, edge, smat, imat, D_max)
    return TT_new, LL_new


def BPupdate(TT, LL, smat, imat, t_max, epsilon, dumping, Dmax):
    TT_old = cp.deepcopy(TT)
    LL_old = cp.deepcopy(LL)
    TT = cp.deepcopy(TT)
    LL = cp.deepcopy(LL)

    graph = defg.Graph()
    graph = MPStoDEnFG_transform(graph, TT, LL, smat)
    graph.sum_product(t_max, epsilon, dumping)
    graph.calc_node_belief()
    for Ek in range(len(LL)):
        P = find_P(graph, Ek, smat, Dmax)
        TT, LL = smart_truncation(TT, LL, P, Ek, smat, imat, Dmax)
        # BPerror = BPupdate_error(TT, LL, TT_old, LL_old, smat)
        # print('BP_error = ', BPerror)
    return TT, LL


def BPupdate_single_edge(TT1, LL1, smat, imat, t_max, epsilon, dumping, Dmax, Ek):
    TT_old = cp.deepcopy(TT1)
    LL_old = cp.deepcopy(LL1)
    TT = cp.deepcopy(TT1)
    LL = cp.deepcopy(LL1)

    graph = defg.Graph()
    graph = MPStoDEnFG_transform(graph, TT, LL, smat)
    graph.sum_product(t_max, epsilon, dumping)
    P = find_P(graph, Ek, smat, Dmax)
    TT, LL = smart_truncation(TT, LL, P, Ek, smat, imat, Dmax)
    # BPerror = BPupdate_error(TT, LL, TT_old, LL_old, smat)
    # print('BP_error = ', BPerror)
    return TT, LL


def MPStoDEnFG_transform(graph, TT, LL, smat):
    factors_list = absorb_all_bond_vectors(TT, LL, smat)
    #factors_list = cp.deepcopy(TT)

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
        graph.add_factor(neighbor_nodes, factors_list[i])
    return graph


def find_P(A, B, D_max):
    A_sqrt = linalg.sqrtm(A)
    B_sqrt = linalg.sqrtm(np.transpose(B))

    ##  Calculate the environment matrix C and its SVD
    C = np.matmul(B_sqrt, A_sqrt)
    #C = np.einsum(B_sqrt, [0, 1], A_sqrt, [1, 0], [0, 1])
    u_env, s_env, vh_env = np.linalg.svd(C, full_matrices=False)

    ##  Define P2
    new_s_env = cp.copy(s_env)
    new_s_env[D_max:] = 0
    P2 = np.zeros((len(s_env), len(s_env)))
    np.fill_diagonal(P2, new_s_env)
    P2 /= np.sum(new_s_env)

    ##  Calculating P = A^(-1/2) * U^(dagger) * P2 * V * B^(-1/2)
    P = np.matmul(np.linalg.inv(A_sqrt), np.matmul(np.transpose(np.conj(vh_env)), np.matmul(P2, np.matmul(np.transpose(np.conj(u_env)), np.linalg.inv(B_sqrt)))))
    # overlap = np.trace(np.matmul(A, np.matmul(P, B)))
    # print('overlap = ', overlap)
    return P


def smart_truncation(TT1, LL1, P, edge, smat, imat, D_max):
    iedges, jedges = get_edges(edge, smat, imat)
    Ti, Tj = get_tensors(edge, TT1, smat, imat)
    Ti = absorb_edges(Ti, iedges, LL1)
    Tj = absorb_edges(Tj, jedges, LL1)
    Ti[0] = np.einsum(Ti[0], range(len(Ti[0].shape)), np.sqrt(LL1[edge]), [Ti[2][0]], range(len(Ti[0].shape)))
    Tj[0] = np.einsum(Tj[0], range(len(Tj[0].shape)), np.sqrt(LL1[edge]), [Tj[2][0]], range(len(Tj[0].shape)))
    Ti = dim_perm(Ti)
    Tj = dim_perm(Tj)
    Ti, Tj, lamda_edge = Accordion(Ti, Tj, P, D_max)
    Ti = dim_perm(Ti)
    Tj = dim_perm(Tj)
    Ti = remove_edges(Ti, iedges, LL1)
    Tj = remove_edges(Tj, jedges, LL1)
    TT1[Ti[1][0]] = cp.deepcopy(Ti[0] / tensor_normalization(Ti[0]))
    TT1[Tj[1][0]] = cp.deepcopy(Tj[0] / tensor_normalization(Tj[0]))
    LL1[edge] = lamda_edge / np.sum(lamda_edge)
    return TT1, LL1


def BPupdate_error(TT, LL, TT_old, LL_old, smat):
    psipsi_T_list, psipsi_idx_list = nlg.ncon_list_generator_for_BPerror(cp.deepcopy(TT), cp.deepcopy(LL),
                                                                         cp.deepcopy(TT), cp.deepcopy(LL), smat)
    psiphi_T_list, psiphi_idx_list = nlg.ncon_list_generator_for_BPerror(cp.deepcopy(TT), cp.deepcopy(LL),
                                                                         cp.deepcopy(TT_old), cp.deepcopy(LL_old), smat)
    phiphi_T_list, phiphi_idx_list = nlg.ncon_list_generator_for_BPerror(cp.deepcopy(TT_old), cp.deepcopy(LL_old),
                                                                         cp.deepcopy(TT_old), cp.deepcopy(LL_old), smat)
    phipsi_T_list, phipsi_idx_list = nlg.ncon_list_generator_for_BPerror(cp.deepcopy(TT_old), cp.deepcopy(LL_old),
                                                                         cp.deepcopy(TT), cp.deepcopy(LL), smat)

    psipsi = ncon.ncon(psipsi_T_list, psipsi_idx_list)
    psiphi = ncon.ncon(psiphi_T_list, psiphi_idx_list)
    phipsi = ncon.ncon(phipsi_T_list, phipsi_idx_list)
    phiphi = ncon.ncon(phiphi_T_list, phiphi_idx_list)

    psi_norm = np.sqrt(psipsi)
    phi_norm = np.sqrt(phiphi)
    # print('overlap_exact = ', psiphi / psi_norm / phi_norm)
    error = 2 - psiphi / psi_norm / phi_norm - phipsi / psi_norm / phi_norm
    return error


def AB_contraction(TT1, LL1, smat, edge):
    # calculating the given edge two sides full contractions A (left) and B (right)
    TT = cp.deepcopy(TT1)
    TTconj = conjTN(cp.deepcopy(TT1))
    LL = cp.deepcopy(LL1)
    l = len(LL)
    tensors = absorb_all_bond_vectors(TT, LL, smat)
    conj_tensors = absorb_all_bond_vectors(TTconj, LL, smat)
    A = np.einsum(tensors[0], [0, 1], conj_tensors[0], [0, 2], [1, 2]) # (i0, i0')
    B = np.einsum(tensors[l], [0, 1], conj_tensors[l], [0, 2], [1, 2]) # (il, il')
    for i in range(edge):
        A_next_block = np.einsum(tensors[i + 1], [0, 1, 2], conj_tensors[i + 1], [0, 3, 4], [1, 3, 2, 4]) #(i0, i0', i1, i1')
        A = np.einsum(A, [0, 1], A_next_block, [0, 1, 2, 3], [2, 3])
    for i in range(l - edge - 1):
        B_next_block = np.einsum(tensors[l - 1 - i], [0, 1, 2], conj_tensors[l - 1 - i], [0, 3, 4], [1, 3, 2, 4])  # (i(l-1), i(l-1)',il, il')
        B = np.einsum(B, [0, 1], B_next_block, [2, 3, 0, 1], [2, 3])
        # A = (ie, ie')
        # B = (i(e+1), i(e+1)')
    return A, B


def Accordion(Ti, Tj, P, D_max):
    L = cp.deepcopy(Ti[0])
    R = cp.deepcopy(Tj[0])
    l = len(L.shape)
    r = len(R.shape)

    if l == 2 and r == 3:
        A = np.einsum(L, [0, 1], P, [1, 3], [0, 3])  # (i, Ek)
        theta = np.einsum(A, [0, 1], R, [2, 1, 3], [0, 2, 3])  # (i, j, d)
    if l == 3 and r == 2:
        A = np.einsum(L, [0, 1, 2], P, [1, 3], [0, 3, 2])  # (i, Ek, d)
        theta = np.einsum(A, [0, 1, 2], R, [3, 1], [2, 0, 3])  # (d, i, j)
    if l == 2 and r == 2:
        A = np.einsum(L, [0, 1], P, [1, 3], [0, 3])  # (i, Ek)
        theta = np.einsum(A, [0, 1], R, [3, 1], [0, 3])  # (i, j)
    if l == 3 and r == 3:
        A = np.einsum(L, [0, 1, 2], P, [1, 3], [0, 3, 2])  # (i, Ek, Q1)
        theta = np.einsum(A, [0, 1, 2], R, [3, 1, 4], [2, 0, 3, 4])  # (Q1, i, j, Q2)

    R_tild, lamda_k, L_tild = svd(theta, range(l - 1), range(l - 1, r + l - 2), keep_s='yes', max_eigen_num=D_max)

    # reshaping R_tild and L_tild back
    if l == 2:
        R_tild_new_shape = [Ti[0].shape[0], R_tild.shape[1]]  # (i, D')
        R_transpose = [0, 1]
    if l == 3:
        R_tild_new_shape = [Ti[0].shape[2], Ti[0].shape[0], R_tild.shape[1]]  # (d, i, D')
        R_transpose = [1, 2, 0]
    if r == 2:
        L_tild_new_shape = [L_tild.shape[0], Tj[0].shape[0]]  # (D', j)
        L_transpose = [1, 0]
    if r == 3:
        L_tild_new_shape = [L_tild.shape[0], Tj[0].shape[0], Tj[0].shape[2]]  # (D', j, d)
        L_transpose = [1, 0, 2]

    R_tild = np.reshape(R_tild, R_tild_new_shape)
    Ti[0] = np.transpose(R_tild, R_transpose)  # (i, D', ...)
    L_tild = np.reshape(L_tild, L_tild_new_shape)
    Tj[0] = np.transpose(L_tild, L_transpose)  # (j, D', ...)

    return Ti, Tj, lamda_k
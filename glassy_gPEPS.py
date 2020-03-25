import numpy as np
import ncon as ncon
import copy as cp
from scipy import linalg


"""
    A module for the function gPEPS which preform the simple update algorithm over a given Tensor Network 
    as specified in the paper https://arxiv.org/abs/1808.00680 by Roman Orus et al.
"""


def simple_update(TT, LL, dt, Jk, h, Opi, Opj, Op_field, imat, smat, D_max):

    """
    :param TT: list of tensors in the tensor network TT = [T1, T2, T3, ..., Tp]
    :param LL: list of lists of the lambdas LL = [L1, L2, ..., Ls]
    :param Uij: The unitary matrix for the Imaginary Time Evolution (ITE)
    :param imat: The incidence matrix which indicates which tensor connect to which edge (as indicated in the paper)
    :param smat: The structure matrix which indicates which leg of every tensor is connected to which edge
    :param D_max: maximal virtual dimension
    :return: updated tensors list TT and updated lambda tuple of lists LL

    The SU preformed here is following step by step after the algorithm specified in the paper above. The only change I
    made is using a SVD decomposition and not QR decomposition in step (d) although I am living a commented QR
    decomposition code just in case.
    """
    n, m = np.shape(imat)
    for Ek in range(m):

        lamda_k = LL[Ek]

        ## (a) Find tensors Ti, Tj and their corresponding legs connected along edge Ek.
        Ti, Tj = get_tensors(Ek, TT, smat, imat)

        # collecting all neighboring (edges, dimensions) without the Ek (edge, dimension)
        i_dim, j_dim = get_edges(Ek, smat, imat)

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
            Tj[0] = np.einsum(Tj[0], range(len(Tj[0].shape)), LL[jedges[ii]], [jlegs[ii]], range(len(Tj[0].shape)))



        ## (b) Absorb bond vectors (lambdas) to all Em != Ek of Ti, Tj tensors
        #Ti = absorb_edges(Ti, i_dim, LL)
        #Tj = absorb_edges(Tj, j_dim, LL)

        # permuting the Ek leg of tensors i and j into the 1'st dimension
        Ti = dim_perm(Ti)
        Tj = dim_perm(Tj)

        ## (c) Group all virtual legs Em!=Ek to form Pl, Pr MPS tensors
        Pl = rankN_to_rank3(cp.deepcopy(Ti[0]))
        Pr = rankN_to_rank3(cp.deepcopy(Tj[0]))

        '''
        # experimenting with QR instead of SVD
        Q1, R = np.linalg.qr(np.transpose(np.reshape(Pl, [Pl.shape[0] * Pl.shape[1], np.prod(Pl.shape[2:])])))
        Q2, L = np.linalg.qr(np.transpose(np.reshape(Pr, [Pr.shape[0] * Pr.shape[1], np.prod(Pr.shape[2:])])))
        Q1 = np.transpose(Q1)
        Q2 = np.transpose(Q2)
        R = np.transpose(R)
        L = np.transpose(L)

        '''
        ## (d) SVD decomposing of Pl, Pr to obtain Q1, R and Q2, L sub-tensors, respectively
        R, sr, Q1 = svd(Pl, [0, 1], [2], keep_s='yes')
        L, sl, Q2 = svd(Pr, [0, 1], [2], keep_s='yes')
        R = R.dot(np.diag(sr))
        L = L.dot(np.diag(sl))

        # reshaping R and L into rank 3 tensors with shape (physical_dim, Ek_dim, Q(1/2).shape[0])
        i_physical_dim = Ti[0].shape[0]
        j_physical_dim = Tj[0].shape[0]
        R = rank2_to_rank3(R, i_physical_dim)  # (i, Ek, Q1) (following the dimensions)
        L = rank2_to_rank3(L, j_physical_dim)  # (j, Ek, Q2)

        ## (e) Contract the ITE gate Uij, with R, L, and lambda_k to form theta tensor.
        theta = imaginary_time_evolution(R, L, lamda_k, Ek, dt, Jk, h, Opi, Opj, Op_field)  # (Q1, i', j', Q2)

        ## (f) Obtain R', L', lambda'_k tensors by applying an SVD to theta
        # and trancating R', lamda_k', L' while keeping only D_max eigenvalues and eigenvectors
        R_tild, lamda_k_tild, L_tild = svd(theta, [0, 1], [2, 3], keep_s='yes', max_eigen_num=D_max)
        # (Q1 * i', D_max) # (D_max, D_max) # (D_max, j' * Q2)

        # reshaping R_tild and L_tild back to rank 3 tensor
        R_tild = np.reshape(R_tild, (Q1.shape[0], i_physical_dim, R_tild.shape[1]))  # (Q1, i', D_max)
        R_tild = np.transpose(R_tild, [1, 2, 0])  # (i', D_max, Q1)
        L_tild = np.reshape(L_tild, (L_tild.shape[0], j_physical_dim, Q2.shape[0]))  # (D_max, j', Q2)
        L_tild = np.transpose(L_tild, [1, 0, 2])  # (j', D_max, Q2)


        ## (g) Glue back the R', L', sub-tensors to Q1, Q2, respectively, to form updated tensors P'l, P'r.
        Pl_prime = np.einsum('ijk,kl->ijl', R_tild, Q1)
        Pr_prime = np.einsum('ijk,kl->ijl', L_tild, Q2)

        ## (h) Reshape back the P`l, P`r to the original rank-(z + 1) tensors Ti, Tj
        Ti_new_shape = list(Ti[0].shape)
        Ti_new_shape[1] = len(lamda_k_tild)
        Tj_new_shape = list(Tj[0].shape)
        Tj_new_shape[1] = len(lamda_k_tild)
        Ti[0] = rank3_to_rankN(Pl_prime, Ti_new_shape)
        Tj[0] = rank3_to_rankN(Pr_prime, Tj_new_shape)

        # permuting back the legs of Ti and Tj
        Ti = dim_perm(Ti)
        Tj = dim_perm(Tj)

        ## (i) Remove bond matrices lambda_m from virtual legs m != Ek to obtain the updated tensors Ti~, Tj~.
        #Ti = remove_edges(Ti, i_dim, LL)
        #Tj = remove_edges(Tj, j_dim, LL)
        for ii in range(len(iedges)):
            Ti[0] = np.einsum(Ti[0], range(len(Ti[0].shape)), LL[iedges[ii]] ** (-1), [ilegs[ii]], range(len(Ti[0].shape)))
            Tj[0] = np.einsum(Tj[0], range(len(Tj[0].shape)), LL[jedges[ii]] ** (-1), [jlegs[ii]], range(len(Tj[0].shape)))


        # Normalize and save new Ti Tj and lambda_k
        TT[Ti[1][0]] = cp.deepcopy(Ti[0] / tensor_normalization(Ti[0]))
        TT[Tj[1][0]] = cp.deepcopy(Tj[0] / tensor_normalization(Tj[0]))
        LL[Ek] = cp.deepcopy(lamda_k_tild / np.sum(lamda_k_tild))

    return cp.deepcopy(TT), cp.deepcopy(LL)


def get_tensors(edge, tensors, structure_matrix, incidence_matrix):
    tidx = np.nonzero(incidence_matrix[:, edge])[0]
    tdim = structure_matrix[tidx, edge]
    Ti = [cp.deepcopy(tensors[tidx[0]]), [tidx[0], 'tensor_number'], [tdim[0], 'tensor_Ek_leg']]
    Tj = [cp.deepcopy(tensors[tidx[1]]), [tidx[1], 'tensor_number'], [tdim[1], 'tensor_Ek_leg']]
    return Ti, Tj


def get_conjugate_tensors(edge, tensors, structure_matrix, incidence_matrix):
    tidx = np.nonzero(incidence_matrix[:, edge])[0]
    tdim = structure_matrix[tidx, edge]
    Ti = [cp.deepcopy(np.conj(tensors[tidx[0]])), [tidx[0], 'tensor_number'], [tdim[0], 'tensor_Ek_leg']]
    Tj = [cp.deepcopy(np.conj(tensors[tidx[1]])), [tidx[1], 'tensor_number'], [tdim[1], 'tensor_Ek_leg']]
    return Ti, Tj


def get_edges(edge, structure_matrix, incidence_matrix):
    tidx = np.nonzero(incidence_matrix[:, edge])[0]
    i_dim = [list(np.nonzero(incidence_matrix[tidx[0], :])[0]), list(structure_matrix[tidx[0], np.nonzero(incidence_matrix[tidx[0], :])[0]])]
    j_dim = [list(np.nonzero(incidence_matrix[tidx[1], :])[0]), list(structure_matrix[tidx[1], np.nonzero(incidence_matrix[tidx[1], :])[0]])]
    # removing the Ek edge and leg
    i_dim[0].remove(edge)
    i_dim[1].remove(structure_matrix[tidx[0], edge])
    j_dim[0].remove(edge)
    j_dim[1].remove(structure_matrix[tidx[1], edge])
    return i_dim, j_dim


def absorb_edges(tensor, edges_dim, bond_vectors):
    for i in range(len(edges_dim[0])):
        tensor[0] = np.einsum(tensor[0], range(len(tensor[0].shape)), bond_vectors[edges_dim[0][i]], [edges_dim[1][i]], range(len(tensor[0].shape)))
    #print(tensor_normalization(tensor[0]))
    return tensor


def remove_edges(tensor, edges_dim, bond_vectors):
    for i in range(len(edges_dim[0])):
        tensor[0] = np.einsum(tensor[0], range(len(tensor[0].shape)), bond_vectors[edges_dim[0][i]] ** (-1), [edges_dim[1][i]], range(len(tensor[0].shape)))
    #print(tensor_normalization(tensor[0]))
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
        rank = len(tensor.shape)
        raise IndexError('expecting tensor rank N>=3. instead got tensor of rank = ' + str(rank))
    shape = np.array(cp.copy(tensor.shape))
    new_shape = [shape[0], shape[1], np.prod(shape[2:])]
    tensor = tensor.reshape(new_shape)
    return tensor


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
    shape = np.array(cp.copy(tensor.shape))
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

def imaginary_time_evolution(left_tensor, right_tensor, bond_vector, Ek, dt, Jk, h, Opi, Opj, Op_field):
    # applying ITE and returning a rank 4 tensor with physical dimensions, i' and j' at (Q1, i', j', Q2)
    # the indices of the unitary_time_op should be (i, j, i', j')
    p = Op_field.shape[0]
    hij = -Jk[Ek] * np.kron(Opi, Opj) - 0.25 * h * (np.kron(np.eye(p), Op_field) + np.kron(Op_field, np.eye(p)))
    unitary_time_op = np.reshape(linalg.expm(-dt * hij), [p, p, p, p])
    bond_matrix = np.diag(bond_vector)
    A = np.einsum(left_tensor, [0, 1, 2], bond_matrix, [1, 3], [0, 3, 2])  # (i, Ek, Q1)
    A = np.einsum(A, [0, 1, 2], right_tensor, [3, 1, 4], [2, 0, 3, 4])  # (Q1, i, j, Q2)
    theta = np.einsum(A, [0, 1, 2, 3], unitary_time_op, [1, 2, 4, 5], [0, 4, 5, 3])  # (Q1, i', j', Q2)
    return theta


def tensor_normalization(T):
    T_conj = np.conj(cp.deepcopy(T))
    idx = range(len(T.shape))
    norm = np.einsum(T, idx, T_conj, idx)
    return np.sqrt(norm)


def single_tensor_expectation(tensor_idx, TT, LL, imat, smat, Oi):
    env_edges = np.nonzero(imat[tensor_idx, :])[0]
    env_legs = smat[tensor_idx, env_edges]

    ## Absorbing the environment
    #T = absorb_edges(cp.deepcopy(TT[tensor_idx]), [env_edges, env_legs], LL)
    #T_conj = absorb_edges(cp.deepcopy(np.conj(TT[tensor_idx])), [env_edges, env_legs], LL)

    T = cp.deepcopy(TT[tensor_idx])
    T_conj = cp.deepcopy(np.conj(TT[tensor_idx]))

    ## absorb its environment
    for j in range(len(env_edges)):
        T = np.einsum(T, range(len(T.shape)), LL[env_edges[j]], [env_legs[j]], range(len(T.shape)))
        T_conj = np.einsum(T_conj, range(len(T_conj.shape)), LL[env_edges[j]], [env_legs[j]], range(len(T_conj.shape)))

    #T /= T.max()
    #T_conj /= T_conj.max()

    T_idx = range(len(T.shape))
    T_conj_idx = range(len(T_conj.shape))
    T_conj_idx[0] = len(T_conj.shape)
    operator_idx = [T_conj_idx[0], T_idx[0]]
    expectation = ncon.ncon([T, T_conj, Oi], [T_idx, T_conj_idx, operator_idx])
    normalization = site_norm(tensor_idx, TT, LL, imat, smat)
    #print('e = ', expectation)
    #print('n = ', normalization)
    return expectation / normalization


def magnetization(TT, LL, imat, smat, Oi):
    # calculating the average magnetization per site
    magnetization = 0
    tensors_indices = range(len(TT))
    for i in tensors_indices:
        magnetization += single_tensor_expectation(i, TT, LL, imat, smat, Oi)
    magnetization /= len(TT)
    return magnetization


def site_norm(tensor_idx, tensors_list, lamdas_list, imat, smat):
    env_edges = np.nonzero(imat[tensor_idx, :])[0]
    env_legs = smat[tensor_idx, env_edges]

    ## Absorbing the environment
    #T = absorb_edges(cp.deepcopy(tensors_list[tensor_idx]), [env_edges, env_legs], lamdas_list)
    #T_conj = absorb_edges(cp.deepcopy(np.conj(tensors_list[tensor_idx])), [env_edges, env_legs], lamdas_list)

    T = cp.deepcopy(tensors_list[tensor_idx])
    T_conj = cp.deepcopy(np.conj(tensors_list[tensor_idx]))

    ## absorb its environment
    for j in range(len(env_edges)):
        T = np.einsum(T, range(len(T.shape)), lamdas_list[env_edges[j]], [env_legs[j]], range(len(T.shape)))
        T_conj = np.einsum(T_conj, range(len(T_conj.shape)), lamdas_list[env_edges[j]], [env_legs[j]], range(len(T_conj.shape)))

    #T /= T.max()
    #T_conj /= T_conj.max()

    T_idx = range(len(T.shape))
    T_conj_idx = range(len(T_conj.shape))
    normalization = np.einsum(T, T_idx, T_conj, T_conj_idx)
    return normalization

def two_site_expectation(Ek, TT, LL, imat, smat, Oij):
    # calculating the two site normalized expectation given a mutual edge Ek of those two sites (tensors) and the operator Oij
    lamda_k = cp.deepcopy(LL[Ek])

    ## (a) Find tensors Ti, Tj and their corresponding legs connected along edge Ek.
    Ti, Tj = get_tensors(Ek, TT, smat, imat)
    Ti_conj, Tj_conj = get_tensors(Ek, TT, smat, imat)

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
    #print('2 sites norm = ', two_site_norm)
    #print('2 sites energy = ', two_site_expec)
    two_site_expec /= two_site_norm
    return two_site_expec


def energy_per_site(TT, LL, imat, smat, Jk, h, Opi, Opj, Op_field):
    # calculating the normalized energy per site(tensor)
    p = Op_field.shape[0]
    energy = 0
    n, m = np.shape(imat)
    for Ek in range(m):
        Oij = np.reshape(-Jk[Ek] * np.kron(Opi, Opj) - 0.25 * h * (np.kron(np.eye(p), Op_field) + np.kron(Op_field, np.eye(p))), (p, p, p, p))
        energy += two_site_expectation(Ek, TT, LL, imat, smat, Oij)
    energy /= n
    return energy


def gauge_fix(TT, LL, imat, smat):
    n, m = np.shape(imat)
    for Ek in range(m):

        lamda_k = LL[Ek]

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


        ## Gauge fixing
        # i environment
        i_idx = range(len(Ti[0].shape))
        i_conj_idx = range(len(Ti[0].shape))
        i_conj_idx[Ti_conj[2][0]] = len(Ti[0].shape)
        Mi = np.einsum(Ti[0], i_idx, Ti_conj[0], i_conj_idx,
                       [i_idx[Ti_conj[2][0]], i_conj_idx[Ti_conj[2][0]]])  # (Ek, Ek*)

        # j environment
        j_idx = range(len(Tj[0].shape))
        j_conj_idx = range(len(Tj[0].shape))
        j_conj_idx[Tj_conj[2][0]] = len(Tj[0].shape)
        Mj = np.einsum(Tj[0], j_idx, Tj_conj[0], j_conj_idx,
                       [j_idx[Tj_conj[2][0]], j_conj_idx[Tj_conj[2][0]]])  # (Ek, Ek*)

        # Environment diagonalization
        di, ui = np.linalg.eig(Mi)
        dj, uj = np.linalg.eig(Mj)

        # contruction
        lamda_k_prime = np.matmul(np.matmul(np.conj(np.transpose(ui)), np.diag(lamda_k)), uj)
        lamda_k_prime = np.matmul(np.diag(np.sqrt(di)), np.matmul(lamda_k_prime, np.diag(np.sqrt(dj))))

        # SVD
        wi, lamda_k_tild, wj = np.linalg.svd(lamda_k_prime)
        # lamda_k_tild /= np.sum(lamda_k_tild)

        # x and y construction
        x = np.matmul(np.matmul(np.conj(np.transpose(wi)), np.diag(np.sqrt(di))), np.conj(np.transpose(ui)))
        y = np.matmul(np.matmul(uj, np.diag(np.sqrt(dj))), np.transpose(wj))

        # fixing Ti and Tj
        Ti_idx_old = range(len(Ti[0].shape))
        Ti_idx_new = range(len(Ti[0].shape))
        Ti_idx_new[Ti[2][0]] = len(Ti_idx_old)
        Tj_idx_old = range(len(Tj[0].shape))
        Tj_idx_new = range(len(Tj[0].shape))
        Tj_idx_new[Tj[2][0]] = len(Tj_idx_old)
        Ti_fixed = np.einsum(Ti[0], Ti_idx_old, np.linalg.pinv(x), [Ti_idx_old[Ti[2][0]], len(Ti_idx_old)],
                             Ti_idx_new)
        Tj_fixed = np.einsum(Tj[0], Tj_idx_old, np.linalg.pinv(y), [Tj_idx_old[Tj[2][0]], len(Tj_idx_old)],
                             Tj_idx_new)

        TT[Ti[1][0]] = cp.deepcopy(Ti_fixed)
        TT[Tj[1][0]] = cp.deepcopy(Tj_fixed)
        LL[Ek] = cp.deepcopy(lamda_k_tild / np.sum(lamda_k_tild))
    return TT, LL



def gauge_fix1(TT, LL, imat, smat):
    n, m = np.shape(imat)
    TT_new = cp.deepcopy(TT)
    LL_new = cp.deepcopy(LL)
    for Ek in range(m):

        ## (a) Find tensors Ti, Tj and their corresponding legs connected along edge Ek.
        tidx = np.nonzero(imat[:, Ek])[0]
        tdim = smat[tidx, Ek]
        Ti = [TT[tidx[0]], tdim[0]]
        Tj = [TT[tidx[1]], tdim[1]]
        lamda_k = LL[Ek]

        # collecting all neighboring (edges, legs)
        Ti_alldim = [list(np.nonzero(imat[tidx[0], :])[0]), list(smat[tidx[0], np.nonzero(imat[tidx[0], :])[0]])]
        Tj_alldim = [list(np.nonzero(imat[tidx[1], :])[0]), list(smat[tidx[1], np.nonzero(imat[tidx[1], :])[0]])]

        # removing the Ek edge and leg
        Ti_alldim[0].remove(Ek)
        Ti_alldim[1].remove(smat[tidx[0], Ek])
        Tj_alldim[0].remove(Ek)
        Tj_alldim[1].remove(smat[tidx[1], Ek])

        # collecting neighboring lamdas
        Ti_lamdas = [LL[Em] for Em in Ti_alldim[0]]
        Tj_lamdas = [LL[Em] for Em in Tj_alldim[0]]

        ## (b) Absorb bond matrices (lambdas) to all Em != Ek of Ti, Tj tensors
        for Em in range(len(Ti_lamdas)):
            Ti[0] = np.einsum(Ti[0], range(len(Ti[0].shape)), Ti_lamdas[Em], [Ti_alldim[1][Em]],
                              range(len(Ti[0].shape)))
        for Em in range(len(Tj_lamdas)):
            Tj[0] = np.einsum(Tj[0], range(len(Tj[0].shape)), Tj_lamdas[Em], [Tj_alldim[1][Em]],
                              range(len(Tj[0].shape)))

        ## Gauge fixing
        # i environment
        i_leg = tdim[0]
        j_leg = tdim[1]
        i_idx = range(len(Ti[0].shape))
        i_conj_idx = range(len(Ti[0].shape))
        i_conj_idx[i_leg] = len(Ti[0].shape)
        Mi = np.einsum(Ti[0], i_idx, np.conj(Ti[0]), i_conj_idx,
                       [i_idx[i_leg], i_conj_idx[i_leg]])

        # j environment
        j_idx = range(len(Tj[0].shape))
        j_conj_idx = range(len(Tj[0].shape))
        j_conj_idx[j_leg] = len(Tj[0].shape)
        Mj = np.einsum(Tj[0], j_idx, np.conj(Tj[0]), j_conj_idx,
                       [j_idx[j_leg], j_conj_idx[j_leg]])

        # Environment diagonalization
        di, ui = np.linalg.eig(Mi)
        dj, uj = np.linalg.eig(Mj)

        # contruction
        lamda_k_prime = np.matmul(np.matmul(np.conj(np.transpose(ui)), np.diag(lamda_k)), uj)
        lamda_k_prime = np.matmul(np.diag(np.sqrt(di)), np.matmul(lamda_k_prime, np.diag(np.sqrt(dj))))

        # SVD
        wi, lamda_k_tild, wj = np.linalg.svd(lamda_k_prime)
        lamda_k_tild /= np.sum(lamda_k_tild)

        # x and y costruction
        x = np.matmul(np.matmul(np.conj(np.transpose(wi)), np.diag(np.sqrt(di))), np.conj(np.transpose(ui)))
        y = np.matmul(np.matmul(uj, np.diag(np.sqrt(dj))), wj)

        # fixing Ti and Tj
        Ti_idx_old = range(len(TT[tidx[0]].shape))
        Ti_idx_new = range(len(TT[tidx[0]].shape))
        Ti_idx_new[i_leg] = len(Ti_idx_old)
        Tj_idx_old = range(len(TT[tidx[1]].shape))
        Tj_idx_new = range(len(TT[tidx[1]].shape))
        Tj_idx_new[j_leg] = len(Tj_idx_old)
        Ti_fixed = np.einsum(TT[tidx[0]], Ti_idx_old, np.linalg.pinv(x), [Ti_idx_old[i_leg], len(Ti_idx_old)], Ti_idx_new)
        Tj_fixed = np.einsum(TT[tidx[1]], Tj_idx_old, np.linalg.pinv(y), [Tj_idx_old[j_leg], len(Tj_idx_old)], Tj_idx_new)

        TT_new[tidx[0]] = cp.copy(Ti_fixed)
        TT_new[tidx[1]] = cp.copy(Tj_fixed)
        LL_new[Ek] = cp.copy(lamda_k_tild / np.sum(lamda_k_tild))

    return TT_new, LL_new

def trace_distance(a, b):
    # returns the trace distance between the two density matrices a & b
    # d = 0.5 * norm(a - b)
    eigenvalues = np.linalg.eigvals(a - b)
    d = 0.5 * np.sum(np.abs(eigenvalues))
    return d


def tensor_reduced_dm(tensor_idx, TT, LL, smat, imat):
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
    normalization = site_norm(tensor_idx, TT, LL, imat, smat)

    return reduced_dm / normalization

def absorb_all_bond_vectors(TT, LL, smat):
    n = len(TT)
    for i in range(n):
        edges = np.nonzero(smat[i, :])[0]
        legs = smat[i, edges]
        for j in range(len(edges)):
            TT[i] = np.einsum(TT[i], range(len(TT[i].shape)), np.sqrt(LL[edges[j]]), [legs[j]], range(len(TT[i].shape)))
    return TT
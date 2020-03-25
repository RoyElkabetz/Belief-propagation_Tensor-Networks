import numpy as np
import copy as cp
import StructureMatrixGenerator as tnf


def ncon_list_generator(TT, LL, smat, O, spin):
    TT = cp.deepcopy(TT)
    LL = cp.deepcopy(LL)

    T_list = []
    idx_list = []
    n, m = smat.shape
    spins_idx = range(2 * m + 1, 2 * m + 1 + n)
    Oidx = [spins_idx[spin], spins_idx[-1] + 1]

    for i in range(n):

        ## pick a tensor T, T*
        T = cp.copy(TT[i])
        Tstar = np.conj(cp.copy(TT[i]))
        edges = np.nonzero(smat[i, :])[0]
        legs = smat[i, edges]
        Tidx = np.zeros((len(T.shape)), dtype=int)
        Tstaridx = np.zeros((len(Tstar.shape)), dtype=int)

        ## creat T, T* indices
        if i == spin:
            Tidx[0] = Oidx[0]
            Tstaridx[0] = Oidx[1]
        else:
            Tidx[0] = spins_idx[i]
            Tstaridx[0] = spins_idx[i]

        ## absorb its environment
        for j in range(len(edges)):
            T = np.einsum(T, range(len(T.shape)), np.sqrt(LL[edges[j]]), [legs[j]], range(len(T.shape)))
            Tstar = np.einsum(Tstar, range(len(Tstar.shape)), np.sqrt(LL[edges[j]]), [legs[j]], range(len(Tstar.shape)))
            Tidx[legs[j]] = edges[j] + 1
            Tstaridx[legs[j]] = edges[j] + 1 + m

        ## add to lists
        T_list.append(cp.copy(T))
        idx_list.append(cp.copy(Tidx))

        if i == spin:
            T_list.append(cp.copy(O))
            idx_list.append(cp.copy(Oidx))

        T_list.append(cp.copy(Tstar))
        idx_list.append(cp.copy(Tstaridx))

    return T_list, idx_list


def ncon_list_generator_reduced_dm(TT, LL, smat, spin):
    TT = cp.deepcopy(TT)
    LL = cp.deepcopy(LL)

    T_list = []
    idx_list = []
    n, m = smat.shape
    spins_idx = range(2 * m + 1, 2 * m + 1 + n)

    #Oidx = [spins_idx[spin], spins_idx[-1] + 1]

    for i in range(n):

        ## pick a tensor T, T*
        T = cp.copy(TT[i])
        Tstar = np.conj(cp.copy(TT[i]))
        edges = np.nonzero(smat[i, :])[0]
        legs = smat[i, edges]
        Tidx = np.zeros((len(T.shape)), dtype=int)
        Tstaridx = np.zeros((len(Tstar.shape)), dtype=int)


        ## creat T, T* indices
        if i == spin:
            Tidx[0] = -1
            Tstaridx[0] = -2
        else:
            Tidx[0] = spins_idx[i]
            Tstaridx[0] = spins_idx[i]

        ## absorb its environment
        for j in range(len(edges)):
            T = np.einsum(T, range(len(T.shape)), np.sqrt(LL[edges[j]]), [legs[j]], range(len(T.shape)))
            Tstar = np.einsum(Tstar, range(len(Tstar.shape)), np.sqrt(LL[edges[j]]), [legs[j]], range(len(Tstar.shape)))
            Tidx[legs[j]] = edges[j] + 1
            Tstaridx[legs[j]] = edges[j] + 1 + m

        ## add to lists
        T_list.append(cp.copy(T))
        idx_list.append(cp.copy(Tidx))
        T_list.append(cp.copy(Tstar))
        idx_list.append(cp.copy(Tstaridx))

    return T_list, idx_list


def ncon_list_generator_for_BPerror(TT1, LL1, TT2, LL2, smat):
    TT1 = cp.deepcopy(TT1)
    LL1 = cp.deepcopy(LL1)
    TT2 = cp.deepcopy(TT2)
    LL2 = cp.deepcopy(LL2)

    T_list = []
    idx_list = []
    n, m = smat.shape
    spins_idx = range(2 * m + 1, 2 * m + 1 + n)
    for i in range(n):

        ## pick a tensor T, T*
        T = cp.copy(TT1[i])
        Tstar = np.conj(cp.copy(TT2[i]))
        edges = np.nonzero(smat[i, :])[0]
        legs = smat[i, edges]
        Tidx = np.zeros((len(T.shape)), dtype=int)
        Tstaridx = np.zeros((len(Tstar.shape)), dtype=int)
        Tidx[0] = spins_idx[i]
        Tstaridx[0] = spins_idx[i]

        ## absorb its environment
        for j in range(len(edges)):
            T = np.einsum(T, range(len(T.shape)), np.sqrt(LL1[edges[j]]), [legs[j]], range(len(T.shape)))
            Tstar = np.einsum(Tstar, range(len(Tstar.shape)), np.sqrt(LL2[edges[j]]), [legs[j]], range(len(Tstar.shape)))
            Tidx[legs[j]] = edges[j] + 1
            Tstaridx[legs[j]] = edges[j] + 1 + m


        ## add to lists
        T_list.append(cp.copy(T))
        idx_list.append(cp.copy(Tidx))
        T_list.append(cp.copy(Tstar))
        idx_list.append(cp.copy(Tstaridx))
    return T_list, idx_list


def ncon_list_generator_two_site_exact_expectation_peps(TT, TTstar, smat, edge, operator):
    TT = cp.deepcopy(TT)
    TTstar = cp.deepcopy(TTstar)

    T_list = []
    idx_list = []
    n, m = smat.shape
    spins_idx = range(2 * m + 1, 2 * m + 1 + n)

    ## fix operator legs
    tensors_indices = np.nonzero(smat[:, edge])[0]
    operator_idx = [spins_idx[tensors_indices[0]], spins_idx[tensors_indices[1]], 1000, 1001]  # [i, j, i', j']

    for i in range(n):
        if i == tensors_indices[0]:

            ## pick a tensor T, T*
            T = cp.copy(TT[i])
            Tstar = cp.copy(TTstar[i])
            edges = np.nonzero(smat[i, :])[0]
            legs = smat[i, edges]
            Tidx = np.zeros((len(T.shape)), dtype=int)
            Tstaridx = np.zeros((len(Tstar.shape)), dtype=int)
            Tidx[0] = spins_idx[i]
            Tstaridx[0] = operator_idx[2]

        elif i == tensors_indices[1]:
            ## pick a tensor T, T*
            T = cp.copy(TT[i])
            Tstar = cp.copy(TTstar[i])
            edges = np.nonzero(smat[i, :])[0]
            legs = smat[i, edges]
            Tidx = np.zeros((len(T.shape)), dtype=int)
            Tstaridx = np.zeros((len(Tstar.shape)), dtype=int)
            Tidx[0] = spins_idx[i]
            Tstaridx[0] = operator_idx[3]

        else:
            ## pick a tensor T, T*
            T = cp.copy(TT[i])
            Tstar = cp.copy(TTstar[i])
            edges = np.nonzero(smat[i, :])[0]
            legs = smat[i, edges]
            Tidx = np.zeros((len(T.shape)), dtype=int)
            Tstaridx = np.zeros((len(Tstar.shape)), dtype=int)
            Tidx[0] = spins_idx[i]
            Tstaridx[0] = spins_idx[i]

        ## arange legs indices
        for j in range(len(edges)):
            Tidx[legs[j]] = edges[j] + 1
            Tstaridx[legs[j]] = edges[j] + 1 + m

        ## add to lists
        T_list.append(cp.copy(T))
        idx_list.append(cp.copy(Tidx))
        T_list.append(cp.copy(Tstar))
        idx_list.append(cp.copy(Tstaridx))

    ## add operator to list
    T_list.append(operator)
    idx_list.append(operator_idx)

    return T_list, idx_list


def ncon_list_generator_braket_peps(TT1, TT2, smat):
    TT1 = cp.deepcopy(TT1)
    TT2 = cp.deepcopy(TT2)

    T_list = []
    idx_list = []
    n, m = smat.shape
    spins_idx = range(2 * m + 1, 2 * m + 1 + n)
    for i in range(n):

        ## pick a tensor T, T*
        T = cp.copy(TT1[i])
        Tstar = cp.copy(TT2[i])
        edges = np.nonzero(smat[i, :])[0]
        legs = smat[i, edges]
        Tidx = np.zeros((len(T.shape)), dtype=int)
        Tstaridx = np.zeros((len(Tstar.shape)), dtype=int)
        Tidx[0] = spins_idx[i]
        Tstaridx[0] = spins_idx[i]

        ## arange legs indices
        for j in range(len(edges)):
            Tidx[legs[j]] = edges[j] + 1
            Tstaridx[legs[j]] = edges[j] + 1 + m


        ## add to lists
        T_list.append(cp.copy(T))
        idx_list.append(cp.copy(Tidx))
        T_list.append(cp.copy(Tstar))
        idx_list.append(cp.copy(Tstaridx))
    return T_list, idx_list


def ncon_list_generator_two_site_exact_expectation_mps(TT, TTstar, smat, edge, operator):
    TT = cp.deepcopy(TT)
    TTstar = cp.deepcopy(TTstar)

    T_list = []
    idx_list = []
    n, m = smat.shape
    spins_idx = range(2 * m + 1, 2 * m + 1 + n)

    ## fix operator legs
    tensors_indices = np.nonzero(smat[:, edge])[0]
    if edge == (m - 1):
        tensors_indices = np.flip(tensors_indices, axis=0)
    operator_idx = [spins_idx[tensors_indices[0]], spins_idx[tensors_indices[1]], 1000, 1001]  # [i, j, i', j']

    for i in range(n):
        if i == tensors_indices[0]:

            ## pick a tensor T, T*
            T = cp.copy(TT[i])
            Tstar = cp.copy(TTstar[i])
            edges = np.nonzero(smat[i, :])[0]
            legs = smat[i, edges]
            Tidx = np.zeros((len(T.shape)), dtype=int)
            Tstaridx = np.zeros((len(Tstar.shape)), dtype=int)
            Tidx[0] = spins_idx[i]
            Tstaridx[0] = operator_idx[2]

        elif i == tensors_indices[1]:
            ## pick a tensor T, T*
            T = cp.copy(TT[i])
            Tstar = cp.copy(TTstar[i])
            edges = np.nonzero(smat[i, :])[0]
            legs = smat[i, edges]
            Tidx = np.zeros((len(T.shape)), dtype=int)
            Tstaridx = np.zeros((len(Tstar.shape)), dtype=int)
            Tidx[0] = spins_idx[i]
            Tstaridx[0] = operator_idx[3]

        else:
            ## pick a tensor T, T*
            T = cp.copy(TT[i])
            Tstar = cp.copy(TTstar[i])
            edges = np.nonzero(smat[i, :])[0]
            legs = smat[i, edges]
            Tidx = np.zeros((len(T.shape)), dtype=int)
            Tstaridx = np.zeros((len(Tstar.shape)), dtype=int)
            Tidx[0] = spins_idx[i]
            Tstaridx[0] = spins_idx[i]

        ## arange legs indices
        for j in range(len(edges)):
            Tidx[legs[j]] = edges[j] + 1
            Tstaridx[legs[j]] = edges[j] + 1 + m

        ## add to lists
        T_list.append(cp.copy(T))
        idx_list.append(cp.copy(Tidx))
        T_list.append(cp.copy(Tstar))
        idx_list.append(cp.copy(Tstaridx))

    ## add operator to list
    T_list.append(operator)
    idx_list.append(operator_idx)

    return T_list, idx_list


def ncon_list_generator_braket_mps(TT, TTstar, smat):
    TT = cp.deepcopy(TT)
    TTstar = cp.deepcopy(TTstar)

    T_list = []
    idx_list = []
    n, m = smat.shape
    spins_idx = range(2 * m + 1, 2 * m + 1 + n)
    for i in range(n):

        ## pick a tensor T, T*
        T = cp.copy(TT[i])
        Tstar = cp.copy(TTstar[i])
        edges = np.nonzero(smat[i, :])[0]
        legs = smat[i, edges]
        Tidx = np.zeros((len(T.shape)), dtype=int)
        Tstaridx = np.zeros((len(Tstar.shape)), dtype=int)
        Tidx[0] = spins_idx[i]
        Tstaridx[0] = spins_idx[i]

        ## arange legs indices
        for j in range(len(edges)):
            Tidx[legs[j]] = edges[j] + 1
            Tstaridx[legs[j]] = edges[j] + 1 + m


        ## add to lists
        T_list.append(cp.copy(T))
        idx_list.append(cp.copy(Tidx))
        T_list.append(cp.copy(Tstar))
        idx_list.append(cp.copy(Tstaridx))
    return T_list, idx_list


def ncon_list_generator_two_site_expectation_with_env_peps_obc(TT, TTstar, Oij, smat, emat, Ek, tensors_list, inside_env, outside_env):
    Oij_flag = 0
    e = smat.shape[1]
    last_edge = 2 * e
    sub_omat = tnf.PEPS_OBC_edge_environment_sub_order_matrix(emat)
    n, m = sub_omat.shape
    if n < m:
        sub_omat = np.transpose(sub_omat)
        n, m = sub_omat.shape
    spins_idx = np.array(range(last_edge, last_edge + len(tensors_list))).reshape(n, m)
    Oij_idx = range(last_edge + len(tensors_list), last_edge + len(tensors_list) + 4)
    t_list = [Oij]
    i_list = [Oij_idx]
    o_list = []
    for i in range(n):
        for j in range(m):

            ## pick a tensor T, T*
            idx = sub_omat[i, j]
            t = TT[idx]
            ts = TTstar[idx]
            edges = np.nonzero(smat[idx, :])[0]
            legs = smat[idx, edges]
            t_idx = [0] * len(t.shape)
            ts_idx = [0] * len(ts.shape)
            t_idx[0] = spins_idx[i, j]
            ts_idx[0] = spins_idx[i, j]

            ## arange legs indices
            for k in range(len(edges)):
                if edges[k] in inside_env:
                    t_idx[legs[k]] = edges[k]
                    ts_idx[legs[k]] = edges[k] + e
                elif edges[k] in outside_env:
                    t_idx[legs[k]] = edges[k]
                    ts_idx[legs[k]] = edges[k]
                if edges[k] == Ek:
                    t_Ek = np.nonzero(smat[:, Ek])[0]
                    if idx == t_Ek[0]:
                        Oij_flag = 1
                        t_idx[0] = Oij_idx[0]
                        ts_idx[0] = Oij_idx[2]
                    if idx == t_Ek[1]:
                        t_idx[0] = Oij_idx[1]
                        ts_idx[0] = Oij_idx[3]

            ## add to lists
            t_list += [t, ts]
            i_list += [t_idx, ts_idx]
            o_list += t_idx
            o_list += ts_idx
            if Oij_flag:
                o_list += Oij_idx
                Oij_flag = 0

    return t_list, i_list, o_list


def ncon_list_generator_two_site_expectation_with_factor_belief_env_peps_obc(Ek, graph, env_size, network_shape, smat, Oij):
    last_edge = smat.shape[1]
    counter_out = 2 * (last_edge + 1)

    # get the environment matrix and the lists of inside and outside edges
    emat = tnf.PEPS_OBC_edge_rect_env(Ek, smat, network_shape, env_size)
    inside, outside = tnf.PEPS_OBC_divide_edge_regions(emat, smat)
    sub_omat = tnf.PEPS_OBC_edge_environment_sub_order_matrix(emat)
    N, M = sub_omat.shape
    if N < M:
        sub_omat = np.transpose(sub_omat)
        N, M = sub_omat.shape
    tensors_indices = sub_omat.ravel()
    Oij_idx = range(2 * (counter_out + len(outside)), 2 * (counter_out + len(outside)) + 4)
    spins_counter = Oij_idx[3] + 1
    Ek_tensors = np.nonzero(smat[:, Ek])[0]

    # make factors and nodes lists
    nodes_out = []
    for n in outside:
        nodes_out.append('n' + str(n))
    factors_list = []
    idx_list = []
    order_list = []
    for i, t in enumerate(tensors_indices):
        f = 'f' + str(t)
        factors_list.append(graph.absorb_message_into_factor_in_env(f, nodes_out))
        idx = [0] * len(factors_list[i].shape)
        idx[0] = spins_counter
        idx[1] = spins_counter
        spins_counter += 1
        edges = np.nonzero(smat[t, :])[0]
        legs = smat[t, edges]
        for l, edge in enumerate(edges):
            if edge in outside:
                idx[2 * legs[l]] = counter_out
                idx[2 * legs[l] + 1] = counter_out
                counter_out += 1
            if edge in inside:
                idx[2 * legs[l]] = edge
                idx[2 * legs[l] + 1] = edge + last_edge
                if (edge == Ek) & (Ek_tensors[0] == t):
                    idx[0] = Oij_idx[0]
                    idx[1] = Oij_idx[2]
                if (edge == Ek) & (Ek_tensors[1] == t):
                    idx[0] = Oij_idx[1]
                    idx[1] = Oij_idx[3]
        idx_list.append(idx)
        order_list += idx
    factors_list.append(Oij)
    idx_list.append(Oij_idx)
    return factors_list, idx_list, order_list


def ncon_list_generator_two_site_expectation_with_factor_belief_env_peps_obc_efficient(Ek, graph, env_size, network_shape, smat, Oij):
    last_edge = smat.shape[1]
    counter_out = 2 * (last_edge + 1)

    # get the environment matrix and the lists of inside and outside edges
    emat = tnf.PEPS_OBC_edge_rect_env(Ek, smat, network_shape, env_size)
    inside, outside = tnf.PEPS_OBC_divide_edge_regions(emat, smat)
    sub_omat = tnf.PEPS_OBC_edge_environment_sub_order_matrix(emat)
    N, M = sub_omat.shape
    if N < M:
        sub_omat = np.transpose(sub_omat)
        N, M = sub_omat.shape
    tensors_indices = sub_omat.ravel()
    Oij_idx = range(2 * (counter_out + len(outside)), 2 * (counter_out + len(outside)) + 4)
    spins_counter = Oij_idx[3] + 1
    Ek_tensors = np.nonzero(smat[:, Ek])[0]

    # make factors and nodes lists
    nodes_out = []
    for n in outside:
        nodes_out.append('n' + str(n))
    factors_list = []
    idx_list = []
    order_list = []
    for i, t in enumerate(tensors_indices):
        f = 'f' + str(t)
        factors_list.append(graph.absorb_message_into_factor_in_env_efficient(f, nodes_out))
        factors_list.append(np.conj(cp.deepcopy(graph.factors[f][1])))

        idx = [0] * len(graph.factors[f][1].shape)
        idx_conj = [0] * len(graph.factors[f][1].shape)
        idx[0] = spins_counter
        idx_conj[0] = spins_counter
        spins_counter += 1
        edges = np.nonzero(smat[t, :])[0]
        legs = smat[t, edges]
        for l, edge in enumerate(edges):
            if edge in outside:
                idx[legs[l]] = counter_out
                idx_conj[legs[l]] = counter_out
                counter_out += 1
            if edge in inside:
                idx[legs[l]] = edge
                idx_conj[legs[l]] = edge + last_edge
                if (edge == Ek) & (Ek_tensors[0] == t):
                    idx[0] = Oij_idx[0]
                    idx_conj[0] = Oij_idx[2]
                if (edge == Ek) & (Ek_tensors[1] == t):
                    idx[0] = Oij_idx[1]
                    idx_conj[0] = Oij_idx[3]
        idx_list.append(idx)
        idx_list.append(idx_conj)
        order_list += idx
        order_list += idx_conj
    factors_list.append(Oij)
    idx_list.append(Oij_idx)
    return factors_list, idx_list, order_list




















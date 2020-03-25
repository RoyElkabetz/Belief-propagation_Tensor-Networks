import numpy as np
import copy as cp
import glassy_gPEPS as su
from scipy import linalg
import matplotlib.pyplot as plt
import ncon_lists_generator as nlg
import ncon
import DEnFG as fg

date = '2019.08.14_'
experiment_num = '_3_'

#---------------------- Tensor Network paramas ------------------

N = 16 # number of spins
L = np.int(np.sqrt(N))



d = 2  # virtual bond dimension
p = 2  # physical bond dimension
D_max = d  # maximal virtual bond dimension
J = 1  # Hamiltonian: interaction coeff
h = np.linspace(0.1, 5., num=50)  # Hamiltonian: magnetic field coeff

mu = 1
sigma = 0
Jk = np.random.normal(mu, sigma, (2 * N))
#Jk = np.ones((2 * N))
print('Jk = ', Jk)
J_prop = 'J = N(' + str(mu) + ',' + str(sigma) + ')_'


time_to_converge = np.zeros((len(h)))
mz_matrix_TN = np.zeros((p, p, len(h)))

E = []
mx = []
mz = []
mx_exact = []
mz_exact = []
mx_graph = []
mz_graph = []
sum_of_trace_distance_exact_graph = []
sum_of_trace_distance_exact_gPEPS = []
sum_of_trace_distance_gPEPS_graph = []
trace_distance_exact_graph = np.zeros((len(h), L, L))
trace_distance_exact_gPEPS = np.zeros((len(h), L, L))
trace_distance_gPEPS_graph = np.zeros((len(h), L, L))


spin_idx_for_trace_distance = 3

mx_mat = np.zeros((len(h), L, L))
mz_mat = np.zeros((len(h), L, L))
mx_mat_exact = np.zeros((len(h), L, L))
mz_mat_exact = np.zeros((len(h), L, L))
mz_mat_graph = np.zeros((len(h), L, L))
mx_mat_graph = np.zeros((len(h), L, L))


pauli_z = np.array([[1, 0], [0, -1]])
pauli_y = np.array([[0, -1j], [1j, 0]])
pauli_x = np.array([[0, 1], [1, 0]])

sz = 0.5 * pauli_z
sy = 0.5 * pauli_y
sx = 0.5 * pauli_x

t_list = [0.1, 0.01, 0.001, 0.0001]
iterations = 100

Opi = pauli_z
Opj = pauli_z
Op_field = pauli_x

#------------- generating the finite PEPS structure matrix------------------
imat = np.zeros((N, 2 * N), dtype=int)
smat = np.zeros((N, 2 * N), dtype=int)
n, m = imat.shape
for i in range(n):
    imat[i, 2 * i] = 1
    imat[i, 2 * i + 1] = 1
    imat[i, 2 * np.mod(i + 1, L) + 2 * L * np.int(np.floor(np.float(i) / np.float(L)))] = 1
    imat[i, 2 * np.mod(i + L, N) + 1] = 1

    smat[i, 2 * i] = 1
    smat[i, 2 * i + 1] = 2
    smat[i, 2 * np.mod(i + 1, L) + 2 * L * np.int(np.floor(np.float(i) / np.float(L)))] = 3
    smat[i, 2 * np.mod(i + L, N) + 1] = 4


for ss in range(h.shape[0]):
    # ------------- generating tensors and bond vectors for each magnetic field ---------------------------

    TT = []
    for ii in range(n):
        TT.append(np.random.rand(p, d, d, d, d))
    LL = []
    for i in range(imat.shape[1]):
        LL.append(np.ones(d, dtype=float) / d)

    counter = 0

    # --------------------------------- iterating the gPEPS algorithm -------------------------------------
    for dt in t_list:
        flag = 0
        for j in range(iterations):
            counter += 2
            print('h, h_idx, t, j = ', h[ss], ss, dt, j)
            TT1, LL1 = su.simple_update(cp.deepcopy(TT), cp.deepcopy(LL), dt, Jk, h[ss], Opi, Opj, Op_field, imat, smat, D_max)
            TT2, LL2 = su.simple_update(cp.deepcopy(TT1), cp.deepcopy(LL1), dt, Jk, h[ss], Opi, Opj, Op_field, imat, smat, D_max)
            energy1 = su.energy_per_site(cp.deepcopy(TT1), cp.deepcopy(LL1), imat, smat, Jk, h[ss], Opi, Opj, Op_field)
            energy2 = su.energy_per_site(cp.deepcopy(TT2), cp.deepcopy(LL2), imat, smat, Jk, h[ss], Opi, Opj, Op_field)
            if np.abs(energy1 - energy2) < 1e-8:
                flag = 1
                TT = cp.deepcopy(TT2)
                LL = cp.deepcopy(LL2)
                break
            else:
                TT = cp.deepcopy(TT2)
                LL = cp.deepcopy(LL2)
        if flag:
            flag = 0
            break

    #------------------------------------- generating the TN dual DEnFG -----------------------------------

    sqrt_LL = []
    for i in range(len(LL)):
        sqrt_LL.append(np.sqrt(LL[i]))

    # Graph initialization
    graph = fg.Graph()

    # Adding virtual nodes
    for i in range(m):
        graph.add_node(D_max, 'n' + str(graph.node_count))
    virtual_nodes_counter = cp.copy(graph.node_count)

    # Adding factors
    for i in range(n):
        # Adding physical node
        graph.add_node(p, 'n' + str(graph.node_count))

        # generating the neighboring nodes of the i'th factor
        neighbor_nodes = {}
        edges = np.nonzero(smat[i, :])[0]
        legs = smat[i, edges]
        neighbor_nodes['n' + str(graph.node_count - 1)] = 0

        for j in range(len(edges)):
            neighbor_nodes['n' + str(edges[j])] = legs[j]

        factor = cp.deepcopy(TT[i])
        for ii in range(len(edges)):
            factor = np.einsum(factor, range(len(factor.shape)), sqrt_LL[edges[ii]], [legs[ii]], range(len(factor.shape)))
        graph.add_factor(neighbor_nodes, cp.deepcopy(factor))

    #----------------------------------- run Belief Propagation over DEnFG ----------------------------
    t_max = 1000
    epsilon = 1e-15
    dumping = 0.1
    graph.sum_product(t_max, epsilon, dumping)
    graph.calc_node_belief()

    # --------------------------------- calculating magnetization matrices -------------------------------
    for l in range(L):
        for ll in range(L):
            spin_index = np.int(L * l + ll)
            T_list_n, idx_list_n = nlg.ncon_list_generator(TT, LL, smat, np.eye(p), spin_index)
            T_listz, idx_listz = nlg.ncon_list_generator(TT, LL, smat, pauli_z, spin_index)
            mz_mat_exact[ss, l, ll] = ncon.ncon(T_listz, idx_listz) / ncon.ncon(T_list_n, idx_list_n)

            T_listx, idx_listx = nlg.ncon_list_generator(TT, LL, smat, pauli_x, spin_index)
            mx_mat_exact[ss, l, ll] = ncon.ncon(T_listx, idx_listx) / ncon.ncon(T_list_n, idx_list_n)

            mz_mat[ss, l, ll] = su.single_tensor_expectation(spin_index, TT, LL, imat, smat, pauli_z)
            mx_mat[ss, l, ll] = su.single_tensor_expectation(spin_index, TT, LL, imat, smat, pauli_x)

            mx_mat_graph[ss, l, ll] = np.real(np.trace(np.matmul(pauli_x, graph.node_belief['n' + str(virtual_nodes_counter + spin_index)])))
            mz_mat_graph[ss, l, ll] = np.real(np.trace(np.matmul(pauli_z, graph.node_belief['n' + str(virtual_nodes_counter + spin_index)])))

            # ------------------------ Trace distances of every spin reduced density matrix results ---------------

            reduced_dm_gPEPS = su.tensor_reduced_dm(spin_index, TT, LL, smat, imat)
            reduced_dm_graph = graph.node_belief['n' + str(virtual_nodes_counter + spin_index)]
            tensors_reduced_dm_list, indices_reduced_dm_list = nlg.ncon_list_generator_reduced_dm(TT, LL, smat, spin_index)
            tensors_reduced_dm_listn, indices_reduced_dm_listn = nlg.ncon_list_generator(TT, LL, smat, np.eye(p), spin_index)
            reduced_dm_exact = ncon.ncon(tensors_reduced_dm_list, indices_reduced_dm_list) / ncon.ncon(tensors_reduced_dm_listn, indices_reduced_dm_listn)

            trace_distance_exact_gPEPS[ss, l, ll] = su.trace_distance(reduced_dm_exact, reduced_dm_gPEPS)
            trace_distance_exact_graph[ss, l, ll] = su.trace_distance(reduced_dm_exact, reduced_dm_graph)
            trace_distance_gPEPS_graph[ss, l, ll] = su.trace_distance(reduced_dm_gPEPS, reduced_dm_graph)


    # ------------------ calculating total magnetization, energy and time to converge -------------------
    mz.append(np.sum(mz_mat[ss, :, :]) / n)
    mx.append(np.sum(mx_mat[ss, :, :]) / n)
    mz_exact.append(np.sum(mz_mat_exact[ss, :, :]) / n)
    mx_exact.append(np.sum(mx_mat_exact[ss, :, :]) / n)
    mz_graph.append(np.sum(mz_mat_graph[ss, :, :]) / n)
    mx_graph.append(np.sum(mx_mat_graph[ss, :, :]) / n)
    time_to_converge[ss] = counter
    E.append(su.energy_per_site(cp.deepcopy(TT), cp.deepcopy(LL), imat, smat, Jk, h[ss], Opi, Opj, Op_field))
    sum_of_trace_distance_exact_graph.append(trace_distance_exact_graph[ss, :, :].sum())
    sum_of_trace_distance_exact_gPEPS.append(trace_distance_exact_gPEPS[ss, :, :].sum())
    sum_of_trace_distance_gPEPS_graph.append(trace_distance_gPEPS_graph[ss, :, :].sum())
    print('Mx_graph, Mz_graph', mx_graph[ss], mz_graph[ss])
    print('Mx_exact, Mz_exact', mx_exact[ss], mz_exact[ss])
    print('E, Mx, Mz: ', E[ss], mx[ss], mz[ss])
    print('\n')

    print('d(exact, DEnFG) = ', sum_of_trace_distance_exact_graph[ss])
    print('d(exact, gPEPS) = ', sum_of_trace_distance_exact_gPEPS[ss])
    print('d(gPEPS, DEnFG) = ', sum_of_trace_distance_gPEPS_graph[ss])

# ------------------------------------- plotting results ----------------------------------------------
file_name_energy = date + 'experiment_#' + experiment_num + 'Energy_' + 'glassy_PEPS_'+ J_prop + str(L) + 'x' + str(L) + '_d-' + str(D_max) +'.pdf'
file_name_magnetization = date + 'experiment_#' + experiment_num + 'Magnetization_' + 'glassy_PEPS_'+ J_prop + str(L) + 'x' + str(L) + '_d-' + str(D_max) +'.pdf'
file_name_TD = date + 'experiment_#' + experiment_num + 'Trace_Distance_' + 'glassy_PEPS_'+ J_prop + str(L) + 'x' + str(L) + '_d-' + str(D_max) +'.pdf'



plt.figure()
plt.title(str(N) + ' spins 2D glassy PEPS (' + J_prop + ') Quantum Ising Model with \n a transverse field and maximal bond dimension d = ' + str(D_max))
plt.subplot()
color = 'tab:red'
plt.xlabel('h')
plt.ylabel('Energy per site', color=color)
plt.plot(h, E, color=color)
plt.grid()
plt.tick_params(axis='y', labelcolor=color)
plt.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
plt.ylabel('# of gPEPS iterations until convergence', color=color)  # we already handled the x-label with ax1
plt.plot(h, time_to_converge,color=color)
plt.tick_params(axis='y', labelcolor=color)
plt.tight_layout()  # otherwise the right y-label is slightly clipped
plt.grid()
plt.savefig(file_name_energy, bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(h, mx, 'go', markersize=3)
plt.plot(h, np.abs(np.array(mz)), 'bo', markersize=3)
plt.plot(h, mx_exact, 'r-', linewidth=2)
plt.plot(h, np.abs(np.array(mz_exact)), 'y-', linewidth=2)
plt.plot(h, mx_graph, 'cv', markersize=5)
plt.plot(h, np.abs(np.array(mz_graph)), 'mv', markersize=5)
plt.title('Averaged magnetization vs h at d = ' + str(D_max) + ' in a ' + str(L) + 'x' + str(L) + ' PEPS')
plt.xlabel('h')
plt.ylabel('Magnetization')
plt.legend(['mx', '|mz|', 'mx exact', '|mz| exact', 'mx DEnFG', '|mz| DEnFG'])
plt.grid()
plt.savefig(file_name_magnetization, bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(h, sum_of_trace_distance_exact_graph, 'o')
plt.plot(h, sum_of_trace_distance_exact_gPEPS, 'v')
#plt.plot(h, trace_distance_gPEPS_graph, 'o')
plt.title('Total Trace Distance comparison of all particles rdms in a ' + str(L) + 'x' + str(L) + ' PEPS')
plt.xlabel('h')
plt.ylabel('Trace distance')
#plt.legend(['d(exact, DEnFG)', 'd(exact, gPEPS)', 'd(gPEPS, DEnFG)'])
plt.legend(['D(exact, BP)', 'D(exact, simple-update)'])
plt.grid()
plt.savefig(file_name_TD, bbox_inches='tight')
plt.show()

'''
plt.figure()
plt.plot(h[60:], mx[60:], 'go', markersize=3)
plt.plot(h[60:], mx_exact[60:], 'r-', linewidth=2)
plt.plot(h[60:], mx_graph[60:], 'cv', markersize=5)
plt.title('Averaged magnetization vs h at Dmax = ' + str(D_max))
plt.xlabel('h')
plt.ylabel('Magnetization')
plt.legend(['mx', 'mx exact', 'mx DEnFG'])
plt.savefig("magnetization_zoomed_with_legend" + file_name, bbox_inches='tight')
plt.grid()
plt.show()
'''

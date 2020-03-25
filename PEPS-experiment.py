import numpy as np
import copy as cp
import gPEPS as su
from scipy import linalg
import matplotlib.pyplot as plt
import ncon_lists_generator as nlg
import ncon
import DEnFG as fg

#---------------------- Tensor Network paramas ------------------
N = 9 # number of spins
L = np.int(np.sqrt(N))


d = 2  # virtual bond dimension
p = 2  # physical bond dimension
D_max = d  # maximal virtual bond dimension
J = -1  # Hamiltonian: interaction coeff
h = np.linspace(0.1, 5., num=50)  # Hamiltonian: magnetic field coeff

time_to_converge = np.zeros((len(h)))
mz_matrix_TN = np.zeros((p, p, len(h)))

E = []
mx = []
mz = []
mx_exact = []
mz_exact = []
mx_graph = []
mz_graph = []

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
iterations = 200


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
    # ------------- generating tensors and bond vectors for each magnetic field---------------------------

    TT = []
    for ii in range(n):
        TT.append(np.random.rand(p, d, d, d, d))
    LL = []
    for i in range(imat.shape[1]):
        LL.append(np.ones(d, dtype=float) / d)

    hij = -J * np.kron(pauli_z, pauli_z) - 0.25 * h[ss] * (np.kron(np.eye(p), pauli_x) + np.kron(pauli_x, np.eye(p)))
    hij_energy_operator = np.reshape(cp.deepcopy(hij), (p, p, p, p))
    hij = np.reshape(hij, [p ** 2, p ** 2])
    unitary = [np.reshape(linalg.expm(-t_list[t] * hij), [p, p, p, p]) for t in range(len(t_list))]
    counter = 0

    # --------------------------------- iterating the gPEPS algorithm -------------------------------------
    for i in range(len(t_list)):
        flag = 0
        for j in range(iterations):
            counter += 2
            print('h, i, j = ', h[ss], ss, i, j)
            TT1, LL1 = su.simple_update(cp.deepcopy(TT), cp.deepcopy(LL), unitary[i], imat, smat, D_max)
            TT2, LL2 = su.simple_update(cp.deepcopy(TT1), cp.deepcopy(LL1), unitary[i], imat, smat, D_max)
            energy1 = su.energy_per_site(cp.deepcopy(TT1), cp.deepcopy(LL1), imat, smat, hij_energy_operator)
            energy2 = su.energy_per_site(cp.deepcopy(TT2), cp.deepcopy(LL2), imat, smat, hij_energy_operator)
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

        # generating the neighboring nodes off the i'th factor
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

    #----------------------------------- run Belief Propagation over DEnFG----------------------------
    t_max = 400
    epsilon = 1e-15
    graph.sum_product(t_max, epsilon)
    graph.calc_node_belief()

    # --------------------------------- calculating magnetization matrices-------------------------------
    for l in range(L):
        for ll in range(L):
            print('l, ll = ', l, ll)

            #T_list_n, idx_list_n = nlg.ncon_list_generator(TT, LL, smat, np.eye(p), np.int(L * l + ll))
            #T_listz, idx_listz = nlg.ncon_list_generator(TT, LL, smat, pauli_z, np.int(L * l + ll))
            #mz_mat_exact[ss, l, ll] = ncon.ncon(T_listz, idx_listz) / ncon.ncon(T_list_n, idx_list_n)

            #T_listx, idx_listx = nlg.ncon_list_generator(TT, LL, smat, pauli_x, np.int(L * l + ll))
            #mx_mat_exact[ss, l, ll] = ncon.ncon(T_listx, idx_listx) / ncon.ncon(T_list_n, idx_list_n)

            mz_mat[ss, l, ll] = su.single_tensor_expectation(np.int(2 * l + ll), TT, LL, imat, smat, pauli_z)
            mx_mat[ss, l, ll] = su.single_tensor_expectation(np.int(2 * l + ll), TT, LL, imat, smat, pauli_x)

            mx_mat_graph[ss, l, ll] = np.real(np.trace(np.matmul(pauli_x, graph.node_belief['n' + str(virtual_nodes_counter + 2 * l + ll)])))
            mz_mat_graph[ss, l, ll] = np.real(np.trace(np.matmul(pauli_z, graph.node_belief['n' + str(virtual_nodes_counter + 2 * l + ll)])))

    # ------------------ calculating total magnetization, energy and time to converge -------------------
    mz.append(np.sum(mz_mat[ss, :, :]) / n)
    mx.append(np.sum(mx_mat[ss, :, :]) / n)
    #mz_exact.append(np.sum(mz_mat_exact[ss, :, :]) / n)
    #mx_exact.append(np.sum(mx_mat_exact[ss, :, :]) / n)
    mz_graph.append(np.sum(mz_mat_graph[ss, :, :]) / n)
    mx_graph.append(np.sum(mx_mat_graph[ss, :, :]) / n)
    time_to_converge[ss] = counter
    E.append(su.energy_per_site(cp.deepcopy(TT), cp.deepcopy(LL), imat, smat, hij_energy_operator))
    print('Mx_graph, Mz_graph', mx_graph[ss], mz_graph[ss])
    #print('Mx_exact, Mz_exact', mx_exact[ss], mz_exact[ss])
    print('E, Mx, Mz: ', E[ss], mx[ss], mz[ss])

# ------------------------------------- plotting results ----------------------------------------------
file_name = 'PEPS' + str(N) + '.pdf'

plt.figure()
plt.title(str(N) + 'spins 2D Quantum Ising Model with a transverse field \n with maximal virtual bond dimension Dmax = ' + str(D_max))
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
plt.plot(h, time_to_converge, color=color)
plt.tick_params(axis='y', labelcolor=color)
plt.tight_layout()  # otherwise the right y-label is slightly clipped
plt.grid()
plt.savefig("energy" + file_name, bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(h, mx, 'go', markersize=3)
plt.plot(h, np.abs(np.array(mz)), 'bo', markersize=3)
#plt.plot(h, mx_exact, 'r-', linewidth=2)
#plt.plot(h, np.abs(np.array(mz_exact)), 'y-', linewidth=2)
plt.plot(h, mx_graph, 'cv', markersize=5)
plt.plot(h, np.abs(np.array(mz_graph)), 'mv', markersize=5)
plt.title('Averaged magnetization vs h at Dmax = ' + str(D_max))
plt.xlabel('h')
plt.ylabel('Magnetization')
#plt.legend(['mx', '|mz|', 'mx exact', '|mz| exact', 'mx DEnFG', '|mz| DEnFG'])
plt.legend(['mx', '|mz|', 'mx DEnFG', '|mz| DEnFG'])
plt.grid()
plt.savefig("magnetization" + file_name, bbox_inches='tight')
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
